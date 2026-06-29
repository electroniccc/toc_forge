"""LLM-based TOC extraction strategies."""

import base64
import json
import logging
import os
import re

import cv2
import httpx
from openai import OpenAI

from .ocr_engine import ocr_toc_pages

logger = logging.getLogger(__name__)

_TOC_LLM_SYSTEM_PROMPT = (
    "You are a table-of-contents parser for a Chinese academic textbook. "
    "The input is a JSON array where each element has:\n"
    '- "page": PDF page index (0-based)\n'
    '- "lines": array of text strings from that TOC page, in reading order. '
    "Each line is a TOC entry (title) optionally followed by its page number.\n\n"
    "How to determine hierarchy from text patterns:\n"
    '- Top level (chapters/parts): "第X章", "第X篇", or standalone "Chapter X"\n'
    '- Second level (sections): "第一节/第二节/…", "X.Y" numbered sections\n'
    '- Third level (subsections): "一、", "二、", "（一）", "1.", "1）"\n'
    '- Exercises: "习题X-Y" nest under their matching section; "总习题X" under the chapter\n'
    "- Entries with deeper numbering (e.g. 1.1.1) nest under their prefix (1.1)\n\n"
    "Rules:\n"
    '- Each output node: "title" (str), "page_num" (int or null), "children" (list of nodes)\n'
    "- The last number on a line is the page number; remove it from the title\n"
    "- Drop trailing dot-leaders (…, ...) from titles\n"
    "- Join fragmented titles that clearly belong together (e.g. a chapter title split across lines)\n"
    '- Preserve the full title text including chapter/section numbers ("第1章", "1.1")\n'
    "- If a page number is missing on a heading, inherit from its first child\n"
    '- Output a JSON object: {"toc": [...]}  -- no markdown fences, no extra text.'
)

_TOC_VLLM_SYSTEM_PROMPT = (
    "You extract table-of-contents from book page images. Output a JSON tree: "
    '{"toc": [node, ...]} where each node has "title" (str), "page_num" (int or null), '
    '"children" (list of nodes).\n\n'
    "How to detect hierarchy:\n"
    "- Chapters (top-level): largest font, bold, leftmost alignment, often start with "
    '"第X章", "Chapter", or a standalone number like "1.", "1 ".\n'
    '- Sections: smaller font or indented rightwards, start with "X.Y", "X.Y.Z", "一、", '
    '"（一）", "§", or just a title indented under a chapter.\n'
    "- Subsections nest under the nearest preceding higher-level entry.\n\n"
    "Rules:\n"
    '- Preserve the full title text including chapter/section numbers ("第1章", "1.1", etc.).\n'
    "- Page numbers are the rightmost numbers on the same line. If missing, inherit from children.\n"
    '- Ignore running headers/footers and standalone "目录"/"Contents" headings.\n'
    "- Do NOT invent entries not visible in the images."
)


def _build_llm_client(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        timeout=httpx.Timeout(300.0, read=300.0, write=60.0, connect=30.0),
    )


def _call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str | list[dict],
) -> dict | list:
    logger.info("Calling LLM model=%s ...", model)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        logger.info("LLM response length=%d, preview=%s", len(raw), raw[:200])
        # Strip markdown fences if present
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        raise


def _simplify_ocr_for_llm(toc_results: list[dict]) -> list[dict]:
    """Strip bounding boxes from OCR results, keeping only text lines per page."""
    simplified = []
    for tp in toc_results:
        page_idx = tp.get("page", 0)
        page_lines = []
        for cb in tp["content_boxes"]:
            # Collect text fragments with their y-ranges for this content box only
            items = []
            for text, box in zip(cb["rec_texts"], cb["rec_boxes"]):
                t = str(text).strip()
                if not t:
                    continue
                items.append({"text": t, "ymin": box[1], "ymax": box[3]})

            if not items:
                continue

            # Group into lines by y-overlap (within this content box)
            items.sort(key=lambda x: x["ymin"])
            line_groups = []
            for item in items:
                placed = False
                for group in line_groups:
                    g_ymin = sum(g["ymin"] for g in group) / len(group)
                    g_ymax = sum(g["ymax"] for g in group) / len(group)
                    if max(item["ymin"], g_ymin) < min(item["ymax"], g_ymax):
                        group.append(item)
                        placed = True
                        break
                if not placed:
                    line_groups.append([item])

            for group in line_groups:
                group.sort(key=lambda x: x["ymin"])
                line_text = " ".join(g["text"] for g in group)
                page_lines.append(line_text)

        simplified.append({"page": page_idx, "lines": page_lines})

    return simplified


def build_toc_llm(
    toc_pages: list[dict],
    page_imgs,
    ocr_model,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
) -> list[dict]:
    """Run local OCR then call a text LLM to build the TOC tree."""
    toc_results = ocr_toc_pages(
        toc_pages,
        page_imgs,
        ocr_model,
        do_debug=do_debug,
        output=output,
        cache_dir=cache_dir,
        pdf_hash=pdf_hash,
    )

    simplified = _simplify_ocr_for_llm(toc_results)
    ocr_json = json.dumps(simplified, ensure_ascii=False)
    logger.info(
        "LLM input: %d pages, %d lines, %d chars",
        len(simplified),
        sum(len(p["lines"]) for p in simplified),
        len(ocr_json),
    )
    if do_debug:
        debug_path = os.path.join(output, "toc_llm_input.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(ocr_json)

    client = _build_llm_client(
        model=llm_model, api_key=llm_api_key, base_url=llm_base_url
    )
    model = llm_model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    result = _call_llm(
        client,
        model,
        _TOC_LLM_SYSTEM_PROMPT,
        [
            {"type": "text", "text": ocr_json},
        ],
    )
    return result.get("toc", []) if isinstance(result, dict) else result


def build_toc_vllm(
    toc_pages: list[dict],
    page_imgs,
    do_debug: bool = False,
    output: str = "output",
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
) -> list[dict]:
    """Build TOC by sending page images to a vision LLM, skipping local OCR."""
    client = _build_llm_client(
        model=llm_model, api_key=llm_api_key, base_url=llm_base_url
    )
    model = llm_model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    content: list[dict] = []
    for tp in toc_pages:
        page_idx = tp["page"]
        img = page_imgs[page_idx]
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
            }
        )
    content.append(
        {
            "type": "text",
            "text": 'Return a JSON object like: {"toc": [{"title": "...", "page_num": 1, "children": [...]}, ...]}',
        }
    )
    result = _call_llm(client, model, _TOC_VLLM_SYSTEM_PROMPT, content)
    return result.get("toc", []) if isinstance(result, dict) else result
