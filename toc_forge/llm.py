"""LLM-based TOC extraction strategies."""

import base64
import json
import logging
import os
import re

import cv2
from openai import OpenAI

from .ocr_engine import ocr_toc_pages
from .utils import NumpyEncoder

logger = logging.getLogger(__name__)

_TOC_LLM_SYSTEM_PROMPT = (
    "You are a table-of-contents parser. The input is a JSON array of OCR results "
    'from a book\'s TOC pages. Each element has: "page" (PDF page index), '
    '"content_boxes" (array of detected TOC regions). Each content_box has: '
    '"rec_texts" (OCR text fragments on that line), "rec_boxes" (bounding box '
    '[x1,y1,x2,y2] for each fragment), "content_box" with a "coordinate" '
    "[x1,y1,x2,y2] of the containing region.\n\n"
    "Rules:\n"
    '- Each output node: "title" (str), "page_num" (int or null), "children" (list of nodes).\n'
    "- Use rec_boxes x1 for indentation: smaller x1 = higher level, larger x1 = deeper nesting.\n"
    "- Fragments at the far right (large x1, near the content_box right edge) are page numbers.\n"
    "- Consecutive fragments with similar x1 that belong to the same title should be joined.\n"
    "- Drop trailing dot-leaders from titles.\n"
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

    ocr_json = json.dumps(toc_results, ensure_ascii=False, cls=NumpyEncoder)
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
