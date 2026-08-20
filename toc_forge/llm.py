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
from .utils import _cache_load, _cache_path, _cache_save

logger = logging.getLogger(__name__)

_TOC_LLM_SYSTEM_PROMPT = (
    "You are a table-of-contents parser for any document (book, textbook, thesis, "
    "report, manual, etc.) in any language (English, Chinese, and others). "
    "The input is a JSON array where each element has:\n"
    '- "page": PDF page index (0-based)\n'
    '- "lines": array of text strings from that TOC page, in reading order. '
    "A line usually contains one TOC entry, but compact layouts may place several "
    "independent entries on the same physical line.\n\n"
    "How to determine hierarchy from text patterns (language-agnostic):\n"
    "- Top level: parts/chapters/front-back matter, e.g. '第X篇', '第X章', 'Part X', "
    "'Chapter X', 'Preface', 'Introduction', 'Appendix', 'Bibliography', 'Index', or "
    "the coarsest repeated numbering scheme ('1', 'I', '一').\n"
    "- Second level: sections, e.g. '第一节', 'X.Y', '§X', or entries whose numbering "
    "is one level deeper than the top level.\n"
    "- Third level: subsections, e.g. '一、', '（一）', '1.', '1）', 'X.Y.Z', 'A.', '(a)', "
    "or any entry whose numbering has a deeper prefix than its parent.\n"
    "- Special entries (exercises, sub-appendices, etc., e.g. '习题X-Y', 'Appendix A.1') "
    "nest under their matching section/chapter when a clear prefix matches.\n"
    "- Entries with deeper numbering (e.g. 1.1.1) nest under their prefix (1.1).\n"
    "- Entries sharing the same numbering scheme or the same level of detail belong to "
    "the same level; follow the document's dominant scheme rather than a fixed vocabulary.\n\n"
    "Rules:\n"
    '- Each output node: "title" (str), "page_num" (int or null), "children" (list of nodes)\n'
    "- IMPORTANT: a page-number marker is metadata, never part of the title. It may "
    "be a trailing plain number or a number in parentheses, using either half-width "
    "or full-width punctuation, such as 'Title 12', 'Title (12)', or '标题（12）'. "
    "Put the number only in page_num and always remove the entire marker, including "
    "its parentheses, from title. If page_num is 12, title must not end in '12', "
    "'(12)', or '（12）'.\n"
    "- If one physical/OCR line contains repeated title-plus-page-number groups, split "
    "them into separate sibling nodes in source order instead of treating the whole "
    "line as one title. Required Chinese output example: '一、映射(1) 二、函数(3) "
    "习题 1–1(16)' becomes "
    '[{"title":"一、映射","page_num":1,"children":[]},'
    '{"title":"二、函数","page_num":3,"children":[]},'
    '{"title":"习题 1–1","page_num":16,"children":[]}]. '
    "Required English output example: 'Limits (105) Derivatives (125) Exercises 2.1 "
    "(140)' becomes "
    '[{"title":"Limits","page_num":105,"children":[]},'
    '{"title":"Derivatives","page_num":125,"children":[]},'
    '{"title":"Exercises 2.1","page_num":140,"children":[]}].\n'
    "- Drop trailing dot-leaders (…, ...) from titles\n"
    "- Join fragmented titles that clearly belong together (e.g. a chapter title split across lines)\n"
    '- Preserve the complete title text, excluding page-number markers, including '
    'chapter/section numbers in the original language ("第1章", "Chapter 1", "1.1").\n'
    "- If a page number is missing on a heading, inherit from its first child\n"
    "- PDF bookmarks cannot render LaTeX. Never use $...$ in titles. Always write\n"
    "  math in plain Unicode: Greek symbols (α, β, λ, ∫, ∑, ∇, Δ, ∞, ℏ, ±, →, ×, ∂),\n"
    "  superscripts (x², zⁿ, eⁱˣ), subscripts (x₁, aₙ), and simple expressions\n"
    "  (w=zⁿ, a+b=0, f(z)=u+iv). Bare variables (z, w, f) need no markup at all.\n"
    "- Only when absolutely no Unicode rendering is possible, write the expression\n"
    "  in plain descriptive text (e.g. 'the function w equals z superscript n'),\n"
    "  never with $ markers.\n"
    '- Output a JSON object: {"toc": [...]}  -- no markdown fences, no extra text.'
)

_TOC_VLLM_SYSTEM_PROMPT = (
    "You extract table-of-contents from page images of a document (book, textbook, "
    "thesis, report, manual, etc.) in any language (English, Chinese, and others). "
    "Output a JSON tree: "
    '{"toc": [node, ...]} where each node has "title" (str), "page_num" (int or null), '
    '"children" (list of nodes).\n\n'
    "How to detect hierarchy (visual cues, independent of language):\n"
    "- Top level: largest font, bold, leftmost alignment; parts/chapters/front-back "
    "matter such as '第X篇', '第X章', 'Part X', 'Chapter X', 'Preface', 'Appendix', or "
    "a standalone number ('1.', 'I.', '一').\n"
    "- Sections: smaller font or indented rightwards; numbering like 'X.Y', '第一节', "
    "'§X', or a title indented under a chapter.\n"
    "- Subsections: further indented or more deeply numbered ('X.Y.Z', '一、', '(a)', "
    "'1.'); nest under the nearest preceding higher-level entry.\n"
    "- Entries at the same indentation and with the same numbering scheme belong to "
    "the same level.\n\n"
    "Rules:\n"
    '- Preserve the complete title text, excluding page-number markers, including '
    'chapter/section numbers in the original language ("第1章", "Chapter 1", "1.1").\n'
    "- IMPORTANT: a page-number marker is metadata, never part of the title. It may "
    "be a trailing plain number or a number in parentheses, using either half-width "
    "or full-width punctuation, such as 'Title 12', 'Title (12)', or '标题（12）'. "
    "Put the number only in page_num and always remove the entire marker, including "
    "its parentheses, from title. If page_num is 12, title must not end in '12', "
    "'(12)', or '（12）'.\n"
    "- Compact TOC layouts may put several independent entries on one physical row. "
    "When a row contains repeated title-plus-page-number groups, split them into "
    "separate sibling nodes in visual/source order; never keep the entire row as one "
    "title. Required Chinese output example: '一、映射(1) 二、函数(3) 习题 1–1(16)' "
    "becomes "
    '[{"title":"一、映射","page_num":1,"children":[]},'
    '{"title":"二、函数","page_num":3,"children":[]},'
    '{"title":"习题 1–1","page_num":16,"children":[]}]. '
    "Required English output example: 'Limits (105) Derivatives (125) Exercises 2.1 "
    "(140)' becomes "
    '[{"title":"Limits","page_num":105,"children":[]},'
    '{"title":"Derivatives","page_num":125,"children":[]},'
    '{"title":"Exercises 2.1","page_num":140,"children":[]}].\n'
    "- If a heading has no page number, inherit it from its first child.\n"
    '- Ignore running headers/footers and standalone TOC headings such as "目录", '
    '"Contents", "Table of Contents".\n'
    "- For mathematical and physical symbols in titles (Greek letters, operators, arrows, "
    "special constants, etc.), use Unicode directly — **do NOT use $...$ LaTeX.** "
    "PDF bookmarks display the raw source and cannot render LaTeX.\n"
    "  • Greek/math symbols: α, β, λ, ∫, ∑, ∇, Δ, ∞, ℏ, °, ±, →, ×, ∂, …\n"
    "  • Superscripts: Unicode superscript digits/letters (e.g. x², zⁿ, eⁱˣ)\n"
    "  • Subscripts: Unicode subscript digits/letters (e.g. x₁, aₙ)\n"
    "  • Simple expressions: w=zⁿ, a+b=0, f(z)=u+iv — just plain Unicode\n"
    '  • Bare variables (z, w, f) need NO markup — write them as-is\n'
    "- Only if absolutely impossible in Unicode, write the expression as plain\n"
    "descriptive text without $ markers.\n"
    "- Do NOT invent entries not visible in the images."
)


def _build_llm_client(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 600.0,
    max_retries: int = 0,
) -> OpenAI:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("llm_base_url is required")
    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
        base_url=base_url,
        timeout=httpx.Timeout(timeout, read=timeout, write=60.0, connect=30.0),
        max_retries=max_retries,
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
            extra_body={"enable_thinking": False},
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
                items.append(
                    {
                        "text": t,
                        "xmin": box[0],
                        "ymin": box[1],
                        "ymax": box[3],
                    }
                )

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
                group.sort(key=lambda x: x["xmin"])
                line_text = " ".join(g["text"] for g in group)
                page_lines.append(line_text)

        simplified.append({"page": page_idx, "lines": page_lines})

    return simplified


def _toc_tree_cache_path(
    cache_dir: str,
    pdf_hash: str,
    toc_strategy: str,
) -> str:
    return _cache_path(cache_dir, pdf_hash, f"toc_tree_{toc_strategy}")


def _load_toc_tree_cache(
    cache_dir: str | None,
    pdf_hash: str | None,
    toc_strategy: str,
) -> list[dict] | None:
    """Return the cached TOC tree for this document/strategy, or None."""
    if not cache_dir or not pdf_hash:
        return None
    cache_path = _toc_tree_cache_path(cache_dir, pdf_hash, toc_strategy)
    cached = _cache_load(cache_path)
    if cached is not None:
        logger.info(
            "Loaded cached TOC tree from %s, skipping LLM call", cache_path
        )
    return cached


def _save_toc_tree_cache(
    cache_dir: str | None,
    pdf_hash: str | None,
    toc_strategy: str,
    toc_tree: list[dict],
) -> None:
    """Persist the TOC tree so repeat runs skip the LLM call."""
    if not cache_dir or not pdf_hash:
        return
    cache_path = _toc_tree_cache_path(cache_dir, pdf_hash, toc_strategy)
    _cache_save(cache_path, toc_tree)
    logger.info("Saved TOC tree to cache: %s", cache_path)


# -----------------------------------------------------------------
# LaTeX → Unicode post-processor
# LLMs sometimes ignore "no $...$" instructions.  This walks the
# returned TOC tree and converts any remaining inline LaTeX to the
# closest Unicode equivalent so that PDF bookmarks are readable.
# -----------------------------------------------------------------

# fmt: off
_LATEX_CMD_MAP: dict[str, str] = {
    # Greek lowercase
    "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
    "\\epsilon": "ε", "\\varepsilon": "ε", "\\zeta": "ζ", "\\eta": "η",
    "\\theta": "θ", "\\vartheta": "ϑ", "\\iota": "ι", "\\kappa": "κ",
    "\\lambda": "λ", "\\mu": "μ", "\\nu": "ν", "\\xi": "ξ",
    "\\omicron": "ο", "\\pi": "π", "\\varpi": "ϖ", "\\rho": "ρ",
    "\\varrho": "ϱ", "\\sigma": "σ", "\\varsigma": "ς", "\\tau": "τ",
    "\\upsilon": "υ", "\\phi": "φ", "\\varphi": "ϕ", "\\chi": "χ",
    "\\psi": "ψ", "\\omega": "ω",
    # Greek uppercase
    "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ",
    "\\Xi": "Ξ", "\\Pi": "Π", "\\Sigma": "Σ", "\\Upsilon": "Υ",
    "\\Phi": "Φ", "\\Psi": "Ψ", "\\Omega": "Ω",
    # Math operators / symbols
    "\\infty": "∞", "\\int": "∫", "\\iint": "∬", "\\iiint": "∭",
    "\\oint": "∮", "\\sum": "∑", "\\prod": "∏", "\\coprod": "∐",
    "\\partial": "∂", "\\nabla": "∇", "\\times": "×", "\\cdot": "·",
    "\\pm": "±", "\\mp": "∓", "\\div": "÷", "\\bullet": "•",
    "\\leq": "≤", "\\geq": "≥", "\\neq": "≠", "\\approx": "≈",
    "\\equiv": "≡", "\\sim": "∼", "\\simeq": "≃", "\\cong": "≅",
    "\\propto": "∝", "\\parallel": "∥", "\\perp": "⊥",
    "\\to": "→", "\\rightarrow": "→", "\\leftarrow": "←",
    "\\Rightarrow": "⇒", "\\Leftrightarrow": "⇔",
    "\\mapsto": "↦", "\\iff": "⇔",
    "\\forall": "∀", "\\exists": "∃", "\\nexists": "∄",
    "\\in": "∈", "\\notin": "∉", "\\ni": "∋",
    "\\subset": "⊂", "\\supset": "⊃", "\\subseteq": "⊆",
    "\\supseteq": "⊇", "\\cup": "∪", "\\cap": "∩",
    "\\emptyset": "∅", "\\varnothing": "∅",
    "\\hbar": "ℏ", "\\hslash": "ℏ", "\\ell": "ℓ",
    "\\Re": "ℜ", "\\Im": "ℑ", "\\aleph": "ℵ",
    "\\angle": "∠", "\\measuredangle": "∡",
    "\\square": "□", "\\triangle": "△", "\\nabla": "∇",
    "\\infty": "∞", "\\propto": "∝",
    "\\star": "⋆", "\\ast": "∗", "\\circ": "∘",
    "\\oplus": "⊕", "\\ominus": "⊖", "\\otimes": "⊗",
    "\\oslash": "⊘", "\\odot": "⊙",
    "\\land": "∧", "\\lor": "∨", "\\lnot": "¬",
    "\\langle": "⟨", "\\rangle": "⟩",
    "\\lceil": "⌈", "\\rceil": "⌉",
    "\\lfloor": "⌊", "\\rfloor": "⌋",
    # Non-math LaTeX that may appear
    "\\textregistered": "®", "\\copyright": "©", "\\trademark": "™",
    "\\dag": "†", "\\ddag": "‡", "\\S": "§", "\\P": "¶",
    "\\dots": "…", "\\ldots": "…", "\\cdots": "⋯",
}

_SANS_SERIF_MAP: dict[str, str] = {
    "\\mathrm": "", "\\textrm": "", "\\text": "", "\\mathrm": "",
    "\\mathbf": "", "\\mathbf": "", "\\mathit": "", "\\mathsf": "",
    "\\mathtt": "", "\\mathcal": "", "\\mathfrak": "", "\\mathbb": "",
    "\\mathscr": "", "\\mathnormal": "",
}

_SUPERSCRIPT_MAP: dict[str, str] = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
    "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ",
    "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ",
    "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ",
    "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
    "A": "ᴬ", "B": "ᴮ", "D": "ᴰ", "E": "ᴱ",
    "G": "ᴳ", "H": "ᴴ", "I": "ᴵ", "J": "ᴶ",
    "K": "ᴷ", "L": "ᴸ", "M": "ᴹ", "N": "ᴺ",
    "O": "ᴼ", "P": "ᴾ", "R": "ᴿ", "T": "ᵀ",
    "U": "ᵁ", "V": "ⱽ", "W": "ᵂ",
}

_SUBSCRIPT_MAP: dict[str, str] = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ",
}
# fmt: on

# Longest-commands-first for substitution ordering
_LATEX_CMDS_SORTED: list[tuple[str, str]] = sorted(
    _LATEX_CMD_MAP.items(), key=lambda kv: -len(kv[0])
)

_STRIP_WRAPPERS: list[tuple[str, str]] = [
    (rf"\{n}\{{", "{") for n in _SANS_SERIF_MAP
] + [(rf"\{n}\b", "") for n in _SANS_SERIF_MAP]


def _sanitize_math_in_title(title: str) -> str:
    """Convert inline LaTeX ``$...$`` fragments in *title* to the closest
    Unicode equivalent, then strip the ``$`` fences.  Returns the title with
    the math part replaced, or the original title if no ``$`` pairs exist."""
    _re_dollar = re.compile(r"\$(.+?)\$")

    def _convert(m: re.Match) -> str:
        body = m.group(1)

        # 1. Strip \mathrm{}, \textrm{}, \mathbf{} etc. — keep inner text
        for pat, repl in _STRIP_WRAPPERS:
            body = re.sub(pat, repl, body)
        # Also handle balanced {…} after these commands
        # e.g. \mathrm{e} → {e} after step above, strip those braces
        body = re.sub(r"\{([^{}]+)\}", r"\1", body)

        # 2. LaTeX commands → Unicode (longest match first)
        for cmd, uni in _LATEX_CMDS_SORTED:
            body = body.replace(cmd, uni)

        # 3. Superscript: ^x → Unicode superscript
        def _sup_repl(m2: re.Match) -> str:
            ch = m2.group(1)
            return _SUPERSCRIPT_MAP.get(ch, "^" + ch)
        body = re.sub(r"\^(\S)", _sup_repl, body)

        # 4. Subscript: _x → Unicode subscript
        def _sub_repl(m2: re.Match) -> str:
            ch = m2.group(1)
            return _SUBSCRIPT_MAP.get(ch, "_" + ch)
        body = re.sub(r"_(?=[^\s\\])(\S)", _sub_repl, body)

        # 5. Remove leftover backslash-commands we didn't recognise
        body = re.sub(r"\\[a-zA-Z]+", "", body)
        # Remove leftover braces
        body = body.replace("{", "").replace("}", "")

        # 6. If nothing recognisable was inside (pure text), just return it
        return body.strip()

    if "$" not in title:
        return title
    return _re_dollar.sub(_convert, title)


def _sanitize_toc_tree(toc_tree: list[dict]) -> list[dict]:
    """Walk the TOC tree and sanitise math in every title."""
    for node in toc_tree:
        node["title"] = _sanitize_math_in_title(node.get("title", ""))
        if "children" in node:
            _sanitize_toc_tree(node["children"])
    return toc_tree


def build_toc_llm(
    toc_pages: list[dict],
    page_imgs,
    ocr_model,
    llm_model: str,
    llm_base_url: str,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
    llm_api_key: str | None = None,
    no_toc_cache: bool = False,
    llm_timeout: float = 600.0,
) -> list[dict]:
    """Run local OCR then call a text LLM to build the TOC tree.

    The resulting tree is cached as ``toc_tree_llm.json`` per document; a
    cache hit skips the LLM call entirely (and the OCR it depends on).
    Set ``no_toc_cache`` to force a re-call even when a cache exists.
    """
    if not isinstance(llm_model, str) or not llm_model.strip():
        raise ValueError("llm_model is required")
    if not isinstance(llm_base_url, str) or not llm_base_url.strip():
        raise ValueError("llm_base_url is required")
    if not no_toc_cache:
        cached = _load_toc_tree_cache(cache_dir, pdf_hash, "llm")
        if cached is not None:
            return _sanitize_toc_tree(cached)

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
        base_url=llm_base_url, api_key=llm_api_key,
        timeout=llm_timeout,
    )
    result = _call_llm(
        client,
        llm_model,
        _TOC_LLM_SYSTEM_PROMPT,
        [
            {"type": "text", "text": ocr_json},
        ],
    )
    toc_tree = result.get("toc", []) if isinstance(result, dict) else result
    toc_tree = _sanitize_toc_tree(toc_tree)
    _save_toc_tree_cache(cache_dir, pdf_hash, "llm", toc_tree)
    return toc_tree


def build_toc_vllm(
    toc_pages: list[dict],
    page_imgs,
    llm_model: str,
    llm_base_url: str,
    do_debug: bool = False,
    output: str = "output",
    llm_api_key: str | None = None,
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
    no_toc_cache: bool = False,
    llm_timeout: float = 600.0,
) -> list[dict]:
    """Build TOC by sending page images to a vision LLM, skipping local OCR.

    The resulting tree is cached as ``toc_tree_vllm.json`` per document; a
    cache hit skips the vision-LLM call entirely. Set ``no_toc_cache`` to
    force a re-call even when a cache exists.
    """
    if not isinstance(llm_model, str) or not llm_model.strip():
        raise ValueError("llm_model is required")
    if not isinstance(llm_base_url, str) or not llm_base_url.strip():
        raise ValueError("llm_base_url is required")
    if not no_toc_cache:
        cached = _load_toc_tree_cache(cache_dir, pdf_hash, "vllm")
        if cached is not None:
            return _sanitize_toc_tree(cached)

    client = _build_llm_client(
        base_url=llm_base_url, api_key=llm_api_key,
        timeout=llm_timeout,
    )
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
    result = _call_llm(
        client, llm_model, _TOC_VLLM_SYSTEM_PROMPT, content,
    )
    toc_tree = result.get("toc", []) if isinstance(result, dict) else result
    toc_tree = _sanitize_toc_tree(toc_tree)
    _save_toc_tree_cache(cache_dir, pdf_hash, "vllm", toc_tree)
    return toc_tree
