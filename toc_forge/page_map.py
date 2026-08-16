"""Roman-numeral page-number mapping for books whose body pages are numbered
"I-1", "I-2", ..., "II-1", ... — a chapter Roman numeral, a dash, and a
page-within-chapter Arabic number (e.g. Morin, "Introduction to Classical
Mechanics": page 13 is "I-1", page 14 "I-2", ...; after chapter I's last
page comes "II-1", etc.).

The pipeline detects this format from the number-box OCR results, then
builds a map {"I-1": 1, "I-2": 2, ..., "II-1": <ch-I length + 1>, ...} by
scanning body pages (rendered on demand, page-number crops OCR'd via the
layout-detected number-box positions) so that bookmark injection can
convert a "X-n" page number to a cumulative Arabic page number before
applying the PDF offset.
"""

import logging
import re
from collections import defaultdict

from .utils import (
    _cache_load,
    _cache_path,
    _cache_save,
    image_from_page,
)

logger = logging.getLogger(__name__)

# "I-28", "IV-3", ... (also tolerates OCR noise like "I_28", "I - 28",
# and lowercase "l" misreads of "I")
_ROMAN_ARABIC_RE = re.compile(r"^([IVXLCDMl]+)[-_]\s*(\d+)$")


def _normalize(text: str) -> str:
    """Normalize OCR noise: strip spaces/underscores around the dash and
    map lowercase 'l' (a common OCR misread of 'I') back to 'I'."""
    t = text.strip()
    if not t:
        return t
    t = t.replace("l", "I")
    t = re.sub(r"\s+", "", t)
    return t.replace("_", "-")


def _parse_roman_arabic(text: str) -> tuple[str, int] | None:
    """Parse a "X-n" page number into (chapter Roman numeral, n), or None.

    Uses a trailing search so a page-number crop whose OCR glued the page
    number to a word (e.g. "...NUMERICALLYXIV-15") still parses.
    """
    t = _normalize(text)
    m = _ROMAN_ARABIC_RE.search(t)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def detect_roman_arabic_format(
    number_ocr_results: list[dict], min_ratio: float = 0.6
) -> bool:
    """True if the majority of number-box OCR texts are "X-n" (Roman chapter
    + within-chapter Arabic page) — the format this module maps.

    ``number_ocr_results`` is the per-page list produced by
    ``ocr_number_boxes``.  Front matter pages that use plain Arabic numbers
    (e.g. Preface "1", "2") are tolerated as long as the body pages
    dominate.
    """
    matches = 0
    total = 0
    for res in number_ocr_results:
        for t in res.get("rec_texts", []):
            total += 1
            if _parse_roman_arabic(str(t)) is not None:
                matches += 1
    if total == 0:
        return False
    ratio = matches / total
    logger.debug(
        f"roman-arabic page format: {matches}/{total} = {ratio:.2f}"
    )
    return ratio >= min_ratio


def _page_number_regions(
    number_pages: list[dict], page_imgs
) -> list[tuple[float, float, float, float]]:
    """Cluster layout-detected number-box centers into fixed crop regions.

    Books print page numbers at a few consistent spots (odd/even pages may
    mirror each other; chapter title pages often center them at the bottom).
    Bucket number-box centers by vertical band (top/mid/bottom) and
    horizontal half (left/right), then take each bucket's median center and
    a padded size so longer numbers (e.g. "I-28") still fit.  Coordinates
    are in the 2x-rendered image space shared by ``page_imgs`` and
    ``image_from_page``.
    """
    buckets: dict[tuple[str, str], list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )
    for np_page in number_pages:
        img = page_imgs[np_page["page"]]
        h, w = img.shape[:2]
        for b in np_page["content_boxes"]:
            x1, y1, x2, y2 = b["coordinate"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if cy < 0.2 * h:
                band = "top"
            elif cy > 0.75 * h:
                band = "bottom"
            else:
                band = "mid"
            half = "left" if cx < w / 2 else "right"
            buckets[(band, half)].append((cx, cy, x2 - x1, y2 - y1))

    regions = []
    for items in buckets.values():
        if not items:
            continue
        n = len(items)
        cxs = sorted(it[0] for it in items)
        cys = sorted(it[1] for it in items)
        bws = sorted(it[2] for it in items)
        bhs = sorted(it[3] for it in items)
        med = lambda xs: xs[n // 2]
        med_cx, med_cy = med(cxs), med(cys)
        pad_w = med(bws) * 2 + 40  # 容纳 "I-28" 这类较长页码
        pad_h = med(bhs) * 1.5 + 10
        regions.append(
            (
                med_cx - pad_w,
                med_cy - pad_h,
                med_cx + pad_w,
                med_cy + pad_h,
            )
        )
    return regions


def _ocr_page_numbers(
    doc,
    page_idx: int,
    ocr_model,
    regions: list[tuple[float, float, float, float]],
    cache_dir: str | None,
    pdf_hash: str | None,
) -> list[str]:
    """OCR the page-number crops of one page; returns raw text per region."""
    cache_path = (
        _cache_path(cache_dir, pdf_hash, "page_map_ocr", page_idx)
        if cache_dir and pdf_hash
        else None
    )
    cached = _cache_load(cache_path) if cache_path else None
    if cached is not None:
        return cached["texts"]

    img = image_from_page(doc[page_idx])
    h, w = img.shape[:2]
    texts = []
    for x1, y1, x2, y2 in regions:
        crop = img[
            max(0, int(y1)) : min(h, int(y2)),
            max(0, int(x1)) : min(w, int(x2)),
        ]
        if crop.size == 0:
            texts.append("")
            continue
        result = ocr_model.predict(crop)[0]
        texts.append("".join(result["rec_texts"]))
    if cache_path:
        _cache_save(cache_path, {"texts": texts})
    return texts


def build_page_map(
    doc,
    start_page: int,
    page_count: int,
    ocr_model,
    number_pages: list[dict],
    page_imgs,
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
    max_scan: int = 2000,
    stop_after_blank: int = 40,
) -> dict[str, int] | None:
    """Scan body pages from ``start_page`` and build {"X-n": cumulative}.

    Algorithm (chapters advance in reading order, page numbers restart at 1
    per chapter):
      - the first page whose number matches "X-n" is the anchor: cumulative
        page 1 (if its n > 1 — e.g. the anchor is "I-2" because "I-1" is an
        unnumbered title page — earlier pages are simply absent from the map)
      - a page whose chapter differs from the previous one and whose n == 1
        starts a new chapter: its cumulative value is the physical page
        distance from the anchor plus one
      - any later page of the same chapter maps to
        chapter_start_cumulative + (n - 1) — OCR misreads of single pages
        cannot corrupt the others
      - a page whose chapter changes but n != 1 is ignored (probably a
        misread); the next "X-1" page re-anchors the new chapter

    Rendering and OCR of the page-number crops are cached per page
    (``page_map_ocr``).  Returns None if the format never appears.
    """
    regions = _page_number_regions(number_pages, page_imgs)
    if not regions:
        logger.debug("build_page_map: no number-box regions, skipping")
        return None

    page_map: dict[str, int] = {}
    anchor_idx: int | None = None
    last_chapter: str | None = None
    chapter_start_cumulative: dict[str, int] = {}
    chapter_max_n: dict[str, int] = {}
    pending_chapter: tuple[str, int] | None = None  # (chapter, inferred start idx)
    blank_streak = 0

    def _fill_chapter(chap: str) -> None:
        """Backfill every within-chapter page number 1..max_n once the chapter
        start is known.  Pages whose OCR failed (or that carry no page number
        at all) would otherwise leave holes in the map, and bookmarks pointing
        at such pages would fall back to page 1."""
        base = chapter_start_cumulative.get(chap)
        max_n = chapter_max_n.get(chap)
        if base is None or max_n is None:
            return
        for nn in range(1, max_n + 1):
            page_map.setdefault(f"{chap}-{nn}", base + (nn - 1))

    end = min(page_count, start_page + max_scan)
    for idx in range(start_page, end):
        texts = _ocr_page_numbers(
            doc, idx, ocr_model, regions, cache_dir, pdf_hash
        )
        parsed = None
        for t in texts:
            parsed = _parse_roman_arabic(t)
            if parsed is not None:
                break
        if parsed is None:
            blank_streak += 1
            if blank_streak >= stop_after_blank:
                logger.debug(
                    f"build_page_map: {blank_streak} pages without a "
                    f"matching page number after p{idx - blank_streak + 1}, "
                    f"stopping ({len(page_map)} entries)"
                )
                break
            continue
        blank_streak = 0

        chap, n = parsed
        if anchor_idx is None:
            anchor_idx = idx
            page_map[f"{chap}-{n}"] = 1
            last_chapter = chap
            chapter_start_cumulative[chap] = 1 - (n - 1)
            logger.debug(
                f"build_page_map: anchor p{idx} = {chap}-{n} (cumulative 1)"
            )
            continue

        if chap != last_chapter:
            if n == 1:
                # 章号变化的首页（页码为 1）直接确认新章起点
                if last_chapter is not None:
                    _fill_chapter(last_chapter)
                cumulative = idx - anchor_idx + 1
                page_map[f"{chap}-{n}"] = cumulative
                chapter_start_cumulative[chap] = cumulative
                last_chapter = chap
                pending_chapter = None
                logger.debug(
                    f"build_page_map: new chapter {chap} starts at p{idx}, "
                    f"cumulative {cumulative}"
                )
                continue
            # 章号变化但页码不是 1：可能首章页码页缺失/OCR 失败（如 XIV 章
            # 的 XIV-1 页没扫到）。用本页反推章起点（idx - (n-1)），等同一
            # 章的第二页出现且反推起点一致时再确认新章。
            inferred = idx - (n - 1)
            if pending_chapter is None or pending_chapter[0] != chap:
                pending_chapter = (chap, inferred)
                continue
            if abs(pending_chapter[1] - inferred) > 2:
                pending_chapter = (chap, inferred)
                continue
            if last_chapter is not None:
                _fill_chapter(last_chapter)
            cumulative = inferred - anchor_idx + 1
            page_map[f"{chap}-1"] = cumulative
            chapter_start_cumulative[chap] = cumulative
            last_chapter = chap
            pending_chapter = None
            logger.debug(
                f"build_page_map: new chapter {chap} confirmed via p{idx} "
                f"({chap}-{n}), inferred start p{inferred}, cumulative {cumulative}"
            )
            # fall through to map this page within the new chapter
        elif pending_chapter is not None and pending_chapter[0] == chap:
            # 回到上一章：pending 作废（之前是误读）
            pending_chapter = None

        # 同章内：累计 = 章起点累计 + (n - 1)
        base = chapter_start_cumulative.get(chap)
        if base is None:
            continue
        page_map[f"{chap}-{n}"] = base + (n - 1)
        chapter_max_n[chap] = max(chapter_max_n.get(chap, 0), n)

    # 扫描结束：补全最后一章（后续章永远不会触发 _fill_chapter）
    if last_chapter is not None:
        _fill_chapter(last_chapter)

    if not page_map:
        return None
    logger.info(
        f"roman page map built: {len(page_map)} entries "
        f"(anchor p{anchor_idx}, scanned p{start_page}-{idx})"
    )
    return page_map


def map_page_num(page_map: dict[str, int] | None, page_num) -> int | None:
    """Convert a "X-n" page number to cumulative Arabic via ``page_map``.

    Returns None when the map is absent, the page number is not a string,
    or the key is not in the map — the caller then falls back to its
    default handling.
    """
    if not page_map or not isinstance(page_num, str):
        return None
    key = _normalize(page_num)
    m = _ROMAN_ARABIC_RE.match(key)
    if not m:
        return None
    return page_map.get(key)
