"""OCR engine: layout detection, text recognition, and page number scanning."""

import json
import logging
import os
import re
import statistics
import time

from .utils import (
    CachedResult,
    NumpyEncoder,
    _cache_load,
    _cache_path,
    _cache_save,
    _cacheable_dict,
    _unwrap_legacy_cache,
    deduplicate_content_boxes,
    filter_toc_result,
)

logger = logging.getLogger(__name__)


def get_toc_pages(
    imgs,
    ocr_model,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> tuple[list[dict], list[dict], list[list[dict]]]:
    t0 = time.perf_counter()
    cached_results = None
    if cache_dir and pdf_hash:
        cached = []
        for i in range(len(imgs)):
            data = _cache_load(_cache_path(cache_dir, pdf_hash, "layout", i))
            if data is None:
                break
            cached.append(CachedResult(_unwrap_legacy_cache(data)))
        else:
            cached_results = cached

    if cached_results is not None:
        results = cached_results
    else:
        results = ocr_model.predict(imgs, layout_nms=True)
        if cache_dir and pdf_hash:
            for i, res in enumerate(results):
                _cache_save(
                    _cache_path(cache_dir, pdf_hash, "layout", i),
                    _cacheable_dict(res),
                )

    if do_debug and cached_results is None:
        for i, res in enumerate(results):
            res.save_to_img(os.path.join(output, f"page_layout_{i}.png"))
            res.save_to_json(os.path.join(output, f"page_layout_{i}.json"))

    toc_pages = []
    for res_i, res in enumerate(results):
        boxes = res["boxes"]
        content_boxes = []
        para_titles = []
        for box in boxes:
            if box["label"] == "content":
                content_boxes.append(box)
            elif box["label"] == "paragraph_title":
                para_titles.append(box)
        if content_boxes:
            # Include paragraph_title boxes that sit between content boxes
            # (or above the first one).  These are often "篇" / "章" headings
            # that the layout model didn't label as "content".
            sorted_cbs = sorted(content_boxes, key=lambda cb: cb["coordinate"][1])
            prev_bottom = -20
            for cb in sorted_cbs:
                cb_top = cb["coordinate"][1]
                between = [
                    pt for pt in para_titles
                    if pt["coordinate"][3] <= cb_top + 5
                    and pt["coordinate"][1] >= prev_bottom - 5
                ]
                content_boxes.extend(between)
                prev_bottom = cb["coordinate"][3]
            content_boxes = deduplicate_content_boxes(content_boxes)
            toc_pages.append({"page": res_i, "content_boxes": content_boxes})
    # pages that may carry printed page numbers
    pages_with_number = []
    for res_i, res in enumerate(results):
        boxes = res["boxes"]
        content_boxes = []
        for box in boxes:
            if box["label"] == "number":
                content_boxes.append(box)
        if content_boxes:
            content_boxes = deduplicate_content_boxes(content_boxes)
            pages_with_number.append({"page": res_i, "content_boxes": content_boxes})
    logger.debug(
        f"layout detection cost: {time.perf_counter() - t0:.2f}s "
        f"({len(imgs)} pages, {'cached' if cached_results is not None else 'inference'})"
    )
    # third return: raw per-page boxes — the keyword supplement
    # (detect_toc_pages_by_keyword) needs them to find paragraph_title /
    # header boxes on pages the layout model did not flag as TOC pages
    return toc_pages, pages_with_number, [res["boxes"] for res in results]


def detect_toc_pages_by_keyword(
    all_boxes: list[list[dict]],
    page_imgs,
    ocr_model,
    skip: set[int] | None = None,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict]:
    """Find TOC pages the layout model missed, by keyword.

    A page whose upper area carries a ``paragraph_title``, or whose top-left /
    top-right corner carries a ``header``, whose OCR text reads
    "Contents"/"目录", is treated as a TOC page even though the layout model
    produced no ``content`` box (common for English textbooks where the TOC
    pages carry only a running "CONTENTS" header).  Position alone is not
    enough — the keyword check is what rejects pages like "PREFACE" or
    "DIAGNOSTIC TESTS" headers.

    Returns pages as {"page": idx, "content_boxes": [synthetic full-content
    box]} so the normal OCR filtering / parsing flow consumes them unchanged.
    Candidate-box OCR is cached per page under the "toc_keyword" stage.
    """
    if skip is None:
        skip = set()
    matched_pages = []
    for page_idx, boxes in enumerate(all_boxes):
        if page_idx in skip or page_idx >= len(page_imgs):
            continue
        img = page_imgs[page_idx]
        h, w = img.shape[:2]
        candidates = []
        for b in boxes:
            label = b.get("label")
            x1, y1, x2, y2 = b["coordinate"]
            if label == "paragraph_title" and (y1 + y2) / 2 <= 0.4 * h:
                # 靠上位置的 paragraph_title（如 CONTENTS 大标题）
                candidates.append(b)
            elif label == "header" and (y1 + y2) / 2 <= 0.15 * h and (
                x1 <= 0.3 * w or x2 >= 0.7 * w
            ):
                # 页眉左上/右上角的 header（如 CONTENTS 运行页眉）
                candidates.append(b)
        if not candidates:
            continue

        cache_path = (
            _cache_path(cache_dir, pdf_hash, "toc_keyword", page_idx)
            if cache_dir and pdf_hash
            else None
        )
        cached = _cache_load(cache_path) if cache_path else None
        if cached is not None:
            cand_texts = cached["candidates"]
        else:
            cand_texts = []
            for b in candidates:
                x1, y1, x2, y2 = b["coordinate"]
                pad = 12
                crop = img[
                    max(0, int(y1) - pad) : min(h, int(y2) + pad),
                    max(0, int(x1) - pad) : min(w, int(x2) + pad),
                ]
                if crop.size == 0:
                    continue
                # 页眉字号很小（2x 渲染下仅 ~15px 高），pad 太紧会切掉字形导致
                # OCR 乱码（"CONTENTS" -> "SSNESEEE"）；放大 2x 提升识别率
                import cv2

                up = cv2.resize(
                    crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                )
                result = ocr_model.predict(up)[0]
                cand_texts.append(
                    {
                        "coordinate": [float(c) for c in b["coordinate"]],
                        "texts": result["rec_texts"],
                    }
                )
            if cache_path:
                _cache_save(cache_path, {"candidates": cand_texts})

        matched_boxes = []
        for ct in cand_texts:
            # join without spaces so split fragments ("目 录") still match
            text = "".join(ct["texts"]).strip().lower()
            if "contents" in text or "目录" in text:
                matched_boxes.append({"coordinate": ct["coordinate"]})
        if not matched_boxes:
            continue

        if do_debug:
            import cv2

            if not os.path.exists(output):
                os.makedirs(output)
            for k, mb in enumerate(matched_boxes):
                x1, y1, x2, y2 = mb["coordinate"]
                pad = 12
                crop = img[
                    max(0, int(y1) - pad) : min(h, int(y2) + pad),
                    max(0, int(x1) - pad) : min(w, int(x2) + pad),
                ]
                cv2.imwrite(
                    os.path.join(output, f"page_toc_keyword_{page_idx}_{k}.png"),
                    crop,
                )
            with open(
                os.path.join(output, f"page_toc_keyword_{page_idx}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "page": page_idx,
                        "matched": matched_boxes,
                        "candidates": cand_texts,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        matched_pages.append(
            {
                "page": page_idx,
                "content_boxes": [
                    _synthetic_content_box(boxes, img.shape, matched_boxes)
                ],
            }
        )
    return matched_pages


def _synthetic_content_box(
    boxes: list[dict], img_shape, matched_boxes: list[dict]
) -> dict:
    """Full-page content box for keyword-detected TOC pages.

    The layout model produced no ``content`` box on these pages, so OCR
    filtering must span the whole content area.  The running-header band
    (top 8% of the page), detected footer boxes, and the matched keyword
    boxes themselves are excluded — otherwise lines like "CONTENTS v" or
    the "Contents" title would be parsed as bogus TOC entries.
    """
    h, w = img_shape[:2]
    y1 = 0.08 * h
    y2 = h
    for b in boxes:
        if b.get("label") == "footer":
            # filter_toc_result 的 is_inside 带 20px 容差，页脚的版权行
            # （紧贴 footer 框上方）会漏进来，需留出至少 tol 的余量
            y2 = min(y2, b["coordinate"][1] - 30)
    for mb in matched_boxes:
        y1 = max(y1, mb["coordinate"][3] + 5)
    if y2 <= y1 + 20:
        y2 = h  # fallback: 无 footer 框或边界异常时退回整页
    return {"coordinate": [0.0, y1, w, y2], "label": "content", "score": 1.0}


def keep_longest_contiguous_pages(toc_pages: list[dict]) -> list[dict]:
    """Drop isolated TOC-page detections: keep only the longest run of
    consecutive page indices (e.g. [5, 7, 8, 9] -> [7, 8, 9], 5 is a stray).
    Ties keep the earliest run.  Returns pages sorted by page index — the
    keyword supplement appends its pages after the layout-detected ones, and
    the tree merge expects ascending page order."""
    if len(toc_pages) <= 1:
        return toc_pages
    pages = sorted(p["page"] for p in toc_pages)
    best_start = cur_start = pages[0]
    best_len = cur_len = 1
    for prev, cur in zip(pages, pages[1:]):
        if cur == prev + 1:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_start, cur_len = cur, 1
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len
    keep = set(range(best_start, best_start + best_len))
    return sorted(
        (p for p in toc_pages if p["page"] in keep), key=lambda p: p["page"]
    )


def detect_toc_pages_by_continuity(
    toc_pages: list[dict],
    all_boxes: list[list[dict]],
    page_imgs,
    ocr_model,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict]:
    """Propagate TOC-page detection to pages following a confirmed TOC page.

    English textbooks' TOC often spans several pages while only the first one
    carries a "Contents" heading — the layout model produces no ``content``
    boxes on any of them, and the keyword supplement (which requires
    "Contents"/"目录" in a header/title box) matches only the heading page.
    Each page following a known TOC page is checked for "TOC style": most
    text lines end with a page number (e.g. "4.10 Antiderivatives 419").
    Pages that pass are added as TOC pages with a synthetic full-page
    content box, and propagation continues; the first page that does not
    look like a TOC page (e.g. the Preface body) stops the run.

    Returns the newly added pages as {"page": idx, "content_boxes":
    [synthetic box]} so the normal OCR / parsing flow consumes them
    unchanged.  Full-page OCR is cached per page under the
    "toc_continuity" stage.
    """
    known = {p["page"] for p in toc_pages}
    found = []
    for base in sorted(known):
        nxt = base + 1
        while nxt < len(page_imgs) and nxt not in known:
            if _is_toc_style_page(
                nxt,
                all_boxes,
                page_imgs,
                ocr_model,
                cache_dir=cache_dir,
                pdf_hash=pdf_hash,
            ):
                found.append(
                    {
                        "page": nxt,
                        "content_boxes": [
                            _synthetic_content_box(
                                all_boxes[nxt], page_imgs[nxt].shape, []
                            )
                        ],
                    }
                )
                known.add(nxt)
                nxt += 1
            else:
                break
    return found


def _is_toc_style_page(
    page_idx: int,
    all_boxes: list[list[dict]],
    page_imgs,
    ocr_model,
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> bool:
    """True if most text lines on the page end with a page number.

    TOC entries of the "Title 419" form (no dot leaders) end with digits;
    running text rarely does.  Lines inside the header band (top 8%) and
    below the first footer box are excluded — page numbers printed at the
    bottom would otherwise inflate the ratio.
    """
    img = page_imgs[page_idx]
    h, w = img.shape[:2]
    y_top = 0.08 * h
    y_bottom = h
    for b in all_boxes[page_idx]:
        if b.get("label") == "footer":
            y_bottom = min(y_bottom, b["coordinate"][1] - 30)

    cache_path = (
        _cache_path(cache_dir, pdf_hash, "toc_continuity", page_idx)
        if cache_dir and pdf_hash
        else None
    )
    cached = _cache_load(cache_path) if cache_path else None
    if cached is not None:
        rec_texts, rec_boxes = cached["rec_texts"], cached["rec_boxes"]
    else:
        result = ocr_model.predict(img)[0]
        rec_texts = list(result["rec_texts"])
        rec_boxes = [[float(v) for v in b] for b in result["rec_boxes"]]
        if cache_path:
            _cache_save(
                cache_path, {"rec_texts": rec_texts, "rec_boxes": rec_boxes}
            )

    items = [
        (str(t).strip(), b)
        for t, b in zip(rec_texts, rec_boxes)
        if str(t).strip()
    ]
    if len(items) < 5:
        return False

    # group into lines by y-overlap (same as _parse_toc_lines' grouping)
    items.sort(key=lambda x: x[1][1])
    lines = []
    for item in items:
        placed = False
        for line in lines:
            line_ymin = sum(b[1] for _, b in line) / len(line)
            line_ymax = sum(b[3] for _, b in line) / len(line)
            ymin, ymax = item[1][1], item[1][3]
            if max(ymin, line_ymin) < min(ymax, line_ymax):
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    checks = []
    for line in lines:
        line.sort(key=lambda x: x[1][0])
        x1, y1, x2, y2 = line[-1][1]
        if y2 < y_top or y1 > y_bottom:
            continue
        checks.append(bool(re.search(r"\d+\s*$", line[-1][0])))
    if len(checks) < 5:
        return False
    return sum(checks) / len(checks) >= 0.5


def ocr_toc_pages(
    toc_pages: list[dict],
    page_imgs,
    ocr_model,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict]:
    t0 = time.perf_counter()
    toc_results = []
    for toc_page in toc_pages:
        page_idx = toc_page["page"]
        img = page_imgs[page_idx]

        cache_path = (
            _cache_path(cache_dir, pdf_hash, "ocr", page_idx)
            if cache_dir and pdf_hash
            else None
        )
        cached = _cache_load(cache_path) if cache_path else None
        if cached is not None:
            result = CachedResult(_unwrap_legacy_cache(cached))
        else:
            results = ocr_model.predict(img)
            if cache_path:
                _cache_save(cache_path, _cacheable_dict(results[0]))
            if do_debug:
                res_dir = os.path.join(output, f"page_ocr_{page_idx}")
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                for i, res in enumerate(results):
                    res.save_to_img(res_dir)
                    res.save_to_json(res_dir)
            result = results[0]

        toc_result = filter_toc_result(result, toc_page["content_boxes"])
        angle = result["doc_preprocessor_res"]["angle"]
        if do_debug:
            if not os.path.exists(output):
                os.makedirs(output)
            with open(
                os.path.join(output, f"page_{page_idx}_toc_result.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(toc_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        toc_results.append(
            {"page": page_idx, "angle": angle, "content_boxes": toc_result}
        )
    logger.debug(
        f"OCR toc pages cost: {time.perf_counter() - t0:.2f}s ({len(toc_pages)} pages)"
    )
    return toc_results


def get_number_box_pages(
    number_pages: list[dict],
    page_imgs,
) -> list[dict]:
    """Aggregate per-page number box bounds from layout results, filter by parity-group consistency.

    Layout detection sometimes produces accidental "number" boxes.  Printed page
    numbers on odd pages mirror those on even pages (e.g. right vs left corner),
    so within each parity group the boxes should mostly overlap: pages whose box
    center deviates from the group median are dropped.
    """
    from collections import defaultdict

    # aggregate per-page number box bounds, grouped by page parity
    groups: dict[int, list[dict]] = defaultdict(list)
    for np_page in number_pages:
        page_idx = np_page["page"]
        boxes = np_page["content_boxes"]
        if not boxes:
            continue
        xs = [b["coordinate"][0] for b in boxes]
        ys = [b["coordinate"][1] for b in boxes]
        x2s = [b["coordinate"][2] for b in boxes]
        y2s = [b["coordinate"][3] for b in boxes]
        groups[page_idx % 2].append(
            {
                "page": page_idx,
                "x1": min(xs),
                "y1": min(ys),
                "x2": max(x2s),
                "y2": max(y2s),
                "cx": (min(xs) + max(x2s)) / 2,
                "cy": (min(ys) + max(y2s)) / 2,
            }
        )

    # parity-group consistency filter: drop pages whose number box center
    # deviates too far from the group median (accidental "number" boxes)
    kept_pages = []
    for parity, items in groups.items():
        if not items:
            continue
        med_cx = statistics.median(it["cx"] for it in items)
        med_cy = statistics.median(it["cy"] for it in items)
        for it in items:
            h = page_imgs[it["page"]].shape[0]
            w = page_imgs[it["page"]].shape[1]
            if (
                abs(it["cx"] - med_cx) <= 0.25 * w
                and abs(it["cy"] - med_cy) <= 0.15 * h
            ):
                kept_pages.append(it)
    return kept_pages


def ocr_number_boxes(
    kept_pages: list[dict],
    page_imgs,
    ocr_model,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict]:
    """Crop each kept page around its number box (with padding) and OCR the crop.

    Returns a list of {"page": page_idx, "rec_texts": [...]}.  Results are cached
    per page as "number_ocr" cache entries.
    """
    t0 = time.perf_counter()
    ocr_results = []
    for it in kept_pages:
        page_idx = it["page"]
        img = page_imgs[page_idx]
        h, w = img.shape[:2]
        pad = 10
        x1 = max(0, int(it["x1"]) - pad)
        y1 = max(0, int(it["y1"]) - pad)
        x2 = min(w, int(it["x2"]) + pad)
        y2 = min(h, int(it["y2"]) + pad)
        crop = img[y1:y2, x1:x2]

        cache_path = (
            _cache_path(cache_dir, pdf_hash, "number_ocr", page_idx)
            if cache_dir and pdf_hash
            else None
        )
        cached = _cache_load(cache_path) if cache_path else None
        if cached is not None:
            result = CachedResult(_unwrap_legacy_cache(cached))
        else:
            results = ocr_model.predict(crop)
            result = results[0]
            if cache_path:
                _cache_save(cache_path, _cacheable_dict(result))

        if do_debug:
            import cv2

            if not os.path.exists(output):
                os.makedirs(output)
            cv2.imwrite(
                os.path.join(output, f"page_number_crop_{page_idx}.png"), crop
            )
            with open(
                os.path.join(output, f"page_number_ocr_{page_idx}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "page": page_idx,
                        "crop_bbox": [x1, y1, x2, y2],
                        "rec_texts": result["rec_texts"],
                        "rec_scores": result["rec_scores"],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                    cls=NumpyEncoder,
                )

        ocr_results.append({"page": page_idx, "rec_texts": result["rec_texts"]})
    logger.debug(
        f"OCR number pages cost: {time.perf_counter() - t0:.2f}s ({len(kept_pages)} pages)"
    )
    return ocr_results


def compute_page_offset(ocr_results: list[dict]) -> int:
    """Parse printed page numbers from number-box OCR text and compute the offset.

    offset = pdf_page_idx - printed_page_num; invalid entries (no digits in the
    OCR text) are skipped, and the mode of the remaining offsets is returned
    (median on ties) — so isolated OCR misreads (e.g. 6/9 confusion) are tolerated.
    """
    offsets = []
    for res in ocr_results:
        texts = "".join(res["rec_texts"])
        page_num = None
        try:
            page_num = int(texts.strip())
        except ValueError:
            m = re.search(r"\d+", texts)
            if m:
                page_num = int(m.group())
        if page_num is not None:
            offsets.append(res["page"] - page_num)
            logger.debug(f"page offset: pdf page {res['page']} -> printed {page_num}")

    if not offsets:
        logger.warning("compute_page_offset: no valid page numbers found, offset=0")
        return 0
    try:
        return statistics.mode(offsets)
    except statistics.StatisticsError:
        # tie — use median for stability
        return int(statistics.median(offsets))


def get_page_offset2(
    number_pages: list[dict],
    page_imgs,
    ocr_model,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> int:
    """Compute page offset from layout number boxes + PaddleOCR, no PPStructureV3.

    Convenience wrapper around the three stages:
    ``get_number_box_pages`` (collect/filter number box regions)
    -> ``ocr_number_boxes`` (OCR the crops)
    -> ``compute_page_offset`` (parse numbers, take mode offset).
    """
    kept_pages = get_number_box_pages(number_pages, page_imgs)
    ocr_results = ocr_number_boxes(
        kept_pages,
        page_imgs,
        ocr_model,
        do_debug=do_debug,
        output=output,
        cache_dir=cache_dir,
        pdf_hash=pdf_hash,
    )
    return compute_page_offset(ocr_results)
