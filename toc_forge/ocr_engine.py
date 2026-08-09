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
) -> tuple[list[dict], list[dict]]:
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
    return toc_pages, pages_with_number


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
