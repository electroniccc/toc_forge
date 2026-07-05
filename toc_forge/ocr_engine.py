"""OCR engine: layout detection, text recognition, and page number scanning."""

import json
import logging
import os

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
    return toc_results


def ocr_number_pages(
    toc_pages: list[dict],
    page_imgs,
    number_pages: list[dict],
    ocr_model,
    do_debug: bool = False,
    output: str = "output",
    half_img: bool = False,
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict]:
    last_toc_page_idx = toc_pages[-1]["page"]
    number_page_results = []
    sample_idxs = []
    page_sample_count = 3
    for number_page in number_pages:
        if page_sample_count == 0:
            break
        page_idx = number_page["page"]
        if page_idx <= last_toc_page_idx:
            continue
        page_sample_count -= 1
        sample_idxs.append(page_idx)
    for i in range(len(number_pages)-3, len(number_pages)):
        if i < 0:
            continue
        number_page = number_pages[i]
        page_idx = number_page["page"]
        if page_idx <= last_toc_page_idx:
            continue
        if page_idx in sample_idxs:
            continue
        sample_idxs.append(page_idx)
    for page_idx in sample_idxs:
        img = page_imgs[page_idx]
        img_h_original = img.shape[0]
        img_y_offset = 0
        if half_img:
            boxes = [cb["coordinate"] for cb in number_page["content_boxes"]]
            if boxes:
                min_y = min(b[1] for b in boxes)
                max_y = max(b[3] for b in boxes)
                if max_y < img_h_original * 0.33:
                    img = img[: int(img_h_original * 0.33), :]
                elif max_y < img_h_original * 0.5:
                    img = img[: int(img_h_original * 0.5), :]
                elif min_y > img_h_original * 0.67:
                    img_y_offset = int(img_h_original * 0.67)
                    img = img[img_y_offset:, :]
                elif min_y > img_h_original * 0.5:
                    img_y_offset = int(img_h_original * 0.5)
                    img = img[img_y_offset:, :]

        cache_path = (
            _cache_path(cache_dir, pdf_hash, "structure", page_idx)
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
                out_dir = os.path.join(output, f"page_structure_{page_idx}")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                for res in results:
                    res.save_to_img(save_path=out_dir)
                    res.save_to_json(save_path=out_dir)
            result = results[0]

        j_result = result.json["res"]
        number_page_results.append(
            {
                "width": result["width"],
                "height": img_h_original,
                "y_offset": img_y_offset,
                "parsing_res_list": j_result["parsing_res_list"],
                "pdf_page_idx": page_idx,
            }
        )
    return number_page_results
