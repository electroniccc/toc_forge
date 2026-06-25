"""Pipeline orchestration: top-level stages and PDF bookmark injection."""

import logging
import os
import re
import statistics
import time
from pathlib import Path

import fitz

from .llm import build_toc_llm, build_toc_vllm
from .ocr_engine import get_toc_pages, ocr_number_pages
from .parsing import inherit_page_numbers, reconstruct_toc1, repair_toc_tree
from .utils import (
    _roman_to_int,
    compute_file_hash,
    image_from_page,
    make_sure_model_exists,
    print_toc_result,
)

logger = logging.getLogger(__name__)


def build_toc_local_ocr(
    toc_pages: list[dict],
    page_imgs,
    ocr_model,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict]:
    """Build TOC tree from local OCR results using heuristic parsing."""
    from .ocr_engine import ocr_toc_pages

    toc_results = ocr_toc_pages(
        toc_pages,
        page_imgs,
        ocr_model,
        do_debug=do_debug,
        output=output,
        cache_dir=cache_dir,
        pdf_hash=pdf_hash,
    )

    # reconstruct structured TOC with y-offset for multi-page
    page_heights = [img.shape[0] for img in page_imgs]

    # per-page level detection + cross-page merge
    toc_tree1 = reconstruct_toc1(toc_results, page_heights)
    toc_tree1 = repair_toc_tree(toc_tree1)
    toc_tree1 = inherit_page_numbers(toc_tree1)
    if do_debug:
        import json

        from .utils import NumpyEncoder

        perpage_path = os.path.join(output, "toc_tree1.json")
        with open(perpage_path, "w", encoding="utf-8") as f:
            json.dump(toc_tree1, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"[per-page] TOC saved to: {perpage_path}")
    return toc_tree1


def get_page_offset(number_page_results: list[dict]) -> int:
    page_offsets = []
    for result in number_page_results:
        parsing_res_list = result["parsing_res_list"]
        height = result["height"]
        y_off = result.get("y_offset", 0)
        for res in parsing_res_list:
            if res["block_label"] != "number":
                continue
            y_min = res["block_bbox"][1] + y_off
            y_max = res["block_bbox"][3] + y_off
            if y_max <= height * 0.1 or y_min >= height * 0.9:
                raw = res["block_content"].strip()
                page_num = None
                try:
                    page_num = int(raw)
                except ValueError:
                    page_num = _roman_to_int(raw)
                if page_num is None:
                    m = re.search(r"\d+", raw)
                    if m:
                        page_num = int(m.group())
                if page_num is not None:
                    page_offsets.append(result["pdf_page_idx"] - page_num)
                break
    if not page_offsets:
        return 0
    try:
        return statistics.mode(page_offsets)
    except statistics.StatisticsError:
        # tie — use median for stability
        return int(statistics.median(page_offsets))


def add_bookmarks_to_pdf(
    doc: fitz.Document,
    toc_tree: list[dict],
    page_offset: int,
    output_path: str,
) -> None:
    """Add PDF outline (bookmarks) from a TOC tree using printed-page -> PDF index mapping."""

    def _page_num_to_pdf(page_num: int | str | None) -> int:
        """Map a printed page number to a 1-based PDF page number (fitz convention)."""
        if isinstance(page_num, int):
            return page_num + page_offset + 1
        if isinstance(page_num, str):
            try:
                return int(page_num) + page_offset + 1
            except ValueError:
                return 1
        return 1

    def _first_page_num(node: dict) -> int | str | None:
        """Find the first valid page number in a subtree."""
        pn = node.get("page_num")
        if isinstance(pn, int):
            return pn
        for child in node.get("children", []):
            result = _first_page_num(child)
            if result is not None:
                return result
        return None

    def _build_outline(node: dict, level: int) -> list[list]:
        entries = []
        title = node["title"][:200]
        pn = node.get("page_num")
        # inherit page from first child if missing
        if not isinstance(pn, int) and not isinstance(pn, str):
            pn = _first_page_num(node)

        pdf_page = _page_num_to_pdf(pn)
        page_count = doc.page_count
        if pdf_page < 1:
            pdf_page = 1
        elif pdf_page > page_count:
            pdf_page = page_count

        entries.append([level, title, pdf_page])
        for child in node.get("children", []):
            entries.extend(_build_outline(child, level + 1))
        return entries

    outline = []
    for node in toc_tree:
        outline.extend(_build_outline(node, 1))

    if outline:
        doc.set_toc(outline)
    doc.save(output_path)


def bookmark_pdf(
    input: str,
    output: str,
    model_dir: str,
    do_debug: bool = False,
    cache_dir: str | None = None,
    toc_strategy: str = "local_ocr",
    api_base_url: str | None = None,
    api_key: str | None = None,
    llm_name: str = "deepseek-v4-flash",
    vllm_name: str = "qwen3.6-35b-a3b",
) -> tuple[str, float, dict]:
    start_time = time.perf_counter()
    pdf_hash = compute_file_hash(input) if cache_dir else None
    doc = fitz.open(input)
    page_imgs = []
    for i in range(min(30, doc.page_count)):
        page = doc[i]
        img = image_from_page(page)
        page_imgs.append(img)
    layout_detection_model = "PP-DocLayout_plus-L"
    make_sure_model_exists(model_dir, layout_detection_model)
    from paddleocr import LayoutDetection

    layout_model = LayoutDetection(
        model_name=layout_detection_model,
        model_dir=os.path.join(model_dir, layout_detection_model),
    )
    toc_pages, number_pages = get_toc_pages(
        page_imgs,
        layout_model,
        do_debug=do_debug,
        output=output,
        cache_dir=cache_dir,
        pdf_hash=pdf_hash,
    )
    if not toc_pages:
        print("未检测到目录页")
        return "", 0, {}

    doc_ori_classify_model = "PP-LCNet_x1_0_doc_ori"
    make_sure_model_exists(model_dir, doc_ori_classify_model)
    text_detection_model = "PP-OCRv5_server_det"
    make_sure_model_exists(model_dir, text_detection_model)
    text_recognition_model = "PP-OCRv5_server_rec"
    make_sure_model_exists(model_dir, text_recognition_model)

    if toc_strategy == "vllm":
        toc_tree1 = build_toc_vllm(
            toc_pages,
            page_imgs,
            do_debug=do_debug,
            output=output,
            llm_model=vllm_name,
            llm_api_key=api_key,
            llm_base_url=api_base_url,
        )
    elif toc_strategy == "llm":
        from paddleocr import PaddleOCR

        ocr_model = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            doc_orientation_classify_model_dir=os.path.join(
                model_dir, doc_ori_classify_model
            ),
            doc_orientation_classify_model_name=doc_ori_classify_model,
            text_detection_model_name=text_detection_model,
            text_detection_model_dir=os.path.join(model_dir, text_detection_model),
            text_recognition_model_name=text_recognition_model,
            text_recognition_model_dir=os.path.join(model_dir, text_recognition_model),
        )
        toc_tree1 = build_toc_llm(
            toc_pages,
            page_imgs,
            ocr_model,
            do_debug=do_debug,
            output=output,
            cache_dir=cache_dir,
            pdf_hash=pdf_hash,
            llm_model=llm_name,
            llm_api_key=api_key,
            llm_base_url=api_base_url,
        )
    else:
        from paddleocr import PaddleOCR

        ocr_model = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            doc_orientation_classify_model_dir=os.path.join(
                model_dir, doc_ori_classify_model
            ),
            doc_orientation_classify_model_name=doc_ori_classify_model,
            text_detection_model_name=text_detection_model,
            text_detection_model_dir=os.path.join(model_dir, text_detection_model),
            text_recognition_model_name=text_recognition_model,
            text_recognition_model_dir=os.path.join(model_dir, text_recognition_model),
        )
        toc_tree1 = build_toc_local_ocr(
            toc_pages,
            page_imgs,
            ocr_model,
            do_debug=do_debug,
            output=output,
            cache_dir=cache_dir,
            pdf_hash=pdf_hash,
        )

    print_toc_result(toc_tree1)

    region_detection_model = "PP-DocBlockLayout"
    make_sure_model_exists(model_dir, region_detection_model)
    # formula_recognition_model = "PP-FormulaNet_plus-L"
    # make_sure_model_exists(model_dir, formula_recognition_model)
    from paddleocr import PPStructureV3

    structure_model = PPStructureV3(
        use_table_recognition=False,
        use_formula_recognition=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_doc_orientation_classify=True,
        doc_orientation_classify_model_name=doc_ori_classify_model,
        doc_orientation_classify_model_dir=os.path.join(
            model_dir, doc_ori_classify_model
        ),
        region_detection_model_name=region_detection_model,
        region_detection_model_dir=os.path.join(model_dir, region_detection_model),
        text_detection_model_name=text_detection_model,
        text_detection_model_dir=os.path.join(model_dir, text_detection_model),
        text_recognition_model_name=text_recognition_model,
        text_recognition_model_dir=os.path.join(model_dir, text_recognition_model),
        layout_detection_model_name=layout_detection_model,
        layout_detection_model_dir=os.path.join(model_dir, layout_detection_model),
        # formula_recognition_model_name=formula_recognition_model,
        # formula_recognition_model_dir=os.path.join(model_dir, formula_recognition_model),
        formula_recognition_batch_size=2,
        format_block_content=True,
    )
    number_page_results = ocr_number_pages(
        toc_pages,
        page_imgs,
        number_pages,
        structure_model,
        do_debug=do_debug,
        output=output,
        half_img=False,
        cache_dir=cache_dir,
        pdf_hash=pdf_hash,
    )
    page_offset = get_page_offset(number_page_results)
    logger.debug(f"page_offset: {page_offset}")

    # add bookmarks to PDF
    pdf_bookmarks_path = os.path.join(output, f"{Path(input).stem}_bookmarked.pdf")
    add_bookmarks_to_pdf(doc, toc_tree1, page_offset, pdf_bookmarks_path)

    def update_page_offset(toc_tree1, page_offset):
        for item in toc_tree1:
            page_num = item["page_num"]
            if isinstance(page_num, int):
                item["page_num"] += page_offset + 1
            if "children" in item:
                update_page_offset(item["children"], page_offset)

    update_page_offset(toc_tree1, page_offset)
    end_time = time.perf_counter()
    time_cost = end_time - start_time
    logger.debug(f"process {Path(input).stem} cost: {time_cost:.2f} seconds")
    return pdf_bookmarks_path, time_cost, toc_tree1
