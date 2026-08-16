"""Pipeline orchestration: top-level stages and PDF bookmark injection."""

import gc
import logging
import os
import re
import statistics
import time
from pathlib import Path

import pymupdf
from paddleocr import LayoutDetection, PaddleOCR

from .llm import build_toc_llm, build_toc_vllm
from .ocr_engine import (
    compute_page_offset,
    detect_toc_pages_by_continuity,
    detect_toc_pages_by_keyword,
    get_number_box_pages,
    get_toc_pages,
    keep_longest_contiguous_pages,
    ocr_number_boxes,
)
from .page_map import (
    build_arabic_page_map,
    build_front_matter_offset,
    build_page_map,
    detect_roman_arabic_format,
    detect_segmented_offset,
    map_arabic_page,
    map_page_num,
)
from .parsing import inherit_page_numbers, reconstruct_toc1, repair_toc_tree
from .utils import (
    _roman_to_int,
    compute_file_hash,
    image_from_page,
    make_sure_model_exists,
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


def add_bookmarks_to_pdf(
    doc: pymupdf.Document,
    toc_tree: list[dict],
    page_offset: int,
    output_path: str,
    page_map: dict[str, int] | None = None,
    front_offset: int | None = None,
    arabic_segments: list[dict] | None = None,
) -> None:
    """Add PDF outline (bookmarks) from a TOC tree using printed-page -> PDF index mapping.

    ``page_map`` (from :mod:`toc_forge.page_map`) converts "X-n" page
    numbers (Roman chapter + within-chapter Arabic, e.g. "I-2") to
    cumulative Arabic page numbers before applying the offset.
    ``front_offset`` maps front-matter pages that use Roman numerals
    ("vii") — their numbering is a separate system from the Arabic body
    pages, so they need their own offset.
    """

    def _page_num_to_pdf(page_num: int | str | None) -> int:
        """Map a printed page number to a 1-based PDF page number (pymupdf convention)."""
        if isinstance(page_num, int):
            mapped = map_arabic_page(arabic_segments, page_num)
            if mapped is not None:
                return mapped + 1
            return page_num + page_offset + 1
        if isinstance(page_num, str):
            mapped = map_page_num(page_map, page_num)
            if mapped is not None:
                return mapped + page_offset + 1
            try:
                return int(page_num) + page_offset + 1
            except ValueError:
                # front matter often uses lowercase Roman numerals ("vii")
                r = _roman_to_int(page_num)
                if r is not None:
                    base = front_offset if front_offset is not None else page_offset
                    return r + base + 1
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
    no_toc_cache: bool = False,
    device: str | None = None,
    llm_timeout: float = 600.0,
    cpu_threads: int | None = None,
    engine: str | None = None,
    enable_mkldnn: bool | None = None,
    ocr_model_size: str = "server",
) -> tuple[str, float, dict]:
    start_time = time.perf_counter()
    pdf_hash = compute_file_hash(input) if cache_dir else None
    doc = pymupdf.open(input)
    page_imgs = []
    for i in range(min(50, doc.page_count)):
        page = doc[i]
        img = image_from_page(page)
        page_imgs.append(img)
    layout_detection_model = "PP-DocLayout_plus-L"
    make_sure_model_exists(model_dir, layout_detection_model)

    # 引擎行为只在被显式指定时才传入，None 保持 PaddleX 默认：
    # - engine: 如 "onnxruntime" 时使用模型目录下的 inference.onnx 推理，
    #   避免依赖 paddle 运行时（WSL/Linux 已验证）
    # - cpu_threads: GUI 用它限制推理线程数（否则 paddlex 默认开 10 个 OpenMP
    #   线程占满 CPU，饿死 UI 主循环）
    # - enable_mkldnn: False 时关闭 MKLDNN —— paddle 3.3.1 的 oneDNN 新执行器
    #   在 Windows CPU 推理有 bug（ConvertPirAttribute2RuntimeAttribute 不支持
    #   ArrayAttribute<DoubleAttribute>，onednn_instruction.cc:118 崩溃），
    #   打包版 GUI 必须关闭；Linux 传 None 保持默认（启用）。
    _engine_kwargs = {}
    if engine:
        _engine_kwargs["engine"] = engine
    if cpu_threads:
        _engine_kwargs["cpu_threads"] = cpu_threads
    if enable_mkldnn is not None:
        _engine_kwargs["enable_mkldnn"] = enable_mkldnn
    layout_model = LayoutDetection(
        model_name=layout_detection_model,
        model_dir=os.path.join(model_dir, layout_detection_model),
        device=device,
        **_engine_kwargs,
    )
    toc_pages, number_pages, all_boxes = get_toc_pages(
        page_imgs,
        layout_model,
        do_debug=do_debug,
        output=output,
        cache_dir=cache_dir,
        pdf_hash=pdf_hash,
    )
    # 布局检测完成后布局模型不再需要，显式释放：onnxruntime 的 CUDA arena 对
    # 动态 shape 会膨胀到数 GB 且不自动归还，与 OCR 模型的 session 叠加可撑爆
    # 6GB 显存导致推理变慢；del + gc 后 onnxruntime session 销毁会归还显存。
    del layout_model
    gc.collect()
    logger.debug(f"number_pages: {number_pages}")

    doc_ori_classify_model = "PP-LCNet_x1_0_doc_ori"
    make_sure_model_exists(model_dir, doc_ori_classify_model)
    # OCR 模型规格：server（默认，精度高）或 mobile（CPU 上快一个量级，
    # GUI 打包版只有 CPU 可用，用 mobile 控制耗时）
    if ocr_model_size == "mobile":
        text_detection_model = "PP-OCRv5_mobile_det"
        text_recognition_model = "PP-OCRv5_mobile_rec"
    else:
        text_detection_model = "PP-OCRv5_server_det"
        text_recognition_model = "PP-OCRv5_server_rec"
    make_sure_model_exists(model_dir, text_detection_model)
    make_sure_model_exists(model_dir, text_recognition_model)

    # 无论哪种策略都创建 OCR 模型：llm/local_ocr 用它做目录 OCR，
    # 页码扫描（get_page_offset2）也需要它，vllm 策略同样需要。
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
        device=device,
        **_engine_kwargs,
    )

    # 布局漏检的目录页补充：页面上部有 paragraph_title 或页角 header 的文本为
    # "Contents"/"目录" 时也判为目录页（如英文教材 CONTENTS 页眉页，Layout
    # 只标了 text/header 而没有 content box）；补充页无 content box，OCR 过滤
    # 用合成的整页框（排除页眉页脚带与命中的标题框，避免 "CONTENTS v" 之类
    # 被解析成目录条目）。
    toc_pages.extend(
        detect_toc_pages_by_keyword(
            all_boxes,
            page_imgs,
            ocr_model,
            skip={p["page"] for p in toc_pages},
            do_debug=do_debug,
            output=output,
            cache_dir=cache_dir,
            pdf_hash=pdf_hash,
        )
    )
    # 目录页连续传播：英文书目录常跨多页，但只有首页带 "Contents" 标题
    # （如 OpenStax），后续页没有关键词可依；对已确认目录页的后续页检查
    # "行尾带页码"风格（"4.10 Antiderivatives 419"），符合则续上目录页。
    toc_pages.extend(
        detect_toc_pages_by_continuity(
            toc_pages,
            all_boxes,
            page_imgs,
            ocr_model,
            do_debug=do_debug,
            output=output,
            cache_dir=cache_dir,
            pdf_hash=pdf_hash,
        )
    )
    # 目录页去噪：只保留最大连续页段，丢弃孤立误检（如 [5, 7, 8, 9] -> [7, 8, 9]）
    toc_pages = keep_longest_contiguous_pages(toc_pages)
    if not toc_pages:
        print("未检测到目录页")
        return "", 0, {}

    if toc_strategy == "vllm":
        toc_tree1 = build_toc_vllm(
            toc_pages,
            page_imgs,
            do_debug=do_debug,
            output=output,
            cache_dir=cache_dir,
            pdf_hash=pdf_hash,
            no_toc_cache=no_toc_cache,
            llm_model=vllm_name,
            llm_api_key=api_key,
            llm_base_url=api_base_url,
            llm_timeout=llm_timeout,
        )
    elif toc_strategy == "llm":
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
            no_toc_cache=no_toc_cache,
            llm_timeout=llm_timeout,
        )
    else:
        toc_tree1 = build_toc_local_ocr(
            toc_pages,
            page_imgs,
            ocr_model,
            do_debug=do_debug,
            output=output,
            cache_dir=cache_dir,
            pdf_hash=pdf_hash,
        )

    # 页码偏移：直接用布局检测的 number box 位置截取小图 + PaddleOCR 扫描
    # （get_page_offset2），不再调用 PPStructureV3（加载子模型多、耗时长）。
    # 这里显式走三阶段而不是 get_page_offset2 wrapper，是为了拿到 number OCR
    # 结果做罗马页码格式检测（detect_roman_arabic_format）。
    kept_pages = get_number_box_pages(number_pages, page_imgs)
    number_ocr_results = ocr_number_boxes(
        kept_pages,
        page_imgs,
        ocr_model,
        do_debug=do_debug,
        output=output,
        cache_dir=cache_dir,
        pdf_hash=pdf_hash,
    )
    page_offset = compute_page_offset(number_ocr_results)
    logger.debug(f"page_offset: {page_offset}")
    # front matter 偏移：有的书 front matter 用罗马页码（"vii"）、正文用阿拉伯
    # 数字（1, 2, ...），两套体系 offset 不同（如 Kibble：正文 offset 20，
    # front matter offset 0）。目录树里存在罗马字符串页码时，扫描目录后到
    # 正文起始（印刷 1 所在页 = page_offset + 1）之间的 front matter 页码。
    front_offset = None

    def _has_roman_page(nodes: list[dict]) -> bool:
        for n in nodes:
            pn = n.get("page_num")
            if isinstance(pn, str) and _roman_to_int(pn) is not None:
                return True
            if _has_roman_page(n.get("children", [])):
                return True
        return False

    if _has_roman_page(toc_tree1):
        toc_end = max((p["page"] for p in toc_pages), default=-1)
        front_offset = build_front_matter_offset(
            doc,
            toc_end + 1,
            page_offset + 1,
            ocr_model,
            cache_dir=cache_dir,
            pdf_hash=pdf_hash,
        )
        logger.debug(f"front matter offset: {front_offset}")

    # 罗马数字页码映射：正文页码为 "I-1"、"II-1"（罗马章号-章内阿拉伯页码）
    # 形式的书（如 Morin 力学），先检测格式，再把每章页码映射成累计阿拉伯
    # 页码，加书签时用映射后的页码计算偏移。
    page_map = None
    if detect_roman_arabic_format(number_ocr_results):
        start_page = max((p["page"] for p in toc_pages), default=-1) + 1
        page_map = build_page_map(
            doc,
            start_page,
            doc.page_count,
            ocr_model,
            number_pages,
            page_imgs,
            cache_dir=cache_dir,
            pdf_hash=pdf_hash,
        )
        logger.info(
            f"roman page map: {len(page_map)} entries"
            if page_map
            else "roman page map: format detected but no entries found"
        )

    # 分段页码映射：有些书（如 Shankar）印刷页码与 PDF 索引不是单一线性
    # 关系（章节边界跳号）。先抽样各章起始页检查 offset 是否一致，只有
    # 发现分段才全书扫描构建段表（书签 int 页码先查段表，段内线性、
    # 段间空洞取最近段反推）——单 offset 的书抽样即可，省去全书扫描。
    arabic_segments = None
    if detect_segmented_offset(
        doc,
        ocr_model,
        toc_tree1,
        page_offset,
        number_pages,
        page_imgs,
        cache_dir=cache_dir,
        pdf_hash=pdf_hash,
    ):
        arabic_segments = build_arabic_page_map(
            doc,
            ocr_model,
            number_pages,
            page_imgs,
            cache_dir=cache_dir,
            pdf_hash=pdf_hash,
        )

    # OCR 模型用完即释放（onnxruntime 的 CUDA arena 显存不自动归还）
    del ocr_model
    gc.collect()

    # add bookmarks to PDF
    if not os.path.exists(output):
        os.makedirs(output)
    pdf_bookmarks_path = os.path.join(output, f"{Path(input).stem}_bookmarked.pdf")
    add_bookmarks_to_pdf(
        doc,
        toc_tree1,
        page_offset,
        pdf_bookmarks_path,
        page_map=page_map,
        front_offset=front_offset,
        arabic_segments=arabic_segments,
    )

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
