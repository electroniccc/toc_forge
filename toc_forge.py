import base64
import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import sys
import time

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, TypedDict

import cv2
import fitz
import numpy as np
from openai import OpenAI
from paddleocr import LayoutDetection, PaddleOCR, PPStructureV3
from paddlex.utils.download import download_and_extract
from PIL import Image
from sklearn.cluster import KMeans

sys.stdout.reconfigure(encoding="utf-8")

logging.getLogger("PIL").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    parts = []

    if h >= 1:
        parts.append(f"{int(h)}h")
    if m >= 1:
        parts.append(f"{int(m)}min")

    parts.append(f"{s:.3f}s")

    return " ".join(parts)


class CachedResult(dict):
    """A dict that mimics PaddleX result objects: dict access + .json property."""

    @property
    def json(self) -> dict[str, Any]:
        return {"res": dict(self)}


def _unwrap_legacy_cache(data: dict[str, Any]) -> dict[str, Any]:
    """If the cached dict has the old ``{"res": {...}}`` wrapper, unwrap it."""
    if isinstance(data, dict) and list(data.keys()) == ["res"]:
        return data["res"]
    return data


def _cacheable_dict(result: Any) -> dict[str, Any]:
    """Return a JSON-safe dict from *result* using PaddleX's own serializer."""
    json_data = result._to_json()
    return json_data["res"]


def compute_file_hash(filepath: str) -> str:
    """SHA256 hex digest of a file, truncated to 16 chars."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def _cache_path(
    cache_dir: str, pdf_hash: str, stage: str, page_idx: int | None = None
) -> str:
    d = os.path.join(cache_dir, pdf_hash)
    if page_idx is not None:
        return os.path.join(d, f"{stage}_page_{page_idx}.json")
    return os.path.join(d, f"{stage}.json")


def _cache_load(path: str) -> Any:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _cache_save(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def image_from_page(page: fitz.Page) -> np.ndarray:
    mat = fitz.Matrix(2, 2)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    if pm.width > 2000 or pm.height > 2000:
        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def is_inside(
    box: list[np.float64], container: list[np.float64], tol: float = 0
) -> bool:
    """
    Check if `box` is completely inside `container` (with optional tolerance).
    Boxes are given as [x1, y1, x2, y2] (x2 >= x1, y2 >= y1).
    """
    bx1, by1, bx2, by2 = box
    cx1, cy1, cx2, cy2 = container
    return (
        bx1 >= cx1 - tol and by1 >= cy1 - tol and bx2 <= cx2 + tol and by2 <= cy2 + tol
    )


def filter_toc_result(
    ocr_res: dict[str, Any],
    content_boxes: list[dict[str, Any]],
    tol: float = 20,
) -> list[dict[str, Any]]:
    rec_texts = ocr_res["rec_texts"]
    rec_boxes = ocr_res["rec_boxes"]
    results = []
    for content_box in content_boxes:
        results.append({"content_box": content_box, "rec_texts": [], "rec_boxes": []})
    for rec_text, rec_box in zip(rec_texts, rec_boxes):
        for idx, content_box in enumerate(content_boxes):
            if is_inside(rec_box, content_box["coordinate"], tol=tol):
                results[idx]["rec_texts"].append(rec_text)
                results[idx]["rec_boxes"].append(rec_box)
    return results


def deduplicate_content_boxes(
    boxes: list[dict[str, Any]], containment_threshold: float = 0.8
) -> list[dict[str, Any]]:
    """Remove boxes that are mostly contained within another box, keeping
    the higher-scoring one."""
    if len(boxes) <= 1:
        return boxes

    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        ci = boxes[i]["coordinate"]
        # area_i = (ci[2] - ci[0]) * (ci[3] - ci[1])
        for j in range(len(boxes)):
            if i == j or not keep[j]:
                continue
            cj = boxes[j]["coordinate"]
            x1 = max(ci[0], cj[0])
            y1 = max(ci[1], cj[1])
            x2 = min(ci[2], cj[2])
            y2 = min(ci[3], cj[3])
            if x1 >= x2 or y1 >= y2:
                continue
            inter_area = (x2 - x1) * (y2 - y1)
            area_j = (cj[2] - cj[0]) * (cj[3] - cj[1])
            if area_j > 0 and inter_area / area_j > containment_threshold:
                if boxes[i]["score"] >= boxes[j]["score"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [b for b, k in zip(boxes, keep) if k]


def _parse_toc_lines(
    toc_results: list[dict[str, Any]],
    page_heights: list[float] | None = None,
) -> tuple[list[dict[str, Any]], list[float] | None]:
    """Shared Steps 1-4: flatten, group lines, parse → list of parsed entries."""
    if not toc_results:
        return [], None

    # ---- Step 1: calculate page heights for y-offset ----
    if page_heights is None:
        page_heights = []
        for tp in toc_results:
            max_y = 0
            for cb in tp["content_boxes"]:
                for box in cb["rec_boxes"]:
                    if box[3] > max_y:
                        max_y = box[3]
            page_heights.append(max_y + 50)

    # ---- Step 2: flatten all items with cumulative y-offset ----
    all_items = []
    cumulative_y = 0.0
    for pi, tp in enumerate(toc_results):
        y_offset = cumulative_y
        page_idx = tp.get("page", pi)
        for cb in tp["content_boxes"]:
            cb_inner = cb.get("content_box", cb)
            cb_right = cb_inner["coordinate"][2]
            for text, box in zip(cb["rec_texts"], cb["rec_boxes"]):
                t = str(text).strip()
                if not t:
                    continue
                if re.sub(r"[\.·…\s]+", "", t) == "":
                    continue
                x1, y1, x2, y2 = box
                all_items.append(
                    {
                        "text": t,
                        "xmin": float(x1),
                        "ymin": float(y1) + y_offset,
                        "xmax": float(x2),
                        "ymax": float(y2) + y_offset,
                        "height": float(y2 - y1),
                        "x_center": float(x1 + x2) / 2.0,
                        "cb_right": float(cb_right),
                    }
                )
        if page_idx < len(page_heights):
            cumulative_y += page_heights[page_idx]
        else:
            cumulative_y += page_heights[pi]

    if not all_items:
        return [], page_heights

    # ---- Step 3: group items into lines by y-overlap ----
    all_items.sort(key=lambda b: b["ymin"])
    lines = []
    for item in all_items:
        placed = False
        for line in lines:
            line_ymin = sum(b["ymin"] for b in line) / len(line)
            line_ymax = sum(b["ymax"] for b in line) / len(line)
            line_h = line_ymax - line_ymin
            overlap_ymin = max(item["ymin"], line_ymin)
            overlap_ymax = min(item["ymax"], line_ymax)
            if overlap_ymax > overlap_ymin:
                overlap_h = overlap_ymax - overlap_ymin
                if overlap_h / max(item["height"], 1) > 0.3 or (
                    line_h > 0 and overlap_h / line_h > 0.3
                ):
                    line.append(item)
                    placed = True
                    break
        if not placed:
            lines.append([item])

    # ---- Step 4: parse each line → (title, page_num, min_xmin, avg_ymin) ----
    ignore_titles = {
        "目录",
        "目次",
        "前言",
        "序言",
        "附录",
        "索引",
        "编后记",
        "作者简介",
    }

    paren_digit_pat = re.compile(r"^[\(（]\d+[\)）]$")
    rightmost_items = []
    for line in lines:
        sorted_line = sorted(line, key=lambda b: b["xmin"])
        if sorted_line:
            rightmost_items.append(sorted_line[-1]["text"])
    paren_ratio = sum(1 for t in rightmost_items if paren_digit_pat.match(t)) / max(
        len(rightmost_items), 1
    )
    use_paren_mode = paren_ratio > 0.5

    parsed = []
    for line in lines:
        line.sort(key=lambda b: b["xmin"])

        page_num = None
        title_end = len(line)

        roman_pat = re.compile(r"^[IVXLCDMivxlcdm]+$")
        digit_pat = re.compile(r"^\d+$")
        trailed_pat = re.compile(r"^[\.…·]{1,3}\s*\d+$")
        # dot-leader + trailing digit merged in one fragment: "…………6"
        dot_leader_num_pat = re.compile(r"[\.…·]{2,}\s*(\d+)$")

        for i in range(len(line) - 1, -1, -1):
            item = line[i]
            t = item["text"]

            is_standalone = bool(digit_pat.match(t) or roman_pat.match(t))
            is_trailed = bool(trailed_pat.match(t)) if not is_standalone else False
            is_paren = (
                bool(paren_digit_pat.match(t))
                if use_paren_mode and not is_standalone
                else False
            )
            dot_m = (
                dot_leader_num_pat.search(t)
                if not is_standalone and not is_trailed and not is_paren
                else None
            )
            # dot-leader page number only when fragment touches content-box right edge
            touches_cb_right = (
                item.get("cb_right") is not None
                and item["xmax"] >= item["cb_right"] - 10
            )
            is_dot_leader = dot_m is not None and touches_cb_right

            if (
                not is_standalone
                and not is_trailed
                and not is_paren
                and not is_dot_leader
            ):
                continue
            if i > 0 and is_dot_leader is False:
                prev = line[i - 1]
                gap = item["xmin"] - prev["xmax"]
                if gap < max(item["height"] * 0.25, 5.0):
                    continue
            if is_trailed:
                page_num = int(re.search(r"\d+$", t).group(0))
            elif is_paren:
                page_num = int(re.search(r"\d+", t).group(0))
            elif is_dot_leader:
                page_num = int(dot_m.group(1))
                # remove the dot-leader + page number tail from this fragment
                trimmed = dot_leader_num_pat.sub("", t).strip()
                if trimmed:
                    line[i]["text"] = trimmed
                    title_end = i + 1
                else:
                    title_end = i
            elif t.isdigit():
                page_num = int(t)
            else:
                page_num = t.upper()
            title_end = i
            break

        title_items = line[:title_end]
        title = " ".join(b["text"] for b in title_items).strip()
        title = re.sub(r"\s+", " ", title)
        title = re.sub(r"[．….]+$", "", title)

        if not title:
            continue

        clean = re.sub(r"[\s\.·…\-]+", "", title)
        if clean in ignore_titles:
            continue

        min_xmin = (
            min(b["xmin"] for b in title_items) if title_items else line[0]["xmin"]
        )
        avg_ymin = sum(b["ymin"] for b in line) / len(line)

        parsed.append(
            {
                "title": title,
                "page_num": page_num,
                "min_xmin": min_xmin,
                "avg_ymin": avg_ymin,
            }
        )

    if not parsed:
        return [], page_heights

    parsed.sort(key=lambda x: x["avg_ymin"])
    return parsed, page_heights


def _build_tree(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Step 6: build hierarchical tree from parsed entries with .level set."""
    root = []
    stack = []
    for p in parsed:
        node = {"title": p["title"], "page_num": p["page_num"], "children": []}
        while len(stack) > p["level"]:
            stack.pop()
        if not stack:
            root.append(node)
        else:
            stack[-1]["children"].append(node)
        stack.append(node)
    return root


def _merge_page_trees(
    per_page_trees: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge per-page mini-trees into a single global tree.

    A root entry from a later page that looks like a section continuation
    (not a new chapter / back matter) is attached under the last chapter
    seen in the merged output so far.
    """
    ch_pat = re.compile(r"^第[一二三四五六七八九十\d]+章(?!习题)")
    sec_pat = re.compile(r"^第[一二三四五六七八九十\d]+节")
    back_pat = re.compile(r"^附录|^参考书目|^参考文献|^名词索引|^索引|^学时分配")

    def _is_chapter_like(node: dict[str, Any]) -> bool:
        """Entry that looks like a top-level chapter (English or Chinese)."""
        return (
            ch_pat.match(node["title"]) is not None
            or back_pat.match(node["title"]) is not None
            or (
                # English: "1. Title", "2. Title" with single number — likely chapter
                re.match(r"^\d+\.\s", node["title"])
                and not re.match(r"^\d+\.\d+", node["title"])
            )
        )

    merged = []
    for page_tree in per_page_trees:
        for node in page_tree:
            # section-like = NOT a chapter/back-matter but looks like a subsection
            sec_pattern = (
                sec_pat.match(node["title"])
                or re.match(r"^\d+\.\d+", node["title"])
                or re.match(r"^本章小结|^练习|^总习题|^思考题", node["title"])
                or re.match(r"^第[一二三四五六七八九十\d]+章习题", node["title"])
                or re.match(r"^[一二三四五六七八九十]、|^习题\s*\d+", node["title"])
            )
            is_section_like = not _is_chapter_like(node) and sec_pattern

            if is_section_like and merged:
                last = merged[-1]
                if _is_chapter_like(last):
                    last["children"].append(node)
                    if last["children"]:
                        last["children"].sort(
                            key=lambda c: _section_sort_key(c["title"])
                        )
                    continue
            merged.append(node)
    return merged


def _merge_content_box_trees(
    content_boxes: list[dict[str, Any]],
    cb_trees: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge per-content-box mini-trees within a single page.

    Content boxes are sorted left-to-right by column, then top-to-bottom
    within each column.  Trees are concatenated in that order.
    """

    def _get_coord(cb: dict[str, Any]) -> list[float]:
        # cb may be the inner content_box dict, or the wrapper with "content_box" key
        inner = cb.get("content_box", cb)
        return inner["coordinate"]

    paired = list(zip(content_boxes, cb_trees))
    paired.sort(
        key=lambda p: (
            round((_get_coord(p[0])[0] + _get_coord(p[0])[2]) / 2 / 200),
            _get_coord(p[0])[1],
        )
    )
    merged = []
    for _, tree in paired:
        merged.extend(tree)
    return merged


def reconstruct_toc(
    toc_results: list[dict[str, Any]],
    page_heights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Semantic-pattern-based TOC reconstruction."""
    parsed, _ = _parse_toc_lines(toc_results, page_heights)
    if not parsed:
        return []

    # ---- Step 5: semantic patterns + xmin proximity fallback ----
    ch_pat = re.compile(r"^第[一二三四五六七八九十\d]+章(?!习题)")
    sec_pat = re.compile(r"^第[一二三四五六七八九十\d]+节")
    sub_pat = re.compile(r"^[一二三四五六七八九十]、")
    exercise_sec_pat = re.compile(r"^习题\s*\d+")
    num_sec_pat = re.compile(r"^\d+\.\d+")
    num_sub_pat = re.compile(r"^\d+\.\d+\.\d+")
    num_sec_single_pat = re.compile(r"^\*?\d+\.")
    back_pat = re.compile(r"^附录|^参考书目|^参考文献|^名词索引|^索引|^学时分配")
    ch_exercise_pat = re.compile(r"^第[一二三四五六七八九十\d]+章习题")
    summary_pat = re.compile(r"^本章小结|^练习|^总习题|^思考题")

    for p in parsed:
        if ch_pat.match(p["title"]):
            p["level"] = 0
        elif re.match(r"^\d+\.\s", p["title"]) and not re.match(
            r"^\d+\.\d+", p["title"]
        ):
            p["level"] = 0
        elif num_sub_pat.match(p["title"]):
            p["level"] = 2
        elif (
            sec_pat.match(p["title"])
            or num_sec_pat.match(p["title"])
            or num_sec_single_pat.match(p["title"])
            or exercise_sec_pat.match(p["title"])
        ):
            p["level"] = 1
        elif sub_pat.match(p["title"]):
            p["level"] = 2
        elif summary_pat.match(p["title"]) or ch_exercise_pat.match(p["title"]):
            p["level"] = 1
        elif back_pat.match(p["title"]):
            p["level"] = 0
        else:
            p["level"] = -1

    unassigned = [p for p in parsed if p["level"] == -1]
    if unassigned:
        assigned_ratio = 1.0 - len(unassigned) / len(parsed)
        if assigned_ratio == 0:
            if len(parsed) >= 3:
                x_arr = np.array([[p["min_xmin"]] for p in parsed], dtype=np.float64)
                if float(np.std(x_arr)) > 12:
                    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x_arr)
                    right_label = int(np.argmax(km.cluster_centers_))
                    for p, lb in zip(parsed, km.labels_):
                        p["level"] = 1 if int(lb) == right_label else 0
                else:
                    for p in parsed:
                        p["level"] = 0
        else:
            l0_x = [p["min_xmin"] for p in parsed if p["level"] == 0]
            l1_x = [p["min_xmin"] for p in parsed if p["level"] == 1]
            l2_x = [p["min_xmin"] for p in parsed if p["level"] == 2]
            avg_l0 = np.mean(l0_x) if l0_x else None
            avg_l1 = np.mean(l1_x) if l1_x else None
            avg_l2 = np.mean(l2_x) if l2_x else None

            for p in unassigned:
                x = p["min_xmin"]
                if avg_l2 is not None and abs(x - avg_l2) < 30:
                    p["level"] = 2
                elif avg_l1 is not None and abs(x - avg_l1) < 30:
                    p["level"] = 1
                elif avg_l0 is not None and abs(x - avg_l0) < 30:
                    p["level"] = 0
                elif avg_l2 is not None and x > avg_l2 + 20:
                    p["level"] = 3
                elif avg_l1 is not None and x > avg_l1 + 30:
                    p["level"] = 2
                elif avg_l0 is not None and x > avg_l0 + 30:
                    p["level"] = 1
                else:
                    p["level"] = 0

    return _build_tree(parsed)


def reconstruct_toc_indent(
    toc_results: list[dict[str, Any]],
    page_heights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Pure indentation-based TOC reconstruction — per-CB clustering + merge."""
    if not toc_results:
        return []

    def _indent_levels(plist: list[dict[str, Any]]) -> None:
        all_xmins = np.array([p["min_xmin"] for p in plist])
        sorted_x = np.sort(all_xmins)
        diffs = np.diff(sorted_x)
        if len(diffs) > 0:
            p50 = float(np.percentile(diffs, 50))
            p90 = float(np.percentile(diffs, 90))
            threshold = p50 + (p90 - p50) * 2.0
            boundaries = [float(sorted_x[0])]
            for i, d in enumerate(diffs):
                if d > max(threshold, 15):
                    boundaries.append(float(sorted_x[i + 1]))
            for p in plist:
                x = p["min_xmin"]
                dists = [abs(x - b) for b in boundaries]
                p["level"] = int(np.argmin(dists))
        else:
            for p in plist:
                p["level"] = 0

    per_page_trees = []
    for tp in toc_results:
        content_boxes = tp["content_boxes"]
        if len(content_boxes) <= 1:
            parsed, _ = _parse_toc_lines([tp], None)
            if not parsed:
                continue
            _indent_levels(parsed)
            per_page_trees.append(_build_tree(parsed))
        else:
            cb_boxes = []
            cb_trees = []
            for cb in content_boxes:
                single_cb_page = [{"page": tp["page"], "content_boxes": [cb]}]
                parsed, _ = _parse_toc_lines(single_cb_page, None)
                if not parsed:
                    continue
                _indent_levels(parsed)
                cb_boxes.append(cb)
                cb_trees.append(_build_tree(parsed))
            if cb_trees:
                per_page_trees.append(_merge_content_box_trees(cb_boxes, cb_trees))

    return _merge_page_trees(per_page_trees)


def reconstruct_toc1(
    toc_results: list[dict[str, Any]],
    page_heights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Per-page level detection + cross-page merge.

    Level detection (semantic + gap clustering) runs inside each page to avoid
    x-coordinate shifts between pages from corrupting the indentation clusters.
    Page-level mini-trees are then merged into the final global tree.
    """
    if not toc_results:
        return []

    # ---- shared patterns (same as reconstruct_toc) ----
    ch_pat = re.compile(r"^第[一二三四五六七八九十\d]+章(?!习题)")
    sec_pat = re.compile(r"^第[一二三四五六七八九十\d]+节")
    sub_pat = re.compile(r"^[一二三四五六七八九十]、")
    exercise_sec_pat = re.compile(r"^习题\s*\d+")
    num_sec_pat = re.compile(r"^\d+\.\d+")
    num_sub_pat = re.compile(r"^\d+\.\d+\.\d+")
    num_sec_single_pat = re.compile(r"^\*?\d+\.")
    back_pat = re.compile(r"^附录|^参考书目|^参考文献|^名词索引|^索引|^学时分配")
    ch_exercise_pat = re.compile(r"^第[一二三四五六七八九十\d]+章习题")
    summary_pat = re.compile(r"^本章小结|^练习|^总习题|^思考题")

    # helper: level detection for a parsed list (semantic + gap fallback)
    def _assign_levels(plist: list[dict[str, Any]]) -> None:
        for p in plist:
            if ch_pat.match(p["title"]):
                p["level"] = 0
            elif re.match(r"^\d+\.\s", p["title"]) and not re.match(
                r"^\d+\.\d+", p["title"]
            ):
                p["level"] = 0
            elif num_sub_pat.match(p["title"]):
                p["level"] = 2
            elif (
                sec_pat.match(p["title"])
                or num_sec_pat.match(p["title"])
                or num_sec_single_pat.match(p["title"])
                or exercise_sec_pat.match(p["title"])
            ):
                p["level"] = 1
            elif sub_pat.match(p["title"]):
                p["level"] = 2
            elif summary_pat.match(p["title"]) or ch_exercise_pat.match(p["title"]):
                p["level"] = 1
            elif back_pat.match(p["title"]):
                p["level"] = 0
            else:
                p["level"] = -1

        unassigned = [p for p in plist if p["level"] == -1]
        if not unassigned:
            return
        assigned_ratio = 1.0 - len(unassigned) / max(len(plist), 1)
        if assigned_ratio == 0:
            if len(plist) >= 3:
                x_arr = np.array([[p["min_xmin"]] for p in plist], dtype=np.float64)
                if float(np.std(x_arr)) > 12:
                    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x_arr)
                    right_label = int(np.argmax(km.cluster_centers_))
                    for p, lb in zip(plist, km.labels_):
                        p["level"] = 1 if int(lb) == right_label else 0
                else:
                    for p in plist:
                        p["level"] = 0
        else:
            all_xmins = np.array([p["min_xmin"] for p in plist])
            sorted_x = np.sort(all_xmins)
            diffs = np.diff(sorted_x)
            if len(diffs) > 0:
                p50 = float(np.percentile(diffs, 50))
                p90 = float(np.percentile(diffs, 90))
                threshold = p50 + (p90 - p50) * 2.0
                boundaries = [float(sorted_x[0])]
                for i, d in enumerate(diffs):
                    if d > max(threshold, 15):
                        boundaries.append(float(sorted_x[i + 1]))
                for p in unassigned:
                    x = p["min_xmin"]
                    dists = [abs(x - b) for b in boundaries]
                    p["level"] = int(np.argmin(dists))
            else:
                for p in unassigned:
                    p["level"] = 0

    per_page_trees = []

    for tp in toc_results:
        content_boxes = tp["content_boxes"]
        if len(content_boxes) <= 1:
            # single content box — process whole page as before
            parsed, _ = _parse_toc_lines([tp], None)
            if not parsed:
                continue
            _assign_levels(parsed)
            per_page_trees.append(_build_tree(parsed))
        else:
            # multiple content boxes — process each independently, then merge
            cb_boxes = []
            cb_trees = []
            for cb in content_boxes:
                single_cb_page = [{"page": tp["page"], "content_boxes": [cb]}]
                parsed, _ = _parse_toc_lines(single_cb_page, None)
                if not parsed:
                    continue
                _assign_levels(parsed)
                cb_boxes.append(cb)
                cb_trees.append(_build_tree(parsed))
            if cb_trees:
                per_page_trees.append(_merge_content_box_trees(cb_boxes, cb_trees))

    return _merge_page_trees(per_page_trees)


_CN_NUM = "一二三四五六七八九十"


def _cn_to_int(s: str) -> int:
    if s in _CN_NUM:
        return _CN_NUM.index(s) + 1
    if len(s) == 2 and s[0] == "十":
        return 10 + (_CN_NUM.index(s[1]) + 1 if s[1] in _CN_NUM else 0)
    if len(s) == 2 and s[1] == "十":
        return (_CN_NUM.index(s[0]) + 1) * 10
    return 0


def _section_sort_key(title: str) -> tuple:
    """Extract numeric sort key from a title like '*4.13' or '4.7.2' or '习题1-5'."""
    cn_sec = re.search(r"第([一二三四五六七八九十]+)节", title)
    if cn_sec:
        return (_cn_to_int(cn_sec.group(1)),)
    cn_ch = re.search(r"第([一二三四五六七八九十]+)章", title)
    if cn_ch:
        return (_cn_to_int(cn_ch.group(1)),)
    ex = re.match(r"^习题(\d+)[-−](\d+)", title)
    if ex:
        return (int(ex.group(1)), int(ex.group(2)))
    zong = re.match(r"^总习题([一二三四五六七八九十\d]+)", title)
    if zong:
        ch = zong.group(1)
        ch_num = _cn_to_int(ch) if ch[0] in _CN_NUM else int(ch)
        return (ch_num, 9000)
    m = re.search(r"(\d+(?:\.\d+)*)", title)
    if not m:
        return (9999, title)
    parts = m.group(1).split(".")
    return tuple(int(x) for x in parts)


def repair_toc_tree(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Post-process: fix misplaced entries at root level based on semantic rules."""
    sec_like = re.compile(r"^\*?\d+\.\d+")
    ch_like = re.compile(r"^第[一二三四五六七八九十\d]+章")

    def _is_chapter(node: dict[str, Any]) -> bool:
        return bool(ch_like.match(node["title"]))

    fixed_root = []
    last_chapter = None
    for node in tree:
        if sec_like.match(node["title"]) and not _is_chapter(node):
            if last_chapter is not None:
                last_chapter["children"].append(node)
                continue
        if _is_chapter(node):
            last_chapter = node
        fixed_root.append(node)

    # sort children of all chapter nodes
    for node in fixed_root:
        if _is_chapter(node) and node.get("children"):
            node["children"].sort(key=lambda c: _section_sort_key(c["title"]))

    return fixed_root


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        # 处理 NumPy 浮点数
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        # 处理 NumPy 整数
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        # 处理 NumPy 数组
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 其他类型交还给父类处理
        return super().default(obj)


class TocNode(TypedDict):
    title: str
    page_num: int
    children: list["TocNode"]


def print_toc_result(toc_result: list[TocNode], indent: int = 0) -> None:
    for node in toc_result:
        print("  " * indent + node["title"] + f" (page {node['page_num']})")
        print_toc_result(node["children"], indent + 2)


class NumberPageResult(TypedDict):
    width: float
    height: float
    y_offset: int
    parsing_res_list: dict
    pdf_page_idx: int


def _roman_to_int(s: str) -> int | None:
    """Convert a Roman numeral string to integer, or None if invalid."""
    roman_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    s = s.upper().replace(" ", "")
    if not re.match(r"^[IVXLCDM]+$", s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = roman_map[ch]
        total += -v if v < prev else v
        prev = v
    return total


def get_page_offset(number_page_results: list[NumberPageResult]) -> int:
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
        # 平票时取中位数更稳
        return int(statistics.median(page_offsets))


def add_bookmarks_to_pdf(
    doc: fitz.Document,
    toc_tree: list[dict[str, Any]],
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

    def _first_page_num(node: dict[str, Any]) -> int | str | None:
        """Find the first valid page number in a subtree."""
        pn = node.get("page_num")
        if isinstance(pn, int):
            return pn
        for child in node.get("children", []):
            result = _first_page_num(child)
            if result is not None:
                return result
        return None

    def _build_outline(node: dict[str, Any], level: int) -> list[list[Any]]:
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


def get_toc_pages(
    imgs: list[np.ndarray],
    ocr_model: LayoutDetection,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        for box in boxes:
            if box["label"] == "content":
                content_boxes.append(box)
        if content_boxes:
            content_boxes = deduplicate_content_boxes(content_boxes)
            toc_pages.append({"page": res_i, "content_boxes": content_boxes})
    # 可能的带有页码的页
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
    toc_pages: list[dict[str, Any]],
    page_imgs: list[np.ndarray],
    ocr_model: PaddleOCR,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict[str, Any]]:
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


def build_toc_local_ocr(
    toc_pages: list[dict[str, Any]],
    page_imgs: list[np.ndarray],
    ocr_model: PaddleOCR,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Build TOC tree from local OCR results using heuristic parsing."""
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

    # semantic-based
    # toc_tree = reconstruct_toc(toc_results, page_heights)
    # toc_tree = repair_toc_tree(toc_tree)
    # os.makedirs(args.output, exist_ok=True)
    # output_path = os.path.join(args.output, "toc_tree.json")
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(toc_tree, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    # print(f"[semantic] TOC saved to: {output_path}")

    # per-page level detection + cross-page merge
    toc_tree1 = reconstruct_toc1(toc_results, page_heights)
    toc_tree1 = repair_toc_tree(toc_tree1)
    if do_debug:
        perpage_path = os.path.join(output, "toc_tree1.json")
        with open(perpage_path, "w", encoding="utf-8") as f:
            json.dump(toc_tree1, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"[per-page] TOC saved to: {perpage_path}")

    # indent-based (pure indentation clustering, no semantics)
    # toc_tree_indent = reconstruct_toc_indent(toc_results, page_heights)
    # toc_tree_indent = repair_toc_tree(toc_tree_indent)
    # indent_path = os.path.join(args.output, "toc_tree_indent.json")
    # with open(indent_path, "w", encoding="utf-8") as f:
    #     json.dump(toc_tree_indent, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    # print(f"[indent]  TOC saved to: {indent_path}")
    return toc_tree1


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
    user_content: str | list[dict[str, Any]],
) -> dict[str, Any] | list[Any]:
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
    toc_pages: list[dict[str, Any]],
    page_imgs: list[np.ndarray],
    ocr_model: PaddleOCR,
    do_debug: bool = False,
    output: str = "output",
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
) -> list[dict[str, Any]]:
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
    toc_pages: list[dict[str, Any]],
    page_imgs: list[np.ndarray],
    do_debug: bool = False,
    output: str = "output",
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
) -> list[dict[str, Any]]:
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


def ocr_number_pages(
    toc_pages: list[dict[str, Any]],
    page_imgs: list[np.ndarray],
    number_pages: list[dict[str, Any]],
    ocr_model: PPStructureV3,
    do_debug: bool = False,
    output: str = "output",
    half_img: bool = False,
    cache_dir: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict[str, Any]]:
    last_toc_page_idx = toc_pages[-1]["page"]
    page_sample_count = 5
    number_page_results = []
    for number_page in number_pages:
        if page_sample_count == 0:
            break
        page_idx = number_page["page"]
        if page_idx <= last_toc_page_idx:
            continue
        page_sample_count -= 1
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


def setup_logger(log_dir: str) -> None:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename = os.path.join(
        os.path.dirname(__file__), log_dir, f"{Path(__file__).stem}.log"
    )
    handler = logging.FileHandler(filename, encoding="utf-8", mode="w")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[handler],
    )


_BOS_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model"
)
_BOS_VERSION = "paddle3.0.0"


def make_sure_model_exists(model_dir: str, model_name: str) -> None:
    target = os.path.join(model_dir, model_name)
    if os.path.isdir(target):
        return

    # Check PaddleX official cache first (copy if found, to avoid re-download)
    cache_home = os.environ.get("PADDLE_PDX_CACHE_HOME", "")
    if cache_home:
        cached = os.path.join(cache_home, "official_models", model_name)
        if os.path.isdir(cached):
            os.makedirs(model_dir, exist_ok=True)
            shutil.copytree(cached, target)
            logger.info("Copied model %s from cache to %s", model_name, target)
            return

    # Not in cache — download directly to target directory
    os.makedirs(model_dir, exist_ok=True)
    url = f"{_BOS_BASE_URL}/{_BOS_VERSION}/{model_name}_infer.tar"
    logger.info("Downloading model %s from %s", model_name, url)
    download_and_extract(url, model_dir, model_name)


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
) -> tuple[str, float]:
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
        return "", 0

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

    end_time = time.perf_counter()
    time_cost = end_time - start_time
    logger.debug(f"process {Path(input).stem} cost: {time_cost:.2f} seconds")
    return pdf_bookmarks_path, time_cost


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input PDF file")
    parser.add_argument("--output", type=str, default="output directory", help="output directory")
    parser.add_argument("--log_dir", type=str, default="log", help="log directory")
    parser.add_argument("--model_dir", type=str, default="./models", help="where the OCR models are stored")
    parser.add_argument("--cache_dir", type=str, default="./.ocr_cache", help="where the OCR cache is stored")
    parser.add_argument("--api_base_url", type=str, default=None, help="OpenAI API base url")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--llm_name", type=str, default="", help="text LLM, like deepseek-v4-flash")
    parser.add_argument("--vllm_name", type=str, default="", help="visual LLM or muti-modal LLM, like qwen3.6-flash")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.model_dir:
        print("model dir required")
        return
    setup_logger(args.log_dir)

    api_base_url = args.api_base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    if api_base_url and api_key and args.vllm_name:
        toc_strategy = "vllm"
    elif api_base_url and api_key and args.llm_name:
        toc_strategy = "llm"
    else:
        toc_strategy = "local_ocr"

    logger.info("toc_strategy=%s", toc_strategy)

    pdf_bookmarks_path, time_cost = bookmark_pdf(
        args.input,
        args.output,
        args.model_dir,
        do_debug=args.debug,
        cache_dir=args.cache_dir,
        toc_strategy=toc_strategy,
        api_base_url=api_base_url,
        api_key=api_key,
        llm_name=args.llm_name,
        vllm_name=args.vllm_name,
    )
    print(
        f"Bookmarked PDF saved to: {pdf_bookmarks_path}, Time elapsed: {format_duration(time_cost)}"
    )


if __name__ == "__main__":
    main()
