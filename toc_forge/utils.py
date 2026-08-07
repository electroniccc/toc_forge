"""Utility functions: caching, image processing, models, and helpers."""

import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import pymupdf
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.CRITICAL)

# ---- Model download constants ----
_BOS_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model"
)
_BOS_VERSION = "paddle3.0.0"

# ---- Chinese numeral helpers ----
_CN_NUM = "一二三四五六七八九十"


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


# ---- Caching ----

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


# ---- Image processing ----

def image_from_page(page: pymupdf.Page) -> np.ndarray:
    mat = pymupdf.Matrix(2, 2)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    if pm.width > 2000 or pm.height > 2000:
        pm = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False)
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


# ---- Numerals and sorting ----

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
    cn_ch = re.search(r"第([一二三四五六七八九十]+)章(?!习题)", title)
    if cn_ch:
        return (_cn_to_int(cn_ch.group(1)),)
    ch_ex = re.match(r"^第([一二三四五六七八九十\d]+)章习题", title)
    if ch_ex:
        cn = ch_ex.group(1)
        ch_num = _cn_to_int(cn) if cn[0] in _CN_NUM else int(cn)
        return (ch_num, 9000)
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
        return (9999,)
    parts = m.group(1).split(".")
    return tuple(int(x) for x in parts)


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


# ---- JSON and data types ----

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class TocNode(dict):
    """TypedDict-alike for TOC tree nodes."""

    def __init__(
        self,
        title: str = "",
        page_num: int | None = None,
        children: list["TocNode"] | None = None,
    ):
        super().__init__(title=title, page_num=page_num, children=children or [])


class NumberPageResult(dict):
    """TypedDict-alike for page-number detection results."""


def print_toc_result(toc_result: list[dict[str, Any]], indent: int = 0) -> None:
    for node in toc_result:
        print("  " * indent + node["title"] + f" (page {node['page_num']})")
        print_toc_result(node["children"], indent + 2)


# ---- Logging and model management ----

def setup_logger(log_dir: str) -> None:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename = os.path.join(log_dir, "toc_forge.log")
    handler = logging.FileHandler(filename, encoding="utf-8", mode="w")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[handler],
    )


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
    from paddlex.utils.download import download_and_extract

    download_and_extract(url, model_dir, model_name)
