"""TOC parsing strategies: heuristic tree reconstruction from OCR results."""

import logging
import re
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from .utils import _section_sort_key

logger = logging.getLogger(__name__)

# English chapter-like title patterns (used by _merge_page_trees' chapter
# detection and by the level floors in _assign_levels).  Covers "1.
# Introduction" (Kibble style), "1 Introduction" (OpenStax/Morin style),
# "Chapter 1", "Part I", "A Table of Integrals", and the front/back-matter
# words (Preface, Appendix, ...).
_EN_CHAPTER_PAT = re.compile(
    r"^\d+\.\s+[A-Z]"  # "1. Introduction"
    r"|^\d+\s+[A-Z]"  # "1 Introduction"
    r"|^Chapter\s+\d+"
    r"|^Part\s+[IVX\d]+"
    r"|^(Preface|Introduction|Appendix|Index|Bibliography|References|Answer Key|Errata)(\s|$)"
    r"|^[A-Z]\s"
)
# number-led subset used as a level-0 floor.  The front/back-matter words are
# excluded on purpose: OpenStax's "Introduction 7" is a per-chapter subheading
# that must stay nested, not a root entry.
_EN_CHAPTER_FLOOR = re.compile(
    r"^\d+\.\s+[A-Z]"
    r"|^\d+\s+[A-Z]"
    r"|^Chapter\s+\d+"
    r"|^Part\s+[IVX\d]+"
    r"|^[A-Z]\s"
)


def _parse_toc_lines(
    toc_results: list[dict],
    page_heights: list[float] | None = None,
) -> tuple[list[dict], list[float] | None]:
    """Shared Steps 1-4: flatten, group lines, parse -> list of parsed entries."""
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
        for cbi, cb in enumerate(tp["content_boxes"]):
            cb_inner = cb.get("content_box", cb)
            cb_right = cb_inner["coordinate"][2]
            cb_label = cb_inner.get("label", "content")
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
                        "cb_label": cb_label,
                        "page_idx": page_idx,
                        "cb_idx": cbi,
                    }
                )
        if page_idx < len(page_heights):
            cumulative_y += page_heights[page_idx]
        else:
            cumulative_y += page_heights[pi]

    if not all_items:
        return [], page_heights

    # ---- Step 3: group items into lines by y-overlap (within each content box) ----
    # Partition items by (page_idx, cb_idx) to avoid cross-column contamination
    cb_buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in all_items:
        cb_buckets[(item["page_idx"], item["cb_idx"])].append(item)

    lines = []
    for bucket_items in cb_buckets.values():
        bucket_items.sort(key=lambda b: b["ymin"])
        for item in bucket_items:
            placed = False
            for line in lines:
                # Only consider lines from the same content box
                if line[0]["page_idx"] != item["page_idx"] or line[0]["cb_idx"] != item["cb_idx"]:
                    continue
                # Check y-overlap against the line's average y-range
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

    # ---- Step 4: parse each line -> (title, page_num, min_xmin, avg_ymin) ----
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
    # English TOCs often print "Title 8" with the page number in the same OCR
    # fragment as the title (no dot leader), e.g. "Preface 1", or glued:
    # "The Limit Laws140".  When most rightmost fragments on the page end in
    # a 1-3 digit number, enable splitting the trailing number off as the
    # page number.
    trailing_ratio = sum(
        1 for t in rightmost_items if re.search(r"\d{1,3}$", t)
    ) / max(len(rightmost_items), 1)
    use_trailing_num = trailing_ratio > 0.5

    parsed = []
    for line in lines:
        line.sort(key=lambda b: b["xmin"])

        page_num = None
        title_end = len(line)

        roman_pat = re.compile(r"^[IVXLCDMivxlcdm]+$")
        digit_pat = re.compile(r"^\d+$")
        # roman chapter + within-chapter page: "I-1", "II-24" — kept as a
        # string page number; the page-map module converts it to cumulative
        # Arabic before bookmark injection
        roman_arabic_pat = re.compile(r"^[IVXLCDMivxlcdm]+-\d+$")
        # dot chars: ASCII dot, fullwidth dot (U+FF0E), ellipsis, middle dot
        _DOT_CHARS = ".．…·"
        trailed_pat = re.compile(rf"^[{_DOT_CHARS}]{{1,3}}\s*\d+$")
        # dot-leader + trailing digit merged in one fragment: "…………6"
        dot_leader_num_pat = re.compile(rf"[{_DOT_CHARS}]{{2,}}\s*(\d+)$")

        for i in range(len(line) - 1, -1, -1):
            item = line[i]
            t = item["text"]
            # 页码只可能在行内最右片段：行首的独立数字（如 "1 Functions and
            # Graphs 7" 里的 "1"）是章节编号，不是页码
            is_rightmost = i == len(line) - 1

            is_standalone = is_rightmost and bool(
                digit_pat.match(t) or roman_pat.match(t)
            )
            is_roman_arabic = is_rightmost and bool(roman_arabic_pat.match(t))
            is_trailed = bool(trailed_pat.match(t)) if not is_standalone else False
            is_paren = (
                bool(paren_digit_pat.match(t))
                if use_paren_mode and is_rightmost and not is_standalone
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
                and not is_roman_arabic
                and not is_trailed
                and not is_paren
                and not is_dot_leader
                and not (use_trailing_num and is_rightmost)
            ):
                continue
            # gap 检查拦"紧贴标题的数字"（如 "Chapter 1"）；但 trailing_num
            # 模式下右端数字大多是页码，且 OpenStax 类目录的页码 OCR 框常与
            # 标题框 x 重叠（"Limits" 与 "105"），此时不拦
            if (
                i > 0
                and is_dot_leader is False
                and not (use_trailing_num and is_rightmost)
            ):
                prev = line[i - 1]
                gap = item["xmin"] - prev["xmax"]
                if gap < max(item["height"] * 0.25, 5.0):
                    continue
            if is_trailed:
                page_num = int(re.search(r"\d+$", t).group(0))
                title_end = i
            elif is_paren:
                page_num = int(re.search(r"\d+", t).group(0))
                title_end = i
            elif is_dot_leader:
                page_num = int(dot_m.group(1))
                trimmed = dot_leader_num_pat.sub("", t).strip()
                if trimmed:
                    line[i]["text"] = trimmed
                    title_end = i + 1
                else:
                    title_end = i
            elif t.isdigit():
                page_num = int(t)
                title_end = i
            elif is_roman_arabic:
                # "I-1" — keep as string; the page map converts it
                page_num = t.upper()
                title_end = i
            elif use_trailing_num and is_rightmost and re.search(r"\d", t):
                # English TOCs: page number in the same fragment as the title,
                # space-separated ("Preface 1", "1.1 Review of Functions 8") or
                # glued ("The Limit Laws140")
                m = re.match(r"^(.*[^\d\s])\s+(\d{1,3})$", t)
                if not m:
                    # glued digits: the title part must end in a letter /
                    # CJK char so pure numbering like "1.1" is not split
                    m = re.match(r"^(.+?[A-Za-z一-鿿])(\d{1,3})$", t)
                if m:
                    line[i]["text"] = m.group(1)
                    page_num = int(m.group(2))
                    title_end = i + 1
                else:
                    continue
            else:
                page_num = t.upper()
                title_end = i
            break

        title_items = line[:title_end]
        title = " ".join(b["text"] for b in title_items).strip()
        title = re.sub(r"\s+", " ", title)
        title = re.sub(r"[．….]+$", "", title)
        title = re.sub(r"^[\*•·]\s*", "", title)
        # Extract parenthesized page number from title tail: "概述 …( 1 )" -> page 1
        if page_num is None:
            m = re.search(r"[\(（]\s*(\d+)\s*[\)）]\s*$", title)
            if m:
                page_num = int(m.group(1))
                title = re.sub(r"\s*[\(（]\s*\d+\s*[\)）]\s*$", "", title)

        # Fallback: trailing dot-leader + digits glued to title tail
        # e.g. "1.1 关于机器学习……2" -> title "1.1 关于机器学习", page 2
        if page_num is None:
            m = re.search(rf"[{_DOT_CHARS}]{{1,}}\s*(\d+)\s*$", title)
            if m:
                page_num = int(m.group(1))
                title = re.sub(rf"[{_DOT_CHARS}]{{1,}}\s*\d+\s*$", "", title)

        # Re-strip trailing dot leaders exposed after parenthesized-number removal
        title = re.sub(r"[．….·\s]+$", "", title)
        # Strip trailing colon / semicolon (OCR artifacts from stray fragments)
        title = re.sub(r"\s*[:：;；]\s*$", "", title)

        if not title:
            continue

        # Skip page-header artifacts: Roman numeral + optional separator + 目录
        # e.g. "xviii目录", "xviii|目录"
        if re.search(r"[IVXLCDMivxlcdm]+\s*[|｜]?\s*(?:目录|目次)", title):
            continue

        clean = re.sub(rf"[\s{re.escape(_DOT_CHARS)}\-]+", "", title)
        if clean in ignore_titles:
            continue

        min_xmin = (
            min(b["xmin"] for b in title_items) if title_items else line[0]["xmin"]
        )
        avg_ymin = sum(b["ymin"] for b in line) / len(line)
        cb_label = line[0].get("cb_label", "content")

        parsed.append(
            {
                "title": title,
                "page_num": page_num,
                "min_xmin": min_xmin,
                "avg_ymin": avg_ymin,
                "cb_label": cb_label,
            }
        )

    if not parsed:
        return [], page_heights

    parsed.sort(key=lambda x: x["avg_ymin"])
    return parsed, page_heights


def _build_tree(parsed: list[dict]) -> list[dict]:
    """Step 6: build hierarchical tree from parsed entries with .level set.

    Each entry is attached as a child of the most recent entry with a
    *strictly smaller* level.  Entries at the same level are siblings,
    not nested.
    """
    root = []
    stack: list[dict] = []  # each item: {"node": ..., "level": ...}
    for p in parsed:
        node = {"title": p["title"], "page_num": p["page_num"], "children": []}
        while stack and stack[-1]["level"] >= p["level"]:
            stack.pop()
        if not stack:
            root.append(node)
        else:
            stack[-1]["node"]["children"].append(node)
        stack.append({"node": node, "level": p["level"]})
    return root


def _merge_page_trees(
    per_page_trees: list[list[dict]],
) -> list[dict]:
    """Merge per-page mini-trees into a single global tree.

    A root entry from a later page that looks like a section continuation
    (not a new chapter / back matter) is attached under the last chapter
    seen in the merged output so far.  Exercises / summaries are placed
    under the matching parent section when one can be identified.
    """
    ch_pat = re.compile(r"^第[一二三四五六七八九十\d]+(?:章(?!习题)|篇)")
    sec_pat = re.compile(r"^第[一二三四五六七八九十\d]+节")
    back_pat = re.compile(r"^附录|^参考书目|^参考文献|^名词索引|^索引|^学时分配")
    # English chapter-like titles come from the module-level _EN_CHAPTER_PAT
    # (used below in _is_chapter_like)

    _cn_num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
               "八": 8, "九": 9, "十": 10}
    _ex_num_pat = re.compile(r"^习题\s*(\d+)[-−](\d+)")
    _sec_num_pat = re.compile(r"^第([一二三四五六七八九十\d]+)节")

    def _is_chapter_like(node: dict) -> bool:
        """Entry that looks like a top-level chapter (Chinese 第X章/第X篇,
        or an English numbered/headed chapter entry)."""
        return (
            ch_pat.match(node["title"]) is not None
            or back_pat.match(node["title"]) is not None
            or _EN_CHAPTER_PAT.match(node["title"]) is not None
        )

    def _find_section_child(children: list[dict], sec_num: int) -> dict | None:
        """Find the N-th section child by extracting its ordinal number."""
        for child in children:
            m = _sec_num_pat.match(child["title"])
            if m:
                cn = m.group(1)
                if cn.isdigit():
                    if int(cn) == sec_num:
                        return child
                elif cn in _cn_num and _cn_num[cn] == sec_num:
                    return child
        return None

    def _parent_section(node: dict, chapter_children: list[dict]) -> dict | None:
        """Try to find the parent section for an exercise node.

        ``总习题`` and similar summaries stay at chapter level (return None).
        """
        ex_m = _ex_num_pat.match(node["title"])
        if ex_m:
            return _find_section_child(chapter_children, int(ex_m.group(2)))
        return None

    merged = []
    for page_tree in per_page_trees:
        for node in page_tree:
            # section-like = NOT a chapter/back-matter but looks like it belongs
            # under a chapter (sections, exercises, summaries, numbered entries)
            sec_pattern = (
                sec_pat.match(node["title"])
                or re.match(r"^\d+\.\d+", node["title"])
                or re.match(r"^本章小结|^练习|^总习题|^思考题", node["title"])
                or re.match(r"^第[一二三四五六七八九十\d]+章习题", node["title"])
                or re.match(r"^[一二三四五六七八九十]、|^习题\s*\d+", node["title"])
                or re.match(r"^\d+\s*\S", node["title"])      # "1 Title"
                or re.match(r"^小结$|^习题$", node["title"])   # standalone
                or re.match(r"^[A-Z]\.\d", node["title"])     # "A.2" appendix section
                or re.match(
                    r"^Chapter Review|^Key Terms|^Key Equations|"
                    r"^Review Exercises|^Chapter Exercises",
                    node["title"],
                )  # English per-chapter back matter
            )
            is_section_like = not _is_chapter_like(node) and sec_pattern

            if is_section_like and merged:
                last = merged[-1]
                if _is_chapter_like(last):
                    children = last.get("children", [])
                    parent = _parent_section(node, children)
                    # If no exercise parent matched and this entry is a
                    # sub-item (not a new section/summary header), attach
                    # it to the last section under the chapter so that
                    # orphan sub-items from continuation pages land correctly.
                    _is_new_section = sec_pat.match(node.get("title", ""))
                    _is_summary = re.match(
                        r"^本章小结|^练习|^总习题|^思考题",
                        node.get("title", ""),
                    )
                    if parent is None and children and not _is_new_section and not _is_summary:
                        last_child = children[-1]
                        if sec_pat.match(last_child.get("title", "")):
                            parent = last_child
                    target = parent if parent is not None else last
                    target["children"].append(node)
                    if target["children"]:
                        target["children"].sort(
                            key=lambda c: (
                                c.get("page_num") or 99999,
                                _section_sort_key(c["title"]),
                            )
                        )
                    continue
            merged.append(node)
    return merged


def _merge_content_box_trees(
    content_boxes: list[dict],
    cb_trees: list[list[dict]],
) -> list[dict]:
    """Merge per-content-box mini-trees within a single page.

    Content boxes are sorted left-to-right by column, then top-to-bottom
    within each column.  Trees are concatenated in that order.
    """

    def _get_coord(cb: dict) -> list[float]:
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
    toc_results: list[dict],
    page_heights: list[float] | None = None,
) -> list[dict]:
    """Semantic-pattern-based TOC reconstruction."""
    parsed, _ = _parse_toc_lines(toc_results, page_heights)
    if not parsed:
        return []

    # ---- Step 5: semantic patterns + xmin proximity fallback ----
    ch_pat = re.compile(r"^第[一二三四五六七八九十\d]+(?:章(?!习题)|篇)")
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
    toc_results: list[dict],
    page_heights: list[float] | None = None,
) -> list[dict]:
    """Pure indentation-based TOC reconstruction — per-CB clustering + merge."""
    if not toc_results:
        return []

    def _indent_levels(plist: list[dict]) -> None:
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
    toc_results: list[dict],
    page_heights: list[float] | None = None,
) -> list[dict]:
    """Per-page level detection + cross-page merge.

    Level detection (semantic + gap clustering) runs inside each page to avoid
    x-coordinate shifts between pages from corrupting the indentation clusters.
    Page-level mini-trees are then merged into the final global tree.
    """
    if not toc_results:
        return []

    # ---- patterns for semantic corrections (non-paragraph_title entries) ----
    ch_pat = re.compile(r"^第[一二三四五六七八九十\d]+(?:章(?!习题)|篇)")
    back_pat = re.compile(r"^附录|^参考书目|^参考文献|^名词索引|^索引|^学时分配")

    # Semantic floor patterns: gap-tree decides the primary level, but these
    # patterns set a *minimum* level so that e.g. a section header that is
    # left-aligned with the chapter is not flattened into level 0.
    _sec_floor_pat = re.compile(
        r"^第[一二三四五六七八九十\d]+节"       # 第一节
        r"|^\*?\d+\.(?!\d)"                     # 1. / *1. (not 1.1)
        r"|^\d+\s*\S"                           # "1 Title" (numbered section)
        r"|^习题\s*\d+"                          # 习题1-5
        r"|^本章小结|^练习|^总习题|^思考题"
        r"|^小结$|^习题$"                        # standalone
        r"|^第[一二三四五六七八九十\d]+章习题"   # 第一章习题
    )
    _sub_floor_pat = re.compile(
        r"^[一二三四五六七八九十]、"             # 一、二、三、
        r"|^\d+\.\d+"                            # 1.1  1.2
    )
    # 字母编号节（附录子节）："A.2 The Scalar Product"
    _alpha_sec_pat = re.compile(r"^[A-Z]\.\d")

    def _assign_levels(plist: list[dict]) -> None:
        """Assign indentation levels via gap-tree + paragraph_title depth.

        Paragraph_title entries above a content box form a nesting chain:
        the first is level 0, the second level 1, etc.  Content entries
        are shifted by the number of consecutive paragraph_titles above them.
        Gap-tree clustering on x-coordinates then adds further indentation
        within the content group.  Semantic floors only correct non-pt entries
        that would otherwise collapse to the wrong level.
        """
        n = len(plist)
        if n < 2:
            for p in plist:
                p["level"] = 0
            return

        all_xmins = np.array([p["min_xmin"] for p in plist], dtype=np.float64)
        # Deduplicate x-coordinates so repeated positions don't inflate the
        # gap statistics with zeros.  The structural gaps between distinct
        # indentation levels are what matter.
        unique_x = np.unique(all_xmins)
        diffs = np.diff(unique_x)

        # ---- gap-tree: baseline levels from x-coordinates ----
        if len(diffs) == 0:
            for p in plist:
                p["level"] = 0
        else:
            p80 = float(np.percentile(diffs, 80))
            min_gap = max(p80, 8.0)

            boundaries = [float(unique_x[0])]
            for i, d in enumerate(diffs):
                if d >= min_gap:
                    boundaries.append(float(unique_x[i + 1]))
            boundaries.sort()

            for p in plist:
                x = p["min_xmin"]
                dists = [abs(x - b) for b in boundaries]
                p["level"] = int(np.argmin(dists))

        # ---- paragraph_title depth (geometry-driven, language-agnostic) ----
        pt_depth = 0
        prev_was_pt = False
        for p in plist:  # already sorted by avg_ymin in caller
            is_pt = p.get("cb_label") == "paragraph_title"
            if is_pt:
                if not prev_was_pt:
                    pt_depth = 0
                p["level"] = pt_depth
                pt_depth += 1
            else:
                p["level"] = pt_depth + p["level"]
            prev_was_pt = is_pt

        # ---- semantic corrections (non-pt entries only) ----
        for p in plist:
            if p.get("cb_label") == "paragraph_title":
                continue
            t = p["title"]
            if ch_pat.match(t):
                p["level"] = 0
            elif back_pat.match(t):
                p["level"] = 0
            # 编号型英文章节（"1 Introduction"、"1. Introduction"、
            # "Chapter 1"、"A Table of Integrals"）：level 0 floor —— 否则
            # "1. Introduction" 会被 "^\d+\s*\S" section floor 抬到 level 1，
            # 嵌套进 List of Symbols 之类的 front matter 条目
            elif _EN_CHAPTER_FLOOR.match(t):
                p["level"] = 0
            # Section-like: at least level 1
            elif _sec_floor_pat.match(t) and p["level"] < 1:
                p["level"] = 1
            elif _alpha_sec_pat.match(t) and p["level"] < 1:
                # 字母编号节："A.2 The Scalar Product"（附录子节）
                p["level"] = 1
            # Subsection-like: at least level 2
            elif _sub_floor_pat.match(t) and p["level"] < 2:
                p["level"] = 2

        # If every entry landed in the same bin but x-std is large,
        # force a 2-cluster split (KMeans).
        levels = {p["level"] for p in plist}
        if len(levels) == 1 and n >= 3:
            x_arr = all_xmins.reshape(-1, 1)
            if float(np.std(x_arr)) > 12:
                km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x_arr)
                right_label = int(np.argmax(km.cluster_centers_))
                for p, lb in zip(plist, km.labels_):
                    t = p["title"]
                    if p.get("cb_label") == "paragraph_title":
                        continue
                    # never override a chapter / back-matter assignment
                    if ch_pat.match(t) or back_pat.match(t) or _EN_CHAPTER_FLOOR.match(t):
                        continue
                    new_level = 1 if int(lb) == right_label else 0
                    # respect semantic floors
                    if _sub_floor_pat.match(t):
                        new_level = max(new_level, 2)
                    elif _sec_floor_pat.match(t):
                        new_level = max(new_level, 1)
                    p["level"] = new_level
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


def repair_toc_tree(tree: list[dict]) -> list[dict]:
    """Post-process: fix misplaced entries based on semantic rules.

    Three passes:
      0. Root-level: re-parent ``第X章`` entries under their preceding
         ``第X篇`` when both exist at the root.
      1. Root-level: re-parent orphan section-like entries under their last chapter.
      2. Chapter children: re-parent exercises under their matching
         parent section (e.g. ``习题1-5`` → ``第五节``).
    """
    sec_like = re.compile(r"^\*?\d+\.\d+")
    pian_like = re.compile(r"^第[一二三四五六七八九十\d]+篇")
    zhang_like = re.compile(r"^第[一二三四五六七八九十\d]+章(?!习题)")
    ch_like = re.compile(r"^第[一二三四五六七八九十\d]+[章篇]")
    _cn_num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
               "八": 8, "九": 9, "十": 10}
    _sec_num_pat = re.compile(r"^第([一二三四五六七八九十\d]+)节")
    _ex_num_pat = re.compile(r"^习题\s*(\d+)[-−](\d+)")

    def _is_chapter(node: dict) -> bool:
        return bool(ch_like.match(node["title"]))

    def _fix_pian_structure(nodes: list[dict]) -> list[dict]:
        """Nest ``第X章`` entries under their preceding ``第X篇``.

        When a tree has both 篇 and 章 at the root level, all chapters
        between two 篇 are collected as children of the first 篇.
        """
        result: list[dict] = []
        current_pian = None
        for node in nodes:
            if pian_like.match(node["title"]):
                current_pian = node
                result.append(node)
            elif zhang_like.match(node["title"]) and current_pian is not None:
                current_pian.setdefault("children", []).append(node)
            else:
                result.append(node)
        return result

    _zhang_num_pat = re.compile(r"^第([一二三四五六七八九十\d]+)章")
    _sec_prefix_pat = re.compile(r"^(\d+)\.")

    def _fix_zhang_sections(nodes: list[dict]) -> list[dict]:
        """Re-parent orphan numbered sections under their matching chapter.

        e.g. ``7.1``, ``7.2``, ``7.3`` → ``第7章`` when both are at the
        same level.
        """
        chapters: dict[int, dict] = {}
        for node in nodes:
            m = _zhang_num_pat.match(node["title"])
            if m:
                cn = m.group(1)
                n = int(cn) if cn.isdigit() else _cn_num.get(cn, 0)
                if n > 0:
                    chapters[n] = node

        if not chapters:
            return nodes

        result: list[dict] = []
        for node in nodes:
            zm = _zhang_num_pat.match(node["title"])
            if zm:
                result.append(node)
                continue
            sm = _sec_prefix_pat.match(node["title"])
            if sm:
                ch_num = int(sm.group(1))
                if ch_num in chapters:
                    chapters[ch_num].setdefault("children", []).append(node)
                    continue
            result.append(node)

        for ch in chapters.values():
            if ch.get("children"):
                ch["children"].sort(
                    key=lambda c: (
                        c.get("page_num") or 99999,
                        _section_sort_key(c["title"]),
                    )
                )
        return result

    def _sec_ordinal(title: str) -> int | None:
        """Return the ordinal number of a section title, e.g. ``第五节`` → 5."""
        m = _sec_num_pat.match(title)
        if m:
            cn = m.group(1)
            return int(cn) if cn.isdigit() else _cn_num.get(cn)
        return None

    def _fix_chapter_children(children: list[dict]) -> list[dict]:
        """Re-parent exercises under their matching sections.

        ``总习题`` / ``本章小结`` / ``思考题`` stay at chapter level
        (same level as sections).
        """
        sections: list[dict] = []
        exercises: list[dict] = []
        keep: list[dict] = []

        for child in children:
            if _sec_num_pat.match(child["title"]):
                sections.append(child)
            elif _ex_num_pat.match(child["title"]):
                exercises.append(child)
            else:
                keep.append(child)

        for ex in exercises:
            ex_m = _ex_num_pat.match(ex["title"])
            if ex_m:
                sec_num = int(ex_m.group(2))
                parent = None
                for sec in sections:
                    if _sec_ordinal(sec["title"]) == sec_num:
                        parent = sec
                        break
                if parent is not None:
                    parent.setdefault("children", []).append(ex)
                    continue
            keep.append(ex)

        for sec in sections:
            if sec.get("children"):
                sec["children"].sort(
                    key=lambda c: (
                        c.get("page_num") or 99999,
                        _section_sort_key(c["title"]),
                    )
                )

        result = sorted(
            sections,
            key=lambda c: (c.get("page_num") or 99999, _section_sort_key(c["title"])),
        )
        result.extend(
            sorted(
                keep,
                key=lambda c: (c.get("page_num") or 99999, _section_sort_key(c["title"])),
            )
        )
        return result

    # ---- Pass 0: nest 章 under 篇 ----
    tree = _fix_pian_structure(tree)

    # ---- Pass 1: nest sections under their matching 章 ----
    tree = _fix_zhang_sections(tree)

    # ---- Pass 2: root-level fixup ----
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

    # ---- Pass 3: re-parent exercises under their sections ----
    for node in fixed_root:
        if node.get("children"):
            if _is_chapter(node) or ch_like.match(node["title"]):
                node["children"] = _fix_chapter_children(node["children"])
            node["children"].sort(
                key=lambda c: (
                    c.get("page_num") or 99999,
                    _section_sort_key(c["title"]),
                )
            )

    return fixed_root


def inherit_page_numbers(
    tree: list[dict], max_depth: int = 3
) -> list[dict]:
    """Fill missing page numbers from descendants or next sibling.

    Nodes with ``page_num is None`` (common for entries from
    ``paragraph_title`` boxes that don't include the page-number area)
    inherit the first non-None page number found in:
      1. children / grandchildren (up to *max_depth*)
      2. the next sibling at the same level
    """

    def _first_child_page(node: dict, depth: int) -> int | None:
        if depth <= 0:
            return None
        for child in node.get("children", []):
            if isinstance(child.get("page_num"), int):
                return child["page_num"]
            result = _first_child_page(child, depth - 1)
            if result is not None:
                return result
        return None

    def _fix_node(node: dict, siblings: list[dict], idx: int) -> None:
        # 只对缺失（None）页码继承：字符串页码（"VII"、"I-1"）是有意义的
        # 页码（front matter 罗马数字 / 罗马-数字映射），继承 int 会覆盖掉
        pn = node.get("page_num")
        if pn is None:
            inherited = _first_child_page(node, max_depth)
            if inherited is None and siblings:
                for sib in siblings[idx + 1 :]:
                    if isinstance(sib.get("page_num"), int):
                        inherited = sib["page_num"]
                        break
            if inherited is not None:
                node["page_num"] = inherited
        for i, child in enumerate(node.get("children", [])):
            _fix_node(child, node["children"], i)

    for i, node in enumerate(tree):
        _fix_node(node, tree, i)
    return tree
