"""Tests for the batched TOC page scan and page-number sampling range."""

import inspect
import sys
import unittest
from unittest.mock import Mock, patch

import pymupdf

from toc_forge.ocr_engine import (
    DEFAULT_TOC_DETECT_MAX_PAGE,
    extract_toc_and_number_pages,
    scan_toc_page_range,
)


def _box(label, coordinate=(0, 0, 100, 100)):
    return {"label": label, "coordinate": list(coordinate), "score": 0.9}


class _FakeLayoutModel:
    """Canned layout results: a monotonic page counter (the scan predicts
    batches strictly in ascending page order) decides which boxes a page
    carries."""

    def __init__(self, content_pages=(), number_pages=()):
        self.content = set(content_pages)
        self.numbers = set(number_pages)
        self._next = 0
        self.predict_calls = 0

    def predict(self, imgs, layout_nms=True):
        del layout_nms
        self.predict_calls += 1
        results = []
        for _ in imgs:
            idx = self._next
            self._next += 1
            boxes = []
            if idx in self.content:
                boxes.append(_box("content"))
            if idx in self.numbers:
                boxes.append(_box("number"))
            results.append({"boxes": boxes})
        return results


def _document(page_count: int) -> pymupdf.Document:
    doc = pymupdf.open()
    for _ in range(page_count):
        doc.new_page()
    return doc


class ScanTocPageRangeTests(unittest.TestCase):
    def test_stops_after_3_consecutive_non_toc_pages_in_first_batch(self):
        doc = _document(40)
        model = _FakeLayoutModel(content_pages={2, 3})
        try:
            scan_end, toc_indices = scan_toc_page_range(doc, model, 75)
        finally:
            doc.close()

        self.assertEqual(scan_end, 7)  # p4, p5, p6 are the 3 non-TOC pages
        self.assertEqual(toc_indices, [2, 3])

    def test_toc_beyond_first_batch_scans_batches_of_25(self):
        doc = _document(40)
        model = _FakeLayoutModel(content_pages={30})
        try:
            scan_end, toc_indices = scan_toc_page_range(doc, model, 75)
        finally:
            doc.close()

        # batch 1 [0,25) empty, batch 2 [25,40) finds p30, then p31/p32/p33
        # non-TOC stop the run
        self.assertEqual(scan_end, 34)
        self.assertEqual(toc_indices, [30])

    def test_batch_size_shrinks_to_10_after_first_toc_page(self):
        doc = _document(40)
        model = _FakeLayoutModel(content_pages={25, 26, 27, 28})
        try:
            scan_end, toc_indices = scan_toc_page_range(doc, model, 75)
        finally:
            doc.close()

        # found at p25 (in the 25-page batch), then p29/p30/p31 stop the run
        self.assertEqual(scan_end, 32)
        self.assertEqual(toc_indices, [25, 26, 27, 28])
        # 2 predict calls: the 25-page batch, then the 10-page batch
        self.assertEqual(model.predict_calls, 2)

    def test_max_pages_cap_without_any_toc_page(self):
        doc = _document(30)
        model = _FakeLayoutModel(content_pages=())
        try:
            scan_end, toc_indices = scan_toc_page_range(doc, model, 20)
        finally:
            doc.close()

        self.assertEqual(scan_end, 20)
        self.assertEqual(toc_indices, [])

    def test_max_pages_capped_by_document_length(self):
        doc = _document(30)
        model = _FakeLayoutModel(content_pages=())
        try:
            scan_end, toc_indices = scan_toc_page_range(doc, model, 75)
        finally:
            doc.close()

        self.assertEqual(scan_end, 30)
        self.assertEqual(toc_indices, [])

    def test_default_max_pages_constant_is_three_batches_of_25(self):
        self.assertEqual(DEFAULT_TOC_DETECT_MAX_PAGE, 25 * 3)


class ExtractTocAndNumberPagesTests(unittest.TestCase):
    def test_start_offset_is_applied_to_page_indices(self):
        results = [
            {"boxes": [_box("content")]},   # page 10
            {"boxes": [_box("number")]},    # page 11
            {"boxes": [_box("text")]},      # page 12
        ]
        toc_pages, number_pages, all_boxes = extract_toc_and_number_pages(
            results, start=10
        )

        self.assertEqual([p["page"] for p in toc_pages], [10])
        self.assertEqual([p["page"] for p in number_pages], [11])
        self.assertEqual(len(all_boxes), 3)

    def test_paragraph_titles_are_merged_into_content_pages(self):
        results = [
            {
                "boxes": [
                    _box("paragraph_title", (0, 0, 100, 20)),
                    _box("content", (0, 25, 100, 120)),
                ]
            }
        ]
        toc_pages, number_pages, _ = extract_toc_and_number_pages(results)

        self.assertEqual(len(toc_pages), 1)
        labels = {b["label"] for b in toc_pages[0]["content_boxes"]}
        self.assertEqual(labels, {"content", "paragraph_title"})
        self.assertEqual(number_pages, [])


class ParameterExposureTests(unittest.TestCase):
    def test_bookmark_pdf_accepts_toc_detect_max_page(self):
        from toc_forge.pipeline import bookmark_pdf

        parameters = inspect.signature(bookmark_pdf).parameters
        self.assertIn("toc_detect_max_page", parameters)
        self.assertIsNone(parameters["toc_detect_max_page"].default)

    def test_cli_passes_toc_detect_max_page_through(self):
        from toc_forge import cli

        mock = Mock(return_value=("out.pdf", 1.0, {}))
        with (
            patch.object(cli, "bookmark_pdf", mock),
            patch.object(
                sys,
                "argv",
                ["toc-forge", "--input", "x.pdf", "--toc_detect_max_page", "60"],
            ),
        ):
            cli.main()

        self.assertEqual(mock.call_args.kwargs["toc_detect_max_page"], 60)

    def test_cli_defaults_toc_detect_max_page_to_none(self):
        from toc_forge import cli

        mock = Mock(return_value=("out.pdf", 1.0, {}))
        with (
            patch.object(cli, "bookmark_pdf", mock),
            patch.object(sys, "argv", ["toc-forge", "--input", "x.pdf"]),
        ):
            cli.main()

        self.assertIsNone(mock.call_args.kwargs["toc_detect_max_page"])


if __name__ == "__main__":
    unittest.main()
