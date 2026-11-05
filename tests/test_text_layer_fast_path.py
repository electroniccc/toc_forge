import unittest
from unittest.mock import Mock

import numpy as np
import pymupdf

from toc_forge.ocr_engine import detect_toc_pages_by_keyword
from toc_forge.page_map import _ocr_front_matter_page_numbers


class TextLayerFastPathTests(unittest.TestCase):
    def test_readable_toc_header_avoids_ocr(self):
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=900)
            page.insert_text((30, 45), "CONTENTS")
            image = np.zeros((1200, 800, 3), dtype=np.uint8)
            boxes = [[{"label": "header", "coordinate": [30, 40, 190, 75]}]]
            ocr = Mock()
            result = detect_toc_pages_by_keyword(
                boxes, [image], ocr, document=document,
            )
            self.assertEqual([item["page"] for item in result], [0])
            ocr.predict.assert_not_called()

    def test_readable_non_toc_header_avoids_ocr(self):
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=900)
            page.insert_text((30, 45), "PREFACE")
            image = np.zeros((1200, 800, 3), dtype=np.uint8)
            boxes = [[{"label": "header", "coordinate": [30, 40, 190, 75]}]]
            ocr = Mock()
            self.assertEqual(
                detect_toc_pages_by_keyword(boxes, [image], ocr, document=document),
                [],
            )
            ocr.predict.assert_not_called()

    def test_scanned_header_still_uses_ocr(self):
        with pymupdf.open() as document:
            document.new_page(width=600, height=900)
            image = np.zeros((1200, 800, 3), dtype=np.uint8)
            boxes = [[{"label": "header", "coordinate": [30, 40, 190, 75]}]]
            ocr = Mock()
            ocr.predict.return_value = [{"rec_texts": ["CONTENTS"]}]
            result = detect_toc_pages_by_keyword(boxes, [image], ocr, document=document)
            self.assertEqual([item["page"] for item in result], [0])
            ocr.predict.assert_called_once()

    def test_corrupted_text_layer_header_falls_back_to_ocr(self):
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=900)
            page.insert_text((30, 45), "C0NTENTS")
            image = np.zeros((1200, 800, 3), dtype=np.uint8)
            boxes = [[{"label": "header", "coordinate": [30, 40, 190, 75]}]]
            ocr = Mock()
            ocr.predict.return_value = [{"rec_texts": ["CONTENTS"]}]
            result = detect_toc_pages_by_keyword(boxes, [image], ocr, document=document)
            self.assertEqual([item["page"] for item in result], [0])
            ocr.predict.assert_called_once()

    def test_roman_front_matter_text_layer_avoids_ocr(self):
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=900)
            page.insert_text((25, 30), "xii")
            ocr = Mock()
            labels = _ocr_front_matter_page_numbers(document, 0, ocr, None, None)
            self.assertEqual(labels, ["xii"])
            ocr.predict.assert_not_called()

    def test_roman_footer_in_outer_corner_avoids_ocr(self):
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=900)
            page.insert_text((550, 875), "xxiii")
            ocr = Mock()
            labels = _ocr_front_matter_page_numbers(document, 0, ocr, None, None)
            self.assertEqual(labels, ["xxiii"])
            ocr.predict.assert_not_called()

    def test_roman_looking_footer_word_is_not_a_page_label(self):
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=900)
            page.insert_text((25, 875), "civil")
            ocr = Mock()
            ocr.predict.return_value = [{"rec_texts": []}]
            _ocr_front_matter_page_numbers(document, 0, ocr, None, None)
            self.assertEqual(ocr.predict.call_count, 3)

    def test_blank_front_matter_page_still_uses_ocr(self):
        with pymupdf.open() as document:
            document.new_page(width=600, height=900)
            ocr = Mock()
            ocr.predict.return_value = [{"rec_texts": []}]
            self.assertEqual(
                _ocr_front_matter_page_numbers(document, 0, ocr, None, None),
                ["", "", ""],
            )
            self.assertEqual(ocr.predict.call_count, 3)


if __name__ == "__main__":
    unittest.main()
