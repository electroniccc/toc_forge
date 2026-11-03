import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import pymupdf

from toc_forge.ocr_engine import detect_toc_pages_by_keyword
from toc_forge.llm import _save_toc_tree_cache, build_toc_vllm


class TextLayerTocTests(unittest.TestCase):
    def test_contents_page_without_layout_title_is_detected(self):
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((200, 75), 'Contents')
            page.insert_text((100, 130), 'Chapter 1\nLinear Equations\n1')
            page.insert_text((100, 175), 'Chapter 2\nMatrix Algebra\n71')
            image = np.zeros((1200, 800, 3), dtype=np.uint8)
            result = detect_toc_pages_by_keyword(
                [[]], [image], None, document=document
            )
            self.assertEqual([item['page'] for item in result], [0])
            self.assertEqual(result[0]['content_boxes'][0]['label'], 'content')

    def test_prose_mention_of_contents_is_not_a_toc(self):
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((100, 75), 'Contents of this chapter')
            page.insert_text((100, 130), 'Chapter 1 introduces the contents of a matrix.')
            image = np.zeros((1200, 800, 3), dtype=np.uint8)
            self.assertEqual(
                detect_toc_pages_by_keyword([[]], [image], None, document=document),
                [],
            )

    def test_newly_detected_first_page_invalidates_old_toc_tree(self):
        with pymupdf.open() as document, tempfile.TemporaryDirectory() as cache_dir:
            page = document.new_page()
            page.insert_text((200, 75), 'Contents')
            page.insert_text((100, 130), 'Chapter 1\nLinear Equations\n1')
            page.insert_text((100, 175), 'Chapter 2\nMatrix Algebra\n71')
            images = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(3)]
            first = detect_toc_pages_by_keyword([[]], images, None, document=document)
            pages = first + [{'page': 1}, {'page': 2}]
            _save_toc_tree_cache(cache_dir, 'book', 'vllm',
                                 [{'title': 'Chapter 4', 'page_num': 191}], [1, 2])
            with patch('toc_forge.llm._build_llm_client', return_value=object()), \
                 patch('toc_forge.llm._call_llm', return_value=(
                     {'toc': [{'title': 'Chapter 1', 'page_num': 1}]}, None
                 )) as call:
                result = build_toc_vllm(pages, images, 'vision-model',
                                        'http://example.invalid/v1',
                                        cache_dir=cache_dir, pdf_hash='book')
            self.assertEqual(result.toc_tree[0]['title'], 'Chapter 1')
            call.assert_called_once()
