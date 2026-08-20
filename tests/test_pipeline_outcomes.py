import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from toc_forge import EmptyTocError, TocNotFoundError
from toc_forge.pipeline import add_bookmarks_to_pdf


class BookmarkWritingTests(unittest.TestCase):
    def _document(self, page_count: int = 4) -> pymupdf.Document:
        doc = pymupdf.open()
        for _ in range(page_count):
            doc.new_page()
        return doc

    def test_unresolved_leaf_is_skipped_instead_of_pointing_to_page_one(self):
        tree = [
            {"title": "无法解析的条目", "page_num": None, "children": []},
            {"title": "第一章", "page_num": 1, "children": []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "bookmarked.pdf")
            doc = self._document()
            skipped = add_bookmarks_to_pdf(doc, tree, 0, output)
            doc.close()

            written = pymupdf.open(output)
            try:
                self.assertEqual(written.get_toc(), [[1, "第一章", 2]])
            finally:
                written.close()

        self.assertEqual(skipped, ["无法解析的条目"])

    def test_empty_tree_is_an_explicit_failure_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "bookmarked.pdf")
            doc = self._document()
            try:
                with self.assertRaises(EmptyTocError):
                    add_bookmarks_to_pdf(doc, [], 0, output)
            finally:
                doc.close()
            self.assertFalse(os.path.exists(output))

    def test_resolvable_child_is_promoted_when_its_parent_has_no_page(self):
        tree = [
            {
                "title": "无法定位的分组",
                "page_num": "not-a-page",
                "children": [
                    {"title": "第一节", "page_num": 2, "children": []}
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "bookmarked.pdf")
            doc = self._document()
            skipped = add_bookmarks_to_pdf(doc, tree, 0, output)
            doc.close()

            written = pymupdf.open(output)
            try:
                self.assertEqual(written.get_toc(), [[1, "第一节", 3]])
            finally:
                written.close()

        self.assertEqual(skipped, ["无法定位的分组"])


class CliOutcomeTests(unittest.TestCase):
    def test_toc_not_found_exits_nonzero_without_success_message(self):
        from toc_forge import cli

        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["toc-forge", "--input", "missing.pdf"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(cli, "bookmark_pdf", side_effect=TocNotFoundError("未检测到目录页")),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main()

        self.assertEqual(raised.exception.code, 2)
        self.assertNotIn("Bookmarked PDF saved", stdout.getvalue())
        self.assertIn("未检测到目录页", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
