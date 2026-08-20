import unittest

from toc_forge.parsing import (
    _page_num_sort_key,
    inherit_page_numbers,
    reconstruct_toc1,
    repair_toc_tree,
    restore_toc_order,
)


class TocOrderingTests(unittest.TestCase):
    def test_missing_earlier_sections_keep_source_order_and_inherit_from_later_section(self):
        tree = [
            {
                "title": "第八章 量子力学若干进展",
                "page_num": None,
                "children": [
                    {"title": "8.1 量子纠缠", "page_num": None, "children": []},
                    {"title": "8.2 量子计算", "page_num": None, "children": []},
                    {"title": "8.3 贝利相", "page_num": 218, "children": []},
                ],
            }
        ]

        repaired = repair_toc_tree(tree)
        inherited = inherit_page_numbers(repaired)
        ordered = restore_toc_order(inherited)

        chapter = ordered[0]
        self.assertEqual(
            [child["title"] for child in chapter["children"]],
            ["8.1 量子纠缠", "8.2 量子计算", "8.3 贝利相"],
        )
        self.assertEqual(
            [child["page_num"] for child in chapter["children"]],
            [218, 218, 218],
        )
        self.assertEqual(chapter["page_num"], 218)

    def test_resolved_siblings_are_restored_to_printed_page_order(self):
        tree = [
            {
                "title": "第八章 量子力学若干进展",
                "page_num": 201,
                "children": [
                    {"title": "8.1 量子纠缠", "page_num": 202, "children": []},
                    {"title": "8.3 贝利相", "page_num": 218, "children": []},
                    {"title": "8.2 量子计算", "page_num": 210, "children": []},
                ],
            }
        ]

        ordered = restore_toc_order(tree)

        self.assertEqual(
            [child["title"] for child in ordered[0]["children"]],
            ["8.1 量子纠缠", "8.2 量子计算", "8.3 贝利相"],
        )

    def test_unresolved_sibling_prevents_partial_reordering(self):
        tree = [
            {
                "title": "Chapter 8",
                "page_num": 200,
                "children": [
                    {"title": "8.3 Berry Phase", "page_num": 218, "children": []},
                    {"title": "8.2 Quantum Computing", "page_num": None, "children": []},
                    {"title": "8.1 Entanglement", "page_num": 202, "children": []},
                ],
            }
        ]

        ordered = restore_toc_order(tree)

        self.assertEqual(
            [child["title"] for child in ordered[0]["children"]],
            ["8.3 Berry Phase", "8.2 Quantum Computing", "8.1 Entanglement"],
        )

    def test_equal_page_siblings_keep_source_order(self):
        tree = [
            {
                "title": "Chapter 1",
                "page_num": 1,
                "children": [
                    {"title": "Introduction", "page_num": 7, "children": []},
                    {"title": "1.1 Review", "page_num": 7, "children": []},
                ],
            }
        ]

        ordered = restore_toc_order(tree)

        self.assertEqual(
            [child["title"] for child in ordered[0]["children"]],
            ["Introduction", "1.1 Review"],
        )

    def test_page_sort_key_supports_string_page_number_formats(self):
        self.assertEqual(_page_num_sort_key(12), 12)
        self.assertEqual(_page_num_sort_key("VII"), 7)
        self.assertEqual(_page_num_sort_key("II-12"), 12)
        self.assertIsNone(_page_num_sort_key(None))

    def test_chapter_local_string_pages_are_sorted_within_one_chapter(self):
        tree = [
            {
                "title": "Chapter II",
                "page_num": "II-1",
                "children": [
                    {"title": "II.3", "page_num": "II-12", "children": []},
                    {"title": "II.2", "page_num": "II-7", "children": []},
                ],
            }
        ]

        ordered = restore_toc_order(tree)

        self.assertEqual(
            [child["page_num"] for child in ordered[0]["children"]],
            ["II-7", "II-12"],
        )

    def test_mixed_page_numbering_systems_keep_source_order(self):
        tree = [
            {
                "title": "Part I",
                "page_num": "vii",
                "children": [
                    {"title": "Introduction", "page_num": 1, "children": []},
                    {"title": "Preface", "page_num": "vii", "children": []},
                ],
            }
        ]

        ordered = restore_toc_order(tree)

        self.assertEqual(
            [child["title"] for child in ordered[0]["children"]],
            ["Introduction", "Preface"],
        )


class FrontBackMatterTests(unittest.TestCase):
    def test_appendix_with_page_number_is_preserved_as_a_toc_entry(self):
        toc_results = [
            {
                "page": 0,
                "content_boxes": [
                    {
                        "content_box": {
                            "coordinate": [0, 0, 120, 100],
                            "label": "content",
                        },
                        "rec_texts": ["附录", "234"],
                        "rec_boxes": [[5, 10, 35, 25], [90, 10, 115, 25]],
                    }
                ],
            }
        ]

        tree = reconstruct_toc1(toc_results, page_heights=[100])

        self.assertEqual(len(tree), 1)
        self.assertEqual(tree[0]["title"], "附录")
        self.assertEqual(tree[0]["page_num"], 234)


if __name__ == "__main__":
    unittest.main()
