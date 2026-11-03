import os
import tempfile
import unittest

import pymupdf

from toc_forge.page_map import find_text_page_offset, map_toc_pages, resolve_pdf_page
from toc_forge.pipeline import add_bookmarks_to_pdf


class PageResolutionTests(unittest.TestCase):
    def test_long_roman_preface_does_not_force_zero_offset(self):
        with pymupdf.open() as doc:
            for _ in range(44): doc.new_page()
            for i in range(8,37): doc[i].insert_text((25,25),'xxvii')
            for i in range(37,44): doc[i].insert_text((25,25),str(i-36))
            self.assertEqual(find_text_page_offset(doc,9),36)

    def test_no_text_or_isolated_formula_digits_are_not_a_page_mapping(self):
        with pymupdf.open() as doc:
            for _ in range(7): doc.new_page()
            self.assertIsNone(find_text_page_offset(doc,0))
            for i in range(7): doc[i].insert_text((100,200),str(i+1))
            self.assertIsNone(find_text_page_offset(doc,0))

    def test_public_tree_and_pdf_outline_use_the_same_segment_mapping(self):
        tree=[{'title':'Chapter','page_num': '4','children':[
            {'title':'Section','page_num':5},
            {'title':'Preface','page_num':'ii'},
            {'title':'Unknown','page_num':'A99'},
        ]}]
        mapping={'arabic_segments':[{'printed_start':4,'printed_end':7,'pdf_start':9}], 'front_offset':0}
        with pymupdf.open() as doc, tempfile.TemporaryDirectory() as tmp:
            for _ in range(16): doc.new_page()
            out=os.path.join(tmp,'book.pdf')
            add_bookmarks_to_pdf(doc,tree,2,out,**mapping)
            map_toc_pages(tree,2,page_count=len(doc),**mapping)
            with pymupdf.open(out) as result:
                self.assertEqual([item[2] for item in result.get_toc()], [10,11,3])
            self.assertEqual(tree[0]['page_num'],10)
            self.assertEqual([n['page_num'] for n in tree[0]['children']],[11,3,None])

    def test_roman_chapter_pages_and_out_of_range_nodes(self):
        self.assertEqual(resolve_pdf_page('II-1',12,page_map={'II-1':40}),53)
        tree=[{'title':'outside','page_num':1000}]
        map_toc_pages(tree,0,page_count=15)
        self.assertIsNone(tree[0]['page_num'])


class AdaptiveSamplingTests(unittest.TestCase):
    def _run_pipeline(self, directory, printed_labels):
        from contextlib import ExitStack
        from unittest.mock import patch
        import numpy as np
        from toc_forge import pipeline
        from toc_forge.llm import LlmBuildResult

        source=os.path.join(directory,'source.pdf')
        with pymupdf.open() as doc:
            for _ in range(65): doc.new_page()
            doc.save(source)
        calls=[]
        def layout(doc, model, start, end, **kw):
            calls.append((start,end))
            return [np.zeros((100,100,3),dtype=np.uint8) for _ in range(start,end)], list(range(start,end))
        def boxes(results, start=0):
            return ([{'page':0}] if start==0 else []), [{'page':i} for i in results], []
        def ocr(kept,*args,**kw):
            return [{'page':r['page'],'rec_texts':[printed_labels(r['page'])]} for r in kept]
        with ExitStack() as stack:
            mocks={
                'make_sure_model_exists':None, 'LayoutDetection':object(), 'PaddleOCR':object(),
                'scan_toc_page_range':(10,[]),
                'detect_toc_pages_by_keyword':[], 'detect_toc_pages_by_continuity':[],
                'find_text_page_offset':None, 'detect_segmented_offset':False,
                'build_toc_vllm':LlmBuildResult([{'title':'Chapter','page_num':1}]),
            }
            for name,value in mocks.items(): stack.enter_context(patch.object(pipeline,name,return_value=value))
            stack.enter_context(patch.object(pipeline,'layout_pages_in_range',side_effect=layout))
            stack.enter_context(patch.object(pipeline,'extract_toc_and_number_pages',side_effect=boxes))
            stack.enter_context(patch.object(pipeline,'get_number_box_pages',side_effect=lambda boxes,imgs:boxes))
            stack.enter_context(patch.object(pipeline,'ocr_number_boxes',side_effect=ocr))
            result=pipeline.bookmark_pdf(source,os.path.join(directory,'out'),directory,toc_strategy='vllm')
        return result,calls

    def test_scanned_book_extends_past_roman_preface_and_maps_returned_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self._run_pipeline(tmp,lambda i: 'xx' if i<37 else str(i-36))
            self.assertGreater(len(calls),2)
            self.assertEqual(result.toc_tree[0]['page_num'],38)
            with pymupdf.open(result.pdf_bookmarks_path) as pdf:
                self.assertEqual(pdf.get_toc()[0][2],38)

    def test_unresolved_page_numbers_stop_instead_of_using_zero_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Cannot determine body page numbering'):
                self._run_pipeline(tmp,lambda i:'xx')
            self.assertFalse(os.path.exists(os.path.join(tmp,'out','source_bookmarked.pdf')))
