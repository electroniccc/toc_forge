import hashlib
import inspect
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from toc_forge import pipeline
from toc_forge.llm import (
    _TOC_LLM_SYSTEM_PROMPT,
    _TOC_VLLM_SYSTEM_PROMPT,
    _build_llm_client,
    _call_llm,
    _load_toc_tree_cache,
    _save_toc_tree_cache,
    _simplify_ocr_for_llm,
    _toc_tree_cache_path,
    build_toc_llm,
    build_toc_vllm,
)
from toc_forge.utils import _cache_load, _cache_path, _cache_save, compute_file_hash


class LlmInputTests(unittest.TestCase):
    def test_call_returns_input_and_output_token_usage(self):
        class Completions:
            def create(self, **_kwargs):
                class Response:
                    usage = type("Usage", (), {
                        "prompt_tokens": 123,
                        "completion_tokens": 17,
                    })()
                    choices = [
                        type("Choice", (), {
                            "message": type("Message", (), {"content": '{"toc": []}'})()
                        })()
                    ]

                return Response()

        client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})()
        })()
        parsed, usage = _call_llm(client, "model", "system", "input")

        self.assertEqual(parsed, {"toc": []})
        self.assertEqual(usage.input_tokens, 123)
        self.assertEqual(usage.output_tokens, 17)

    def test_prompts_split_compact_multientry_rows_in_any_language(self):
        for prompt in (_TOC_LLM_SYSTEM_PROMPT, _TOC_VLLM_SYSTEM_PROMPT):
            self.assertIn("separate sibling nodes", prompt)
            self.assertIn("一、映射(1) 二、函数(3) 习题 1–1(16)", prompt)
            self.assertIn(
                "Limits (105) Derivatives (125) Exercises 2.1 (140)",
                prompt,
            )
            self.assertIn("full-width punctuation", prompt)
            self.assertIn("metadata, never part of the title", prompt)
            self.assertIn(
                '{"title":"一、映射","page_num":1,"children":[]}',
                prompt,
            )
            self.assertIn(
                '{"title":"Limits","page_num":105,"children":[]}',
                prompt,
            )
            self.assertIn("title must not end in", prompt)

    def test_model_name_and_base_url_are_required_function_arguments(self):
        for function in (build_toc_llm, build_toc_vllm):
            parameters = inspect.signature(function).parameters
            self.assertIs(parameters["llm_model"].default, inspect.Parameter.empty)
            self.assertIs(parameters["llm_base_url"].default, inspect.Parameter.empty)

        client_parameters = inspect.signature(_build_llm_client).parameters
        self.assertIs(
            client_parameters["base_url"].default,
            inspect.Parameter.empty,
        )

        with self.assertRaisesRegex(ValueError, "llm_model is required"):
            build_toc_vllm([], [], "", "http://localhost/v1")
        with self.assertRaisesRegex(ValueError, "llm_base_url is required"):
            build_toc_vllm([], [], "vision-model", "")

    def test_fragments_on_the_same_line_are_ordered_left_to_right(self):
        toc_results = [
            {
                "page": 3,
                "content_boxes": [
                    {
                        "rec_texts": ["Chapter 1", "Introduction", "7"],
                        # Deliberately reverse the y minima.  The boxes still
                        # overlap vertically and form one visual line.
                        "rec_boxes": [
                            [5, 12, 45, 28],
                            [50, 11, 100, 27],
                            [110, 10, 118, 26],
                        ],
                    }
                ],
            }
        ]

        simplified = _simplify_ocr_for_llm(toc_results)

        self.assertEqual(simplified, [{"page": 3, "lines": ["Chapter 1 Introduction 7"]}])


class CacheIdentityTests(unittest.TestCase):
    def test_pdf_cache_identity_uses_the_complete_sha256_digest(self):
        content = b"a PDF-shaped cache identity fixture"
        with tempfile.TemporaryDirectory() as cache_dir:
            path = os.path.join(cache_dir, "fixture.pdf")
            with open(path, "wb") as f:
                f.write(content)

            digest = compute_file_hash(path)

        self.assertEqual(digest, hashlib.sha256(content).hexdigest())
        self.assertEqual(len(digest), 64)

    def test_same_pdf_threads_share_one_locked_cache_fill(self):
        worker_count = 6
        start_barrier = threading.Barrier(worker_count)
        counter_lock = threading.Lock()
        producer_calls = 0

        with tempfile.TemporaryDirectory() as root:
            input_path = os.path.join(root, "fixture.pdf")
            cache_dir = os.path.join(root, "cache")
            with open(input_path, "wb") as f:
                f.write(b"same PDF bytes for every thread")
            pdf_hash = compute_file_hash(input_path)
            cache_path = _cache_path(cache_dir, pdf_hash, "ocr", 0)

            def fake_pipeline(**_kwargs):
                nonlocal producer_calls
                cached = _cache_load(cache_path)
                if cached is None:
                    with counter_lock:
                        producer_calls += 1
                    # Keep the miss window open long enough that this test
                    # reproduces the old cache-stampede behavior without the
                    # bookmark_pdf-level lock.
                    time.sleep(0.05)
                    cached = {"producer": producer_calls, "complete": True}
                    _cache_save(cache_path, cached)
                return cached

            def invoke_bookmark_pdf():
                start_barrier.wait()
                return pipeline.bookmark_pdf(
                    input=input_path,
                    output=os.path.join(root, "output"),
                    model_dir=os.path.join(root, "models"),
                    cache_dir=cache_dir,
                )

            with patch.object(
                pipeline, "_bookmark_pdf_impl", side_effect=fake_pipeline
            ):
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    results = list(
                        executor.map(
                            lambda _index: invoke_bookmark_pdf(),
                            range(worker_count),
                        )
                    )

            self.assertEqual(producer_calls, 1)
            self.assertTrue(all(result == results[0] for result in results))
            self.assertEqual(
                _cache_load(cache_path), {"producer": 1, "complete": True}
            )
            self.assertTrue(
                os.path.isfile(os.path.join(cache_dir, pdf_hash, ".cache.lock"))
            )

    def test_failed_cache_write_preserves_the_previous_complete_value(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            path = f"{cache_dir}/entry.json"
            _cache_save(path, {"status": "complete"})
            circular = {}
            circular["self"] = circular

            with self.assertRaises(ValueError):
                _cache_save(path, circular)

            self.assertEqual(_cache_load(path), {"status": "complete"})

    def test_llm_cache_uses_one_fixed_file_per_strategy(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            path = _toc_tree_cache_path(
                cache_dir,
                "pdfhash",
                "llm",
            )

        self.assertEqual(
            path,
            os.path.join(cache_dir, "pdfhash", "toc_tree_llm.json"),
        )

    def test_llm_cache_loads_the_single_saved_tree(self):
        tree = [{"title": "Chapter 1", "page_num": 1, "children": []}]
        with tempfile.TemporaryDirectory() as cache_dir:
            _save_toc_tree_cache(
                cache_dir,
                "pdfhash",
                "llm",
                tree,
            )

            cached = _load_toc_tree_cache(cache_dir, "pdfhash", "llm")

        self.assertEqual(cached, tree)


if __name__ == "__main__":
    unittest.main()
