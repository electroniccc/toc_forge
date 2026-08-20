import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from toc_forge import gui_support


class _IncompleteResponse:
    headers = {"content-length": "10"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=65536):
        del chunk_size
        yield b"short"


class DownloadTests(unittest.TestCase):
    def test_incomplete_download_does_not_replace_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "inference.onnx")
            Path(dst).write_bytes(b"known-good")

            with (
                patch.object(gui_support.requests, "get", return_value=_IncompleteResponse()),
                self.assertRaises(requests.ConnectionError),
            ):
                gui_support.stream_download("https://example.invalid/model", dst, None, retries=1)

            self.assertEqual(Path(dst).read_bytes(), b"known-good")
            self.assertFalse(os.path.exists(dst + ".part"))

    def test_model_requires_both_nontrivial_onnx_and_yaml_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in gui_support.MODEL_NAMES:
                target = Path(tmp, name)
                target.mkdir()
                with open(target / "inference.onnx", "wb") as f:
                    f.truncate(1_100_000)

            self.assertFalse(gui_support.all_models_exist(tmp))

            for name in gui_support.MODEL_NAMES:
                Path(tmp, name, "inference.yml").write_text(
                    "Global:\n  model_name: regression-test\n", encoding="utf-8"
                )

            self.assertTrue(gui_support.all_models_exist(tmp))


class GuiPathTests(unittest.TestCase):
    def test_duplicate_input_stems_receive_distinct_deterministic_output_names(self):
        paths = ["/library/a/book.pdf", "/library/b/book.pdf", "/library/c/other.pdf"]

        first = gui_support.plan_output_names(paths)
        second = gui_support.plan_output_names(paths)

        self.assertEqual(first, second)
        self.assertNotEqual(first[0], first[1])
        self.assertEqual(first[2], "other_bookmarked.pdf")
        self.assertTrue(all(name.endswith("_bookmarked.pdf") for name in first))

    def test_linux_settings_path_uses_the_user_configuration_directory(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"XDG_CONFIG_HOME": tmp}
        ):
            path = gui_support.default_settings_path()

        self.assertEqual(path, os.path.join(tmp, "toc-forge", "settings.json"))


class PackagingTests(unittest.TestCase):
    def test_gui_optional_dependencies_are_declared(self):
        with open("pyproject.toml", "rb") as f:
            project = tomllib.load(f)["project"]
        gui_dependencies = " ".join(project["optional-dependencies"]["gui"]).lower()

        self.assertIn("requests", gui_dependencies)
        self.assertIn("sv-ttk", gui_dependencies)
        self.assertIn("onnxruntime", gui_dependencies)


if __name__ == "__main__":
    unittest.main()
