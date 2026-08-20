"""Reliable, Tk-independent support functions for the desktop GUI."""

import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import requests

MODEL_NAMES = (
    "PP-DocLayout_plus-L",
    "PP-LCNet_x1_0_doc_ori",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_mobile_rec",
)
MODEL_FILES = ("inference.yml", "inference.onnx")

_MIN_ONNX_BYTES = 1_000_000
_MIN_YAML_BYTES = 16


def model_is_complete(model_dir: str, model_name: str) -> bool:
    """Return whether an ONNX model has both non-trivial required files."""
    target = Path(model_dir, model_name)
    onnx_path = target / "inference.onnx"
    yaml_path = target / "inference.yml"
    try:
        return (
            onnx_path.is_file()
            and onnx_path.stat().st_size >= _MIN_ONNX_BYTES
            and yaml_path.is_file()
            and yaml_path.stat().st_size >= _MIN_YAML_BYTES
        )
    except OSError:
        return False


def all_models_exist(model_dir: str) -> bool:
    return all(model_is_complete(model_dir, name) for name in MODEL_NAMES)


def stream_download(
    url: str,
    dst: str,
    progress_cb: Callable[[float], None] | None,
    retries: int = 3,
) -> None:
    """Download to ``dst.part`` and atomically replace ``dst`` on success."""
    destination = os.path.abspath(dst)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    part_path = destination + ".part"
    last_err: Exception | None = None

    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb and total:
                            progress_cb(downloaded / total)
                    f.flush()
                    os.fsync(f.fileno())
                if downloaded == 0 or (total and downloaded != total):
                    raise requests.ConnectionError(
                        f"incomplete download: {downloaded} of {total} bytes"
                    )
            os.replace(part_path, destination)
            return
        except (requests.RequestException, OSError) as exc:
            try:
                os.unlink(part_path)
            except FileNotFoundError:
                pass
            if (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and 400 <= exc.response.status_code < 500
            ):
                raise
            last_err = exc
            if attempt < retries - 1:
                time.sleep(1.0 + attempt)

    raise requests.ConnectionError(
        f"download failed after {retries} attempts: {last_err}"
    )


def default_settings_path() -> str:
    """Return a writable per-user settings path for the current platform."""
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME")
        root = Path(base) if base else Path.home() / ".config"
    return str(root / "toc-forge" / "settings.json")


def load_settings(
    path: str | None = None, *, legacy_path: str | None = None
) -> dict:
    settings_path = path or default_settings_path()
    candidates = [settings_path]
    if legacy_path and legacy_path != settings_path:
        candidates.append(legacy_path)
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def save_settings(settings: dict, path: str | None = None) -> None:
    settings_path = path or default_settings_path()
    directory = os.path.dirname(settings_path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".settings-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, settings_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def plan_output_names(pdf_paths: Sequence[str]) -> list[str]:
    """Return deterministic, collision-free output basenames for a GUI batch."""
    stems = [Path(path).stem for path in pdf_paths]
    counts = Counter(stem.casefold() for stem in stems)
    used: set[str] = set()
    names: list[str] = []

    for path, stem in zip(pdf_paths, stems):
        if counts[stem.casefold()] > 1:
            normalized = os.path.normcase(os.path.abspath(path))
            suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
            candidate = f"{stem}_{suffix}_bookmarked.pdf"
        else:
            candidate = f"{stem}_bookmarked.pdf"

        base = candidate
        serial = 2
        while candidate.casefold() in used:
            candidate = f"{Path(base).stem}_{serial}.pdf"
            serial += 1
        used.add(candidate.casefold())
        names.append(candidate)
    return names
