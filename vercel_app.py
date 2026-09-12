"""Vercel entrypoint for the serverless FastAPI deployment."""

import logging
import os
import tempfile
from pathlib import Path

import web_app


_runtime_dir = Path(tempfile.gettempdir()) / "toc_forge"
_runtime_dir.mkdir(parents=True, exist_ok=True)

web_app._cfg.update(
    model_dir=os.environ.get("TOC_FORGE_MODEL_DIR", str(_runtime_dir / "models")),
    cache_dir=os.environ.get("TOC_FORGE_CACHE_DIR", str(_runtime_dir / "cache")),
    llm_name=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
    vllm_name=os.environ.get("VLLM_MODEL", "qwen3.6-35b-a3b"),
    engine="onnxruntime",
    device="cpu",
    ocr_model_size="mobile",
)

logging.basicConfig(
    level=os.environ.get("TOC_FORGE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = web_app.app
