import argparse
import asyncio
import logging
import os
import shutil
import tempfile
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

import toc_forge

logger = logging.getLogger("web_app")

app = FastAPI(title="TOC Forge")

_cfg: dict = {}


def _run_bookmark(
    input_path: str,
    output_dir: str,
    toc_strategy: str,
    api_base_url: str | None,
    api_key: str | None,
) -> tuple[str, float]:
    return toc_forge.bookmark_pdf(
        input=input_path,
        output=output_dir,
        model_dir=_cfg["model_dir"],
        cache_dir=_cfg["cache_dir"],
        toc_strategy=toc_strategy,
        api_base_url=api_base_url,
        api_key=api_key,
        llm_name=_cfg["llm_name"],
        vllm_name=_cfg["vllm_name"],
    )


@app.post("/api/bookmark_pdf")
async def bookmark_pdf(
    file: UploadFile = File(...),
    toc_strategy: str = Form("local_ocr"),
    api_base_url: str | None = Form(None),
    api_key: str | None = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    if toc_strategy not in ("local_ocr", "llm", "vllm"):
        raise HTTPException(400, "Invalid toc_strategy")

    tmpdir = tempfile.mkdtemp(prefix="toc_forge_")
    input_path = os.path.join(tmpdir, file.filename)
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_path, "wb") as f:
            while chunk := await file.read(8 * 1024 * 1024):
                f.write(chunk)

        logger.info(
            "Processing %s (size=%d, strategy=%s)",
            file.filename, os.path.getsize(input_path), toc_strategy,
        )
        loop = asyncio.get_running_loop()
        pdf_path, elapsed = await loop.run_in_executor(
            None, _run_bookmark, input_path, output_dir, toc_strategy, api_base_url, api_key,
        )
        logger.info("Done %s in %.1fs", file.filename, elapsed)

        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(500, "Bookmark generation failed")

        stem = Path(file.filename).stem
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{stem}_bookmarked.pdf",
            background=BackgroundTask(shutil.rmtree, tmpdir, ignore_errors=True),
        )

    except HTTPException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.exception("Failed to process %s", file.filename)
        raise HTTPException(500, str(e))


static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="TOC Forge Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--cache_dir", type=str, default="./.ocr_cache")
    parser.add_argument("--log_dir", type=str, default="log")
    args = parser.parse_args()

    toc_forge.setup_logger(args.log_dir)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _cfg.update(
        model_dir=args.model_dir,
        cache_dir=args.cache_dir,
        llm_name=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        vllm_name=os.environ.get("VLLM_MODEL", "qwen3.6-35b-a3b"),
    )

    logger.info(
        "Starting server on %s:%d (model_dir=%s, cache_dir=%s)",
        args.host, args.port, args.model_dir, args.cache_dir,
    )

    host_render = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{host_render}:{args.port}"
    print(f"  TOC Forge: {url}")
    webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
