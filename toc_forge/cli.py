"""CLI entry point for toc-forge."""

import logging
import os
from argparse import ArgumentParser

from .pipeline import bookmark_pdf
from .utils import format_duration, setup_logger

logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input PDF file")
    parser.add_argument(
        "--output", type=str, default="output directory", help="output directory"
    )
    parser.add_argument("--log_dir", type=str, default="log", help="log directory")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="where the OCR models are stored",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./.ocr_cache",
        help="where the OCR cache is stored",
    )
    parser.add_argument(
        "--api_base_url", type=str, default=None, help="OpenAI API base url"
    )
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument(
        "--llm_name", type=str, default="", help="text LLM, like deepseek-v4-flash"
    )
    parser.add_argument(
        "--vllm_name",
        type=str,
        default="",
        help="visual LLM or multi-modal LLM, like qwen3.6-flash",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--hash", action="store_true", help="print input file hash and exit")
    args = parser.parse_args()

    if args.hash and args.input:
        from .utils import compute_file_hash

        print(f"file_hash: {compute_file_hash(args.input)}")
        return

    if not args.model_dir:
        print("model dir required")
        return
    setup_logger(args.log_dir)

    api_base_url = args.api_base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    if api_base_url and api_key and args.vllm_name:
        toc_strategy = "vllm"
    elif api_base_url and api_key and args.llm_name:
        toc_strategy = "llm"
    else:
        toc_strategy = "local_ocr"

    logger.info("toc_strategy=%s", toc_strategy)

    pdf_bookmarks_path, time_cost, _ = bookmark_pdf(
        args.input,
        args.output,
        args.model_dir,
        do_debug=args.debug,
        cache_dir=args.cache_dir,
        toc_strategy=toc_strategy,
        api_base_url=api_base_url,
        api_key=api_key,
        llm_name=args.llm_name,
        vllm_name=args.vllm_name,
    )
    print(
        f"Bookmarked PDF saved to: {pdf_bookmarks_path}, "
        f"Time elapsed: {format_duration(time_cost)}"
    )
