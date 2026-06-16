"""toc_forge - Extract table of contents from PDFs and inject navigable bookmarks."""

import sys

__version__ = "0.1.0"

sys.stdout.reconfigure(encoding="utf-8")

from .pipeline import bookmark_pdf  # noqa: E402
from .cli import main  # noqa: E402
from .utils import setup_logger, make_sure_model_exists  # noqa: E402
