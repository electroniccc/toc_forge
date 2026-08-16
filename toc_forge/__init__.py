"""toc_forge - Extract table of contents from PDFs and inject navigable bookmarks."""

import io
import sys

__version__ = "0.2.0"

# 打包为无控制台窗口的程序（--noconsole）时，sys.stdout/stderr 是 None，
# 此时 print() 和 reconfigure() 都会抛 AttributeError。
# 替换成 StringIO 后 print 变成无害的空操作；StringIO 没有 reconfigure，编码设置仅对真实控制台流生效。
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .pipeline import bookmark_pdf  # noqa: E402
from .cli import main  # noqa: E402
from .utils import setup_logger, make_sure_model_exists  # noqa: E402
