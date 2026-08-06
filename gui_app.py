"""TOC Forge — desktop GUI."""
import json
import locale
import os
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, ttk

import requests
import sv_ttk

_MODEL_NAMES = [
    "PP-DocLayout_plus-L",
    "PP-LCNet_x1_0_doc_ori",
    "PP-OCRv5_server_det",
    "PP-OCRv5_server_rec",
    "PP-DocBlockLayout",
]
_BOS_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model"
)
_BOS_VERSION = "paddle3.0.0"

# 每个推理模型由 3 个文件组成；BOS 是打包的 tar，ModelScope/HF 是散文件（内容一致）
_MODEL_FILES = ("inference.yml", "inference.json", "inference.pdiparams")
_MODEL_SOURCES = {
    "baidu": None,  # tar 归档，走 _download_tar
    "modelscope": "https://www.modelscope.cn/models/PaddlePaddle/{name}/resolve/master/{file}",
    "huggingface": "https://huggingface.co/PaddlePaddle/{name}/resolve/main/{file}",
}
# (key, i18n label) — Combobox 显示 label，取值用 key
_SOURCE_KEYS = [
    ("baidu", "src_baidu"),
    ("modelscope", "src_modelscope"),
    ("huggingface", "src_huggingface"),
]

# ---------------------------------------------------------------------------
#  i18n
# ---------------------------------------------------------------------------

_LANG = "en"
if os.name == "nt":
    import ctypes as _ctypes
    try:
        if _ctypes.windll.kernel32.GetUserDefaultUILanguage() == 0x0804:
            _LANG = "zh"
    except Exception:
        pass
else:
    try:
        lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
        if lang.startswith("zh"):
            _LANG = "zh"
    except Exception:
        pass

_T = {
    "en": {
        "title": "TOC Forge",
        "subtitle": "PDF Bookmark Generator",
        "strategy_group": "Extraction method",
        "local_ocr": "Pure OCR",
        "llm": "OCR + Text LLM",
        "vllm": "Vision LLM",
        "api_url": "API Base URL",
        "api_key": "API Key",
        "io_group": "Input / Output",
        "input_pdf": "Input PDF",
        "output_dir": "Output",
        "browse": "Browse…",
        "model_group": "Model directory",
        "model_path": "Path",
        "model_source": "Source",
        "src_baidu": "Baidu (Official)",
        "src_modelscope": "ModelScope (Alibaba)",
        "src_huggingface": "HuggingFace",
        "models_ok": "All models present.",
        "models_missing": "{n} model(s) need to be downloaded.",
        "download_btn": "Download / Verify Models",
        "process_btn": "Generate Bookmarks",
        "processing": "Processing…",
        "open_output": "Open Output",
        "select_pdf": "Select a valid PDF file first.",
        "pdf_only": "Only PDF files are accepted.",
        "need_models": "Download models first.",
        "extracting": "Extracting table of contents …",
        "done": "Done ({t:.1f}s)  —  {name}",
        "error_see_below": "Error — see details below",
        "dl_exists": "{name} already exists",
        "dl_downloading": "Downloading {name} …",
        "dl_extracting": "Extracting {name} …",
        "dl_ready": "{name} ready",
        "model_name": "Model name",
        "file_pdf": "PDF files",
        "file_all": "All files",
    },
    "zh": {
        "title": "TOC Forge",
        "subtitle": "PDF 书签生成器",
        "strategy_group": "提取方式",
        "local_ocr": "纯 OCR",
        "llm": "OCR + 文本 LLM",
        "vllm": "视觉 LLM",
        "api_url": "API 地址",
        "api_key": "API 密钥",
        "io_group": "输入 / 输出",
        "input_pdf": "输入 PDF",
        "output_dir": "输出目录",
        "browse": "浏览…",
        "model_name": "模型名称",
        "model_group": "模型目录",
        "model_path": "路径",
        "model_source": "下载源",
        "src_baidu": "百度官方",
        "src_modelscope": "阿里 ModelScope",
        "src_huggingface": "HuggingFace",
        "models_ok": "所有模型已就位。",
        "models_missing": "还需下载 {n} 个模型。",
        "download_btn": "下载 / 检查模型",
        "process_btn": "生成书签",
        "processing": "处理中…",
        "open_output": "打开输出目录",
        "select_pdf": "请先选择一个有效的 PDF 文件。",
        "pdf_only": "仅支持 PDF 文件。",
        "need_models": "请先下载模型。",
        "extracting": "正在提取目录 …",
        "done": "完成 ({t:.1f}秒)  —  {name}",
        "error_see_below": "错误 — 详情见下方",
        "dl_exists": "{name} 已存在",
        "dl_downloading": "正在下载 {name} …",
        "dl_extracting": "正在解压 {name} …",
        "dl_ready": "{name} 已就位",
        "file_pdf": "PDF 文件",
        "file_all": "所有文件",
    },
}


def t(key: str, **kwargs: object) -> str:
    return _T[_LANG].get(key, _T["en"].get(key, key)).format(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
#  model download with progress
# ---------------------------------------------------------------------------

def _stream_download(url: str, dst: str, progress_cb: Callable[[float], None] | None, retries: int = 3) -> None:
    """带重试的流式下载。

    CDN/中间代理经常在正文未传完时断连（requests 抛 IncompleteRead），
    此时本地文件是残缺的，必须整文件重下。4xx（如 404）不重试。
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                dl = 0
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        dl += len(chunk)
                        f.write(chunk)
                        if progress_cb and total:
                            progress_cb(dl / total)
                if total and dl != total:
                    raise requests.ConnectionError(f"incomplete download: {dl} of {total} bytes")
            return
        except (requests.RequestException, OSError) as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None and 400 <= exc.response.status_code < 500:
                raise
            last_err = exc
            if attempt < retries - 1:
                time.sleep(1.0 + attempt)
    raise requests.ConnectionError(f"download failed after {retries} attempts: {last_err}")


def _download_tar(model_dir: str, model_name: str, progress_cb: Callable[[str, float], None] | None) -> None:
    """百度官方源：下载 {name}_infer.tar 并解压到 model_dir/{name}/。"""
    target = os.path.join(model_dir, model_name)
    url = f"{_BOS_BASE_URL}/{_BOS_VERSION}/{model_name}_infer.tar"
    if progress_cb:
        progress_cb(t("dl_downloading", name=model_name), 0)

    def _prog(frac: float) -> None:
        if progress_cb:
            progress_cb(t("dl_downloading", name=model_name), frac)

    with tempfile.TemporaryDirectory() as td:
        arc_path = os.path.join(td, f"{model_name}_infer.tar")
        _stream_download(url, arc_path, _prog)

        if progress_cb:
            progress_cb(t("dl_extracting", name=model_name), 1.0)
        extract_dir = os.path.join(td, "extract")
        shutil.unpack_archive(arc_path, extract_dir)
        entries = os.listdir(extract_dir)
        src = os.path.join(extract_dir, entries[0] if len(entries) == 1 else model_name)
        if os.path.isdir(src):
            shutil.copytree(src, target, symlinks=True)
        else:
            os.makedirs(target, exist_ok=True)
            shutil.copy2(src, target)


def _download_model_files(model_dir: str, model_name: str, source: str, progress_cb: Callable[[str, float], None] | None) -> None:
    """ModelScope / HuggingFace 源：逐文件下载 3 个推理文件（与 BOS tar 内容一致）。"""
    template = _MODEL_SOURCES[source]
    n = len(_MODEL_FILES)
    target = os.path.join(model_dir, model_name)
    os.makedirs(target, exist_ok=True)

    for i, fname in enumerate(_MODEL_FILES):
        url = template.format(name=model_name, file=fname)
        label = f"{model_name}/{fname}"
        if progress_cb:
            progress_cb(t("dl_downloading", name=label), i / n)
        dst = os.path.join(target, fname)

        def _prog(frac: float) -> None:
            if progress_cb:
                progress_cb(t("dl_downloading", name=label), (i + frac) / n)

        _stream_download(url, dst, _prog)


def _download_model(model_dir: str, model_name: str, source: str = "baidu", progress_cb: Callable[[str, float], None] | None = None) -> None:
    target = os.path.join(model_dir, model_name)
    if os.path.isdir(target):
        if progress_cb:
            progress_cb(t("dl_exists", name=model_name), 1.0)
        return

    os.makedirs(model_dir, exist_ok=True)
    if source == "baidu":
        _download_tar(model_dir, model_name, progress_cb)
    else:
        _download_model_files(model_dir, model_name, source, progress_cb)

    if progress_cb:
        progress_cb(t("dl_ready", name=model_name), 1.0)


def download_all_models(model_dir: str, source: str = "baidu", progress_cb: Callable[[str, float], None] | None = None) -> None:
    """Download all models in parallel; overall progress = mean of per-model progress."""
    n = len(_MODEL_NAMES)
    lock = threading.Lock()
    fracs: dict[str, float] = {}

    def _cb(name: str, msg: str, frac: float) -> None:
        with lock:
            fracs[name] = frac
            overall = sum(fracs.values()) / n
        if progress_cb:
            progress_cb(msg, overall)

    # 3 workers: 5 models × ~130MB, 网络是瓶颈，再多线程收益有限
    with ThreadPoolExecutor(max_workers=min(3, n)) as pool:
        futures = [
            pool.submit(
                _download_model, model_dir, name, source,
                lambda msg, frac, _name=name: _cb(_name, msg, frac),
            )
            for name in _MODEL_NAMES
        ]
        for future in futures:
            future.result()  # 第一个异常往上抛，其余 worker 继续跑完


def all_models_exist(model_dir: str) -> bool:
    return all(os.path.isdir(os.path.join(model_dir, n)) for n in _MODEL_NAMES)


# ---------------------------------------------------------------------------
#  settings persistence
# ---------------------------------------------------------------------------

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), ".gui_settings.json")


def _load_settings() -> dict:
    if os.path.isfile(_SETTINGS_PATH):
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_settings(settings: dict) -> None:
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
#  GUI
# ---------------------------------------------------------------------------

class TocForgeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(t("title"))
        root.geometry("620x560")
        root.minsize(520, 460)

        self.settings = _load_settings()
        self._running = False
        self._last_output_dir: str = ""

        self._build_ui()
        self._load_settings_to_ui()

        if not all_models_exist(self.model_dir_var.get()):
            root.after(300, self._expand_model_section)

    # ------------------------------------------------------------------
    #  UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding="24 20 24 20")
        main.pack(fill=tk.BOTH, expand=True)

        # --- header ---
        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 16))
        ttk.Label(header, text=t("title"), font=("Segoe UI Variable", 18, "bold")).pack(side=tk.LEFT)
        ttk.Label(
            header, text=t("subtitle"),
            font=("Segoe UI Variable", 10), foreground="#888",
        ).pack(side=tk.LEFT, padx=(12, 0), pady=(4, 0))

        # --- strategy ---
        strat_frame = ttk.LabelFrame(main, text=t("strategy_group"), padding="12 8 12 12")
        strat_frame.pack(fill=tk.X, pady=(0, 12))

        self.strategy_var = tk.StringVar(value="local_ocr")
        strat_row = ttk.Frame(strat_frame)
        strat_row.pack(fill=tk.X)
        for val in ("local_ocr", "llm", "vllm"):
            rb = ttk.Radiobutton(
                strat_row, text=t(val), variable=self.strategy_var,
                value=val, command=self._on_strategy_change,
            )
            rb.pack(side=tk.LEFT, padx=(0, 24))

        # --- API fields ---
        self.api_frame = ttk.Frame(strat_frame)

        url_row = ttk.Frame(self.api_frame)
        url_row.pack(fill=tk.X, pady=(12, 6))
        ttk.Label(url_row, text=t("api_url"), width=14).pack(side=tk.LEFT)
        self.api_url_var = tk.StringVar()
        ttk.Entry(url_row, textvariable=self.api_url_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        key_row = ttk.Frame(self.api_frame)
        key_row.pack(fill=tk.X)
        ttk.Label(key_row, text=t("api_key"), width=14).pack(side=tk.LEFT)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(key_row, textvariable=self.api_key_var, show="•")
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._key_visible = False
        self._eye_canvas = tk.Canvas(key_row, width=20, height=16, highlightthickness=0, bd=0, cursor="hand2")
        self._eye_canvas.pack(side=tk.RIGHT, padx=(4, 0))
        self._draw_eye()
        self._eye_canvas.bind("<Button-1>", lambda _e: self._toggle_key_visibility())

        self.model_name_row = ttk.Frame(self.api_frame)
        self.model_name_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(self.model_name_row, text=t("model_name"), width=14).pack(side=tk.LEFT)
        self.model_name_var = tk.StringVar()
        ttk.Entry(self.model_name_row, textvariable=self.model_name_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- file paths ---
        path_frame = ttk.LabelFrame(main, text=t("io_group"), padding="12 8 12 12")
        path_frame.pack(fill=tk.X, pady=(0, 12))

        pdf_row = ttk.Frame(path_frame)
        pdf_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(pdf_row, text=t("input_pdf"), width=14).pack(side=tk.LEFT)
        self.pdf_path_var = tk.StringVar()
        ttk.Entry(pdf_row, textvariable=self.pdf_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(pdf_row, text=t("browse"), command=self._browse_pdf).pack(side=tk.RIGHT, padx=(8, 0))

        out_row = ttk.Frame(path_frame)
        out_row.pack(fill=tk.X)
        ttk.Label(out_row, text=t("output_dir"), width=14).pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(out_row, textvariable=self.output_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text=t("browse"), command=self._browse_output).pack(side=tk.RIGHT, padx=(8, 0))

        # --- model section ---
        self.model_frame = ttk.LabelFrame(main, text=t("model_group"), padding="12 8 12 12")

        model_row = ttk.Frame(self.model_frame)
        model_row.pack(fill=tk.X)
        ttk.Label(model_row, text=t("model_path"), width=14).pack(side=tk.LEFT)
        self.model_dir_var = tk.StringVar(value="./models")
        ttk.Entry(model_row, textvariable=self.model_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_row, text=t("browse"), command=self._browse_model_dir).pack(side=tk.RIGHT, padx=(8, 0))

        src_row = ttk.Frame(self.model_frame)
        src_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(src_row, text=t("model_source"), width=14).pack(side=tk.LEFT)
        self.model_source_var = tk.StringVar()
        ttk.Combobox(
            src_row, textvariable=self.model_source_var, state="readonly",
            values=[t(label) for _k, label in _SOURCE_KEYS],
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.model_status_var = tk.StringVar()
        ttk.Label(
            self.model_frame, textvariable=self.model_status_var,
            font=("Segoe UI Variable", 9), foreground="#666",
        ).pack(anchor=tk.W, pady=(8, 0))

        self.model_progress = ttk.Progressbar(self.model_frame, mode="determinate")
        self.model_progress.pack(fill=tk.X, pady=(6, 8))

        btn_row = ttk.Frame(self.model_frame)
        btn_row.pack(fill=tk.X)
        self.download_btn = ttk.Button(btn_row, text=t("download_btn"), command=self._download_models)
        self.download_btn.pack()

        # --- bottom bar ---
        bottom = ttk.Frame(main)
        bottom.pack(fill=tk.X, pady=(4, 0))

        self.status_var = tk.StringVar()
        ttk.Label(
            bottom, textvariable=self.status_var,
            font=("Segoe UI Variable", 9), foreground="#888",
        ).pack(side=tk.LEFT, pady=(0, 0))

        # Error detail box — hidden by default, shown on error with selectable text
        self.error_text = tk.Text(
            main, font=("Consolas", 9), fg="#c44", bg="#fef2f2",
            borderwidth=1, relief=tk.SOLID, highlightthickness=0,
            height=4, wrap=tk.WORD, state=tk.DISABLED,
        )

        # --- progress bar (shown during processing) ---
        self.process_progress = ttk.Progressbar(bottom, mode="indeterminate", length=160)

        self.process_btn = ttk.Button(bottom, text=t("process_btn"), command=self._on_action_btn)
        self.process_btn.pack(side=tk.RIGHT)

        sv_ttk.set_theme("light")

    # ------------------------------------------------------------------
    #  settings
    # ------------------------------------------------------------------

    def _load_settings_to_ui(self) -> None:
        s = self.settings
        self.strategy_var.set(s.get("strategy", "local_ocr"))
        self.pdf_path_var.set(s.get("pdf_path", ""))
        self.output_dir_var.set(s.get("output_dir", ""))
        self.model_dir_var.set(s.get("model_dir", "./models"))
        self._set_source(str(s.get("model_source", "baidu")))
        self.api_url_var.set(s.get("api_url", ""))
        self.api_key_var.set(s.get("api_key", ""))
        self.model_name_var.set(s.get("model_name", ""))
        self._on_strategy_change()
        self._update_model_status()

    def _persist_settings(self) -> None:
        self.settings.update(
            strategy=self.strategy_var.get(),
            pdf_path=self.pdf_path_var.get(),
            output_dir=self.output_dir_var.get(),
            model_dir=self.model_dir_var.get(),
            model_source=self._source_key(),
            api_url=self.api_url_var.get(),
            api_key=self.api_key_var.get(),
            model_name=self.model_name_var.get(),
        )
        _save_settings(self.settings)

    def _source_key(self) -> str:
        display = self.model_source_var.get()
        for key, label in _SOURCE_KEYS:
            if t(label) == display:
                return key
        return "baidu"

    def _set_source(self, key: str) -> None:
        for k, label in _SOURCE_KEYS:
            if k == key:
                self.model_source_var.set(t(label))
                return
        self.model_source_var.set(t(_SOURCE_KEYS[0][1]))

    # ------------------------------------------------------------------
    #  action button (process / open output)
    # ------------------------------------------------------------------

    def _on_action_btn(self) -> None:
        if self._running:
            return
        current_text = self.process_btn.cget("text")
        if current_text == t("open_output") and self._last_output_dir:
            self._open_dir(self._last_output_dir)
        else:
            self._process_pdf()

    def _switch_to_open_btn(self) -> None:
        self.process_btn.configure(text=t("open_output"))

    def _switch_to_process_btn(self) -> None:
        self.process_btn.configure(text=t("process_btn"))

    @staticmethod
    def _open_dir(path: str) -> None:
        if os.path.isdir(path):
            subprocess.Popen(["explorer", os.path.abspath(path)])

    # ------------------------------------------------------------------
    #  callbacks
    # ------------------------------------------------------------------

    def _draw_eye(self) -> None:
        c = self._eye_canvas
        c.delete("all")
        w, h = 20, 16
        if self._key_visible:
            # eye with slash — draw an X over the eye
            c.create_oval(2, 3, 17, 12, outline="#555", width=1.5)
            c.create_oval(8, 6, 11, 9, outline="#555", fill="#555", width=1)
            c.create_line(0, 0, w, h, fill="#555", width=1.5)
        else:
            c.create_oval(2, 3, 17, 12, outline="#555", width=1.5)
            c.create_oval(8, 6, 11, 9, outline="#555", fill="#555", width=1)

    def _toggle_key_visibility(self) -> None:
        self._key_visible = not self._key_visible
        self.api_key_entry.configure(show="" if self._key_visible else "•")
        self._draw_eye()

    def _on_strategy_change(self, *_args: object) -> None:
        strategy = self.strategy_var.get()
        if strategy == "local_ocr":
            self.api_frame.pack_forget()
        else:
            if not self.api_frame.winfo_ismapped():
                self.api_frame.pack(fill=tk.X)
        self._persist_settings()

    def _browse_pdf(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[(t("file_pdf"), "*.pdf"), (t("file_all"), "*.*")],
        )
        if path:
            self.pdf_path_var.set(path)
            self._persist_settings()

    def _browse_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)
            self._persist_settings()

    def _browse_model_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.model_dir_var.set(path)
            self._persist_settings()
            self._update_model_status()

    def _update_model_status(self) -> None:
        model_dir = self.model_dir_var.get()
        if all_models_exist(model_dir):
            self.model_status_var.set(t("models_ok"))
            self.model_progress["value"] = 100
        else:
            missing = sum(1 for n in _MODEL_NAMES if not os.path.isdir(os.path.join(model_dir, n)))
            self.model_status_var.set(t("models_missing", n=missing))
            self.model_progress["value"] = 0
        self._toggle_model_section()

    def _toggle_model_section(self) -> None:
        if all_models_exist(self.model_dir_var.get()) and not self.model_frame.winfo_ismapped():
            return
        if not self.model_frame.winfo_ismapped():
            self.model_frame.pack(fill=tk.X, pady=(0, 12), before=self.process_btn.master)

    def _expand_model_section(self) -> None:
        if not self.model_frame.winfo_ismapped():
            self.model_frame.pack(fill=tk.X, pady=(0, 12), before=self.process_btn.master)

    # ------------------------------------------------------------------
    #  model download
    # ------------------------------------------------------------------

    def _download_models(self) -> None:
        model_dir = self.model_dir_var.get()
        source = self._source_key()
        self._persist_settings()
        self.download_btn.configure(state=tk.DISABLED)
        self.process_btn.configure(state=tk.DISABLED)
        self.model_progress["value"] = 0
        self.error_text.pack_forget()

        def _cb(msg: str, frac: float) -> None:
            self.root.after(0, lambda: self._on_dl_progress(msg, frac))

        def _work() -> None:
            try:
                download_all_models(model_dir, source, _cb)
            except (OSError, requests.RequestException) as exc:
                err_msg = str(exc)
                self.root.after(0, lambda m=err_msg: self._on_dl_error(m))

        threading.Thread(target=_work, daemon=True).start()

    def _on_dl_progress(self, msg: str, frac: float) -> None:
        self.model_status_var.set(msg)
        self.model_progress["value"] = frac * 100
        if frac >= 1.0 and self.model_progress["value"] >= 100:
            self._update_model_status()
            self.download_btn.configure(state=tk.NORMAL)
            self.process_btn.configure(state=tk.NORMAL)
            if all_models_exist(self.model_dir_var.get()):
                self.error_text.pack_forget()
                self.root.after(1500, self._maybe_hide_model_section)

    def _on_dl_error(self, err: str) -> None:
        self.download_btn.configure(state=tk.NORMAL)
        self.process_btn.configure(state=tk.NORMAL)
        self._show_error(err)

    def _maybe_hide_model_section(self) -> None:
        if all_models_exist(self.model_dir_var.get()):
            self.model_frame.pack_forget()

    # ------------------------------------------------------------------
    #  processing
    # ------------------------------------------------------------------

    def _process_pdf(self) -> None:
        if self._running:
            return
        pdf_path = self.pdf_path_var.get().strip()
        if not pdf_path or not os.path.isfile(pdf_path):
            self.status_var.set(t("select_pdf"))
            return
        if not pdf_path.lower().endswith(".pdf"):
            self.status_var.set(t("pdf_only"))
            return
        output_dir = self.output_dir_var.get().strip() or "output"
        model_dir = self.model_dir_var.get().strip() or "./models"
        strategy = self.strategy_var.get()

        if not all_models_exist(model_dir):
            self._expand_model_section()
            self.status_var.set(t("need_models"))
            return

        self._persist_settings()

        api_base = self.api_url_var.get().strip().replace("\n", "").replace("\r", "") if strategy != "local_ocr" else None
        api_key = self.api_key_var.get().strip() if strategy != "local_ocr" else None

        self._running = True
        self._last_output_dir = ""
        self._switch_to_process_btn()
        self.process_btn.configure(state=tk.DISABLED, text=t("processing"))
        self.process_progress.pack(side=tk.RIGHT, padx=(0, 8))
        self.process_progress.start()
        self.status_var.set(t("extracting"))

        def _work() -> None:
            try:
                # 限制推理线程数，给 UI 留 CPU：paddle 的 OpenMP 推理默认开满
                # 所有核（paddlex 默认 10 线程 + busy-wait），会把 tkinter 主循环
                # 饿死、界面完全卡死。留出 2 个核给 UI 和系统。
                n_cpu = os.cpu_count() or 8
                ocr_threads = max(2, n_cpu - 2)
                # 必须在 import paddle 之前设置（paddle 首次导入发生在
                # bookmark_pdf 内部）：OMP_NUM_THREADS 管 OpenMP 算子，
                # PADDLE_PDX_CPU_NUM_THREADS 管 paddlex predictor 的默认线程数
                os.environ["OMP_NUM_THREADS"] = str(ocr_threads)
                os.environ["PADDLE_PDX_CPU_NUM_THREADS"] = str(ocr_threads)

                import toc_forge
                toc_forge.setup_logger("log")
                model_name = self.model_name_var.get().strip() or None
                llm_name = model_name if strategy == "llm" else None
                vllm_name = model_name if strategy == "vllm" else None
                pdf_out, elapsed, _ = toc_forge.bookmark_pdf(
                    input=pdf_path,
                    output=output_dir,
                    model_dir=model_dir,
                    cache_dir="./.ocr_cache",
                    toc_strategy=strategy,
                    api_base_url=api_base,
                    api_key=api_key,
                    llm_name=llm_name,
                    vllm_name=vllm_name,
                    # 强制 CPU：paddle 3.x 统一 wheel 自带 CUDA 内核，在装了 N 卡驱动
                    # 的机器上会自动选 gpu 并尝试加载 cudnn64_9.dll，而打包程序不带
                    # CUDA/cuDNN 库，会直接崩溃（error code 126）
                    device="cpu",
                    cpu_threads=ocr_threads,
                )
                self.root.after(0, lambda: self._on_done(pdf_out, elapsed))
            except Exception as exc:
                err_msg = str(exc)
                self.root.after(0, lambda m=err_msg: self._on_error(m))

        threading.Thread(target=_work, daemon=True).start()

    def _on_done(self, pdf_out: str, elapsed: float) -> None:
        self._running = False
        self._last_output_dir = os.path.dirname(pdf_out)
        self.process_progress.stop()
        self.process_progress.pack_forget()
        self.process_btn.configure(state=tk.NORMAL)
        self._switch_to_open_btn()
        self.error_text.pack_forget()
        self.status_var.set(t("done", t=elapsed, name=os.path.basename(pdf_out)))

    def _on_error(self, msg: str) -> None:
        self._running = False
        self.process_progress.stop()
        self.process_progress.pack_forget()
        self.process_btn.configure(state=tk.NORMAL)
        self._switch_to_process_btn()
        self._show_error(msg)

    def _show_error(self, msg: str) -> None:
        self.status_var.set(t("error_see_below"))
        self.error_text.configure(state=tk.NORMAL)
        self.error_text.delete("1.0", tk.END)
        self.error_text.insert("1.0", msg)
        self.error_text.configure(state=tk.DISABLED)
        self.error_text.pack(fill=tk.X, pady=(8, 0))


# ---------------------------------------------------------------------------
#  entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    TocForgeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
