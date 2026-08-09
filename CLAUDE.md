# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`toc_forge` extracts a table of contents from a PDF using PaddleOCR layout/OCR models and injects the resulting TOC tree back into the PDF as navigable bookmarks (outline). The code is organized as an installable Python package under `toc_forge/`.

## Environment setup

Source the env script before running:
```powershell
.\.set_paddlex_env.ps1
```
This sets `PADDLE_HUB_HOME`, `PADDLE_PDX_CACHE_HOME`, and disables model source checks. The script checks for `PADDLE_PDX_CACHE_HOME` at startup and refuses to run without it.

The project uses a local `.venv`. Install in editable mode:
```powershell
pip install -e .
```
Key dependencies: `paddleocr`, `paddlex`, `opencv-python` (cv2), `PyMuPDF` (fitz), `Pillow`, `numpy`, `scikit-learn`, `openai`.

## Running

After `pip install -e .`:
```powershell
toc-forge --input <pdf_path> --output <output_dir> [--model_dir ./models] [--debug]
```

Or without installing:
```powershell
python -m toc_forge --input <pdf_path> --output <output_dir> [--model_dir ./models] [--debug]
```

- `--input`: path to the source PDF
- `--output`: output directory (default: `output directory`)
- `--model_dir`: directory containing PaddleOCR models (default: `./models`)
- `--cache_dir`: OCR result cache directory (default: `./.ocr_cache`)
- `--log_dir`: log directory (default: `log`)
- `--debug`: saves intermediate layout/OCR/parsing results as images and JSON
- `--hash`: print the input file's hash (used to locate OCR cache) and exit
- `--api_base_url`: OpenAI-compatible API base URL (also reads `OPENAI_BASE_URL` env var)
- `--api_key`: API key (also reads `OPENAI_API_KEY` env var)
- `--llm_name`: text LLM model name for the `llm` strategy (e.g. `deepseek-v4-flash`)
- `--vllm_name`: vision LLM model name for the `vllm` strategy (e.g. `qwen3.6-35b-a3b`)
- `--no_toc_cache`: for `llm`/`vllm` strategies, re-call the LLM even if a cached TOC tree exists
- `--device`: device for PaddleOCR inference — `cpu`, `gpu`, `gpu:0`, etc. (default: auto-detect)
- `--llm_timeout`: LLM API request timeout in seconds (default: `600`). Increase if using slow reasoning models.
- `--engine`: inference engine for PaddleOCR models — `paddle`, `paddle_static`, `paddle_dynamic`, `onnxruntime`, etc. (default: PaddleX auto). With `onnxruntime`, each model directory must contain an `inference.onnx` — see "ONNX Runtime engine" below.
- `--disable_mkldnn`: disable MKLDNN for CPU inference (workaround for the paddle 3.3.1 oneDNN executor crash on Windows, `ConvertPirAttribute2RuntimeAttribute`). Sets `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=false` in `cli.py` **before** importing paddle — the `enable_mkldnn` kwarg is a no-op in paddlex 3.5.2/3.7.2 (verified), the env var is the real switch. Default: MKLDNN on (Linux runs it normally). Only affects the paddle engine — ignored by `onnxruntime`.
- `--ocr_model_size`: OCR det/rec model size — `server` (default, higher accuracy) or `mobile` (much faster on CPU, what the GUI uses). With `--engine onnxruntime`, the chosen model directory must contain `inference.onnx` (see "ONNX Runtime engine").

Output: `{output}/{input_stem}_bookmarked.pdf` with injected PDF outline.

## Architecture

The pipeline has five stages (`bookmark_pdf` in `toc_forge/pipeline.py:118`):

1. **Page image extraction** (`image_from_page`): renders the first 30 PDF pages to numpy arrays via PyMuPDF at 2x zoom, falling back to 1x if the image exceeds 2000px in either dimension.

2. **Layout detection** (`get_toc_pages`): runs `PP-DocLayout_plus-L` on the page images to find "content" boxes (TOC blocks) and "number" boxes (page numbers). On pages that have `content` boxes, `paragraph_title` boxes sitting above each content box are also collected — these are often chapter/part headings (e.g. "第一篇", "第1章") that the layout model didn't label as "content". Deduplicates overlapping boxes via `deduplicate_content_boxes`.

3. **OCR on TOC pages** (`ocr_toc_pages`): runs PaddleOCR on each detected TOC page, then filters results to only keep text inside the content boxes (`filter_toc_result`).

4. **TOC tree reconstruction** — three strategies exist, auto-selected in `cli.py:main()`:
   - **`local_ocr` (default)**: uses gap-clustering + indentation parsing (`reconstruct_toc1`). No API needed.
   - **`llm`**: runs local OCR first, then sends OCR JSON to a text LLM (`build_toc_llm`). Selected when `--api_base_url`, `--api_key`, and `--llm_name` are set but `--vllm_name` is not.
   - **`vllm`**: sends TOC page images directly to a vision LLM, skipping local OCR (`build_toc_vllm`). Selected when all three of `--api_base_url`, `--api_key`, and `--vllm_name` are set.

   For `local_ocr`, three heuristic parsers exist:
   - `reconstruct_toc` (legacy): purely semantic — regex patterns for chapter/section/subsection, KMeans fallback.
   - `reconstruct_toc_indent` (legacy): pure indentation clustering on x-coordinates.
   - `reconstruct_toc1` (**the active/default strategy**): per-page level detection + cross-page merge. Level assignment is **geometry-driven and language-agnostic**:
     1. **Gap-tree clustering**: x-coordinate gaps determine baseline indentation levels for all entries on a page.
     2. **Paragraph-title depth**: consecutive `paragraph_title` entries above a content box form a nesting chain (first → level 0, second → level 1, …). Content entries are shifted by the number of consecutive paragraph_titles above them.
     3. **Semantic floors** (supplementary): chapter-like patterns (e.g. `第X章`) are forced to level 0; section/subsection patterns set minimum levels — applied only to entries from `content` boxes, never to `paragraph_title` entries.
     4. **KMeans fallback**: if gap-tree produces a single bin but x-variance is large, force a 2-cluster split.
   - Per-page mini-trees are built via `_build_tree` (same-level entries are siblings, only strictly-deeper entries nest).
   - `_merge_page_trees` merges per-page trees: recognizes section-like entries (sections, exercises, numbered entries) and attaches them under the last chapter, with exercises matched to their parent section by number.
   - `inherit_page_numbers` fills missing page numbers (common for `paragraph_title` entries) by inheriting from descendants (max depth 3) or the next sibling.
   - `repair_toc_tree` post-processes in four passes:
     - **Pass 0** (`_fix_pian_structure`): nests `第X章` entries under their preceding `第X篇`.
     - **Pass 1** (`_fix_zhang_sections`): re-parents orphan numbered sections (e.g. `7.1`) under their matching chapter (`第7章`) by prefix number.
     - **Pass 2**: root-level fixup — re-parents orphan section-like entries under the last chapter.
     - **Pass 3** (`_fix_chapter_children`): re-parents exercises (e.g. `习题1-5`) under their matching section (`第五节`).

   The shared parsing pipeline (`_parse_toc_lines`):
   - Accepts optional `page_heights: list[float] | None`
   - Flattens OCR text boxes across pages with cumulative y-offsets
   - Each item carries `cb_label` (`"content"` or `"paragraph_title"`) from its layout box
   - Groups items into lines by y-overlap
   - Parses each line into (title, page_num) by detecting page number fragments at the rightmost end — handles plain digits, Roman numerals, parenthesized digits, trailed dots, and dot-leader patterns (e.g. `"…………6"`)
   - Strips leading bullet markers (`*`, `•`, `·`) from titles

5. **Page offset + bookmark injection** (`get_page_offset2`, `add_bookmarks_to_pdf`): computes the offset between printed-page and PDF-page indexing **without PPStructureV3**. Layout detection already locates "number" blocks (printed page numbers) per page; `get_page_offset2` filters them by parity-group position consistency (odd pages' number boxes should mostly overlap, even pages likewise — accidental boxes are dropped), crops each kept page around its number box, OCRs just the crop with the existing `PaddleOCR` model, parses the printed number, and takes the mode of `pdf_page_idx - page_num`. Then calls `doc.set_toc()` + `doc.save()` to write the PDF outline. The logic is split into three independently-testable stages: `get_number_box_pages` (collect + parity-consistency filter), `ocr_number_boxes` (crop + OCR + cache), `compute_page_offset` (parse + mode), with `get_page_offset2` as the wrapper.

### Module organization

| File | Responsibility |
|---|---|
| `toc_forge/__init__.py` | Package version, `sys.stdout` encoding setup, re-exports `bookmark_pdf` and `main` |
| `toc_forge/cli.py` | CLI entry point: argparse config, strategy auto-detection, calls `bookmark_pdf` |
| `toc_forge/pipeline.py` | Top-level orchestration: `bookmark_pdf`, `build_toc_local_ocr`, `add_bookmarks_to_pdf`, `get_page_offset` |
| `toc_forge/ocr_engine.py` | PaddleOCR calls: `get_toc_pages` (layout), `ocr_toc_pages` (text), `get_number_box_pages` / `ocr_number_boxes` / `compute_page_offset` (page-offset stages, wrapped by `get_page_offset2`) |
| `toc_forge/parsing.py` | Heuristic TOC tree reconstruction: `_parse_toc_lines`, `_build_tree`, `reconstruct_toc1`, `_merge_page_trees`, `_merge_content_box_trees`, `repair_toc_tree`, `_fix_pian_structure`, `_fix_zhang_sections`, `inherit_page_numbers` |
| `toc_forge/llm.py` | LLM strategies: `build_toc_llm`, `build_toc_vllm`, `_call_llm`, system prompts |
| `toc_forge/utils.py` | Shared utilities: caching, image processing, model download, numeral helpers (`_section_sort_key`, `_roman_to_int`), `NumpyEncoder`, logging |
| `gui_app.py` | Desktop GUI (Tkinter + sv_ttk): strategy selection, model download (2 onnx sources: ModelScope / HuggingFace `{name}_onnx` repos), settings persistence; runs `bookmark_pdf` (onnxruntime engine, CPU) in a worker thread |
| `build_gui.ps1` / `build_gui_pyinstaller.ps1` | Nuitka / PyInstaller packaging scripts for the GUI (console-less exe) |

### OCR models used

The pipeline downloads/caches four PaddleOCR model types via `make_sure_model_exists()`:

| Model | Stage |
|---|---|
| `PP-DocLayout_plus-L` | Layout detection |
| `PP-LCNet_x1_0_doc_ori` | Document orientation classification |
| `PP-OCRv5_{server,mobile}_det` | Text detection (size per `--ocr_model_size`) |
| `PP-OCRv5_{server,mobile}_rec` | Text recognition (size per `--ocr_model_size`) |

`PP-DocBlockLayout` is **not used** — it was region detection for the removed PPStructureV3 page-number scan; it is gone from the GUI download list (`_MODEL_NAMES` in `gui_app.py`), though its files may still exist in `models/`.

**`PP-DocLayout-M` was evaluated and rejected.** It has no official `_onnx` repo; the onnx comes from RapidDoc (ModelScope `RapidAI/RapidDoc`, `layout/PP-DocLayout-M/pp_doclayout_m.onnx`), verified bit-identical in output to the official paddle weights on the same pages (paddle2onnx itself fails to load against paddle 3.3.1: `DLL load failed … 找不到指定的程序`). It runs ~2× faster than plus-L (0.10–0.12 s vs ~0.25 s per page) but detects far fewer boxes on TOC pages (e.g. 0 vs 3, 8 vs 20) — page 4 of the test book produced **zero** boxes, which makes `bookmark_pdf` exit with "未检测到目录页". The pipeline stays on `PP-DocLayout_plus-L`; the downloaded model dir may still exist under `models/`.

Models are downloaded from `paddle-model-ecology.bj.bcebos.com` if not found locally, or copied from `PADDLE_PDX_CACHE_HOME/official_models` if already cached there.

### OCR result caching

Layout, OCR, and structure results are cached per-PDF and per-page under `--cache_dir/{pdf_hash}/`. Cache keys are `{stage}_page_{idx}.json` (or `{stage}.json` for non-per-page results). Caching uses `_cache_load` / `_cache_save` with the `CachedResult` wrapper class that mimics PaddleX result objects. Legacy wrapped format `{"res": {...}}` is unwrapped via `_unwrap_legacy_cache`.

The final TOC tree produced by the `llm`/`vllm` strategies is also cached as `toc_tree_llm.json` / `toc_tree_vllm.json` under the same directory, so repeat runs of the same document skip the LLM call (and the OCR it depends on). `--no_toc_cache` forces a fresh LLM call; the fresh result still refreshes the cache. Handled by `_load_toc_tree_cache` / `_save_toc_tree_cache` in `toc_forge/llm.py`.

### LLM integration

LLM strategies use the `openai` SDK via `_build_llm_client()` and `_call_llm()`. Both system prompts are defined as module-level constants:
- `_TOC_LLM_SYSTEM_PROMPT` — instructs the LLM to parse OCR JSON into a TOC tree
- `_TOC_VLLM_SYSTEM_PROMPT` — instructs the vision LLM to parse page images directly

Both prompts are **language-agnostic** (not limited to Chinese academic textbooks): hierarchy is inferred from numbering depth (llm) or visual cues like font size, boldness, and indentation (vllm), with Chinese/English/roman-numeral patterns all treated as examples of the same structural rules. Front/back matter (Preface, Appendix, Bibliography, Index) is recognized as top-level entries.

Both prompts explicitly forbid inline LaTeX (`$...$`) in titles — PDF bookmarks cannot render it.  The model is instructed to use Unicode for all mathematical notation: Greek letters and math symbols (`α`, `β`, `∫`, `∑`, `∇`, `∞`, `ℏ`), superscripts (`x²`, `zⁿ`), subscripts (`x₁`, `aₙ`), and simple expressions (`w=zⁿ`, `f(z)=u+iv`).  As a safety net, `_sanitize_math_in_title` / `_sanitize_toc_tree` post-process every returned TOC tree before caching — it converts any remaining `$...$` to their closest Unicode equivalents (LaTeX commands → Unicode glyphs, `^x` → superscript chars, `_x` → subscript chars, stripping `\mathrm{}` etc.).  The mapping tables cover all common Greek letters, math operators, and superscript/subscript digits and Latin letters.

`_call_llm` always passes `extra_body={"enable_thinking": False}` — TOC extraction is a structured parsing task that does not benefit from reasoning mode, and disabling it avoids wasted latency.  It strips markdown fences from the response before JSON parsing, and logs response length + preview at INFO level.  `_build_llm_client` uses `httpx.Timeout` with the configured `llm_timeout` (default 600 s) for both connect and read phases, and sets `max_retries=0` — timeout retries are pointless.  Set `--llm_timeout` higher (e.g. `1200`) for exceptionally large documents.

### TOC tree data structure

```python
class TocNode(TypedDict):
    title: str
    page_num: int | None  # printed page number, may be inherited from children
    children: list[TocNode]
```

## GUI (gui_app.py) — Windows adaptations and known limitations

The desktop GUI works on Windows, but has known limitations that are accepted for now (fixing them is deferred):

- **CPU-only, and runs the onnxruntime engine.** The GUI always passes `device="cpu"` + `engine="onnxruntime"` (`gui_app.py`, `_process_pdf`). CPU-only because paddle 3.x's unified wheel bundles CUDA kernels — on machines with an NVIDIA driver installed, device auto-detection would pick `gpu` and try to load `cudnn64_9.dll`, which the packaged app does not ship, crashing with error code 126. The onnxruntime engine was adopted because its CPU inference is substantially faster than the paddle engine (measured ~37 s end-to-end, cold cache, for a full textbook with mobile models; the paddle engine took ~3 min 15 s for the same flow). OCR is still CPU-only, so large PDFs remain slow.
- **The UI can still hang or fail to render.** Paddle CPU inference spawns OpenMP threads that saturate all cores (paddlex defaults to 10 threads), starving the tkinter main loop. Mitigation (already applied): `cpu_threads = max(2, os.cpu_count() - 2)` plus `OMP_NUM_THREADS` / `PADDLE_PDX_CPU_NUM_THREADS` env vars, both set in the worker thread **before** the first paddle import; the value is threaded through `bookmark_pdf(..., cpu_threads=...)` to both inference models (`LayoutDetection`, `PaddleOCR` via `_engine_kwargs` in `pipeline.py`). This reduces but does not fully eliminate jank — a fully responsive UI would require moving OCR to a separate process (not done).
- **GUI uses mobile OCR models.** CPU-only means server det/rec are painfully slow (measured ~89 s per TOC page vs ~28 s for mobile, i.e. 3.2× faster on the same page). The GUI passes `ocr_model_size="mobile"` (`PP-OCRv5_mobile_det`/`PP-OCRv5_mobile_rec`), and `_MODEL_NAMES` in `gui_app.py` downloads the mobile pair. Slightly lower accuracy than server — if TOC parsing quality drops, switch back via the `ocr_model_size` param (CLI: `--ocr_model_size server`). CLI default stays `"server"`.
- **MKLDNN is irrelevant for the GUI now (onnxruntime engine).** The oneDNN executor crash (`ConvertPirAttribute2RuntimeAttribute`, paddle 3.3.1, Windows CPU) only affects the paddle engine — and the `enable_mkldnn` kwarg is a **no-op in paddlex 3.5.2 and 3.7.2** (verified by grepping the installed packages; the parameter no longer exists). The real switch is the `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT` env var, which the CLI sets under `--disable_mkldnn` (`cli.py`). The GUI still passes `enable_mkldnn=False` — harmless; it just doesn't do anything. All engine-related kwargs (`engine`, `cpu_threads`, `enable_mkldnn`) are collected into `_engine_kwargs` in `pipeline.py` and forwarded to the two model constructors (`LayoutDetection`, `PaddleOCR`).
- **PDF multi-select.** `_browse_pdf` uses `askopenfilenames`; multiple paths are joined with `os.pathsep` (`;` on Windows — NTFS filenames can't contain it) in the `pdf_path_var` StringVar, which also makes manual `;`-separated input work. `_process_pdf` validates each path (exists + `.pdf`) and processes files **serially** in a worker thread; the first failure aborts with the offending filename in the error message. `_on_done` shows `done` for one file or `done_multi` (`已完成 {n}/{total} 个文件`) for several. OCR cache is per-file-hash, so re-runs of previously processed files are fast.
- **Action-button reset on PDF change.** After a successful run the button becomes "打开输出目录" (pointing at the old document's output). A `trace_add("write")` on `pdf_path_var` resets it to "生成书签" whenever the path changes (browse or manual edit), unless processing is running.
- **Model downloads are onnx-only.** Because the GUI runs the onnxruntime engine, every model directory needs `inference.onnx` (+ `inference.yml`). `_MODEL_FILES` in `gui_app.py` downloads exactly these two files from the official `PaddlePaddle/{name}_onnx` repos (ModelScope / HuggingFace). The former Baidu tar source was removed — its archives are paddle-format only and contain no onnx. `all_models_exist` / `_download_model` check for the `inference.onnx` file (not the directory), so pre-existing paddle-format model dirs are re-downloaded automatically.
- **PyInstaller packaging collects onnxruntime.** `build_gui_pyinstaller.ps1` adds `--collect-all=onnxruntime` (the capi `.pyd`/`.dll` are loaded by path at runtime). The `--copy-metadata` package name is auto-detected at build time — `.venv-onnx` installs `onnxruntime-gpu` (dist-info `onnxruntime_gpu-*.dist-info`), `.venv` installs `onnxruntime`; PyInstaller's `copy_metadata` needs the exact metadata name, so the script probes both and passes whichever exists (paddlex's dependency check reads `importlib.metadata`). Build environment must have onnxruntime installed (1.27 recommended — 1.28's `get_available_providers()` misreports CUDA/TensorRT).
- **Build output carries the version.** The script reads `toc_forge.__version__` (`toc_forge/__init__.py`) and names every artifact `TOC-Forge-{version}` (exe / onedir / zip). Reading failure degrades to plain `TOC-Forge` with a warning. Bump the version in `__init__.py` to change the artifact name. PyInstaller's generated `TOC-Forge-{version}.spec` is gitignored (`TOC-Forge*.spec`).
- **onnxruntime-gpu inflates the package by ~1 GB.** Building with `.venv-onnx` (onnxruntime-gpu) pulls in nvidia CUDA DLLs (cublasLt 435 MB + cufft 277 MB + cublas 49 MB at the dist root) plus `onnxruntime_providers_cuda.dll` 233 MB — the zip grew from 0.9 GB to 1.4 GB — none of which the CPU-only GUI ever uses. Build with `.venv-onnx-cpu` (CPU-only onnxruntime 1.27) instead; see the ONNX section below.
- **noconsole packaging.** Both build scripts produce console-less exes; `toc_forge/__init__.py` replaces `sys.stdout`/`sys.stderr` with `StringIO` when they are `None`, so `print()` is a no-op instead of crashing.
- `.gui_settings.json` stores the API key in plaintext — it is gitignored, never commit it.

## ONNX Runtime engine (optional)

PaddleX 支持用 onnxruntime 引擎推理，避免依赖 paddle 运行时（例如 `.venv-onnx` 里
被 uv 解析到的 `paddlepaddle-gpu==2.6.2` 与 paddleocr 3.7.0 不兼容，但 ONNX 引擎
不需要 paddle 推理，目前工作正常）。用法：

```bash
source .set_onnx_env.sh          # 把 cuDNN/cuBLAS 库路径加进 LD_LIBRARY_PATH
.venv-onnx/bin/python -m toc_forge --input <pdf> ... --engine onnxruntime --device gpu
```

- **`.venv-onnx`**：专用虚拟环境（用 `uv` 管理），装了 `onnxruntime==1.27` 和
  paddleocr 相关库。必须用 **onnxruntime 1.27** —— 1.28 有 bug：
  `get_available_providers()` 不报告 CUDA/TensorRT（只报 CPU/Azure），导致 PaddleX
  的 provider 检测误判；1.27 正常。
- **`.venv-onnx-cpu`（Windows，推荐 GUI 打包构建环境）**：python 3.12 +
  **onnxruntime 1.27 CPU 版** + paddlex 3.7.2 + paddle 3.3.1。CPU 版
  onnxruntime 没有 nvidia pip 依赖，PyInstaller 打包不会收集 CUDA DLL ——
  用 `.venv-onnx`（onnxruntime-gpu）打包会膨胀 ~1 GB（见 GUI 章节）。paddle
  引擎在 paddlex 3.7.2 下无法关闭 MKLDNN（参数已移除），paddle 3.3.1 的
  oneDNN 崩溃无解，**该环境只能跑 onnxruntime 引擎**（GUI 正好是）。
- **GPU 依赖**：onnxruntime CUDA provider 需要 `libcudnn.so.9` / `libcublas.so.13`，
  由 pip 包 `nvidia-cudnn-cu13`（9.24.0.43）和 `nvidia-cublas`（13.6.1.10）提供，
  装在 `.venv-onnx` 内。`.set_onnx_env.sh` 在运行前扩展 `LD_LIBRARY_PATH` 指向
  `site-packages/nvidia/{cudnn,cublas}/lib`，否则 provider 加载失败。
- **模型来源**：每个模型目录需含 `inference.onnx`（模型文件前缀为 `inference`）。
  两个途径：① 官方发布（推荐）—— PaddlePaddle 组织在 HuggingFace / ModelScope
  上有 `{model_name}_onnx` 仓库（如 `PaddlePaddle/PP-OCRv5_server_det_onnx`），
  含 `inference.onnx` + `inference.yml`，直接下载放进模型目录即可（官方
  README 用法即 `--engine onnxruntime`）；② 本地转换 —— `paddlex
  --paddle2onnx -m <model_dir> -s <output_dir>`（需 `paddle2onnx==2.0.2rc3`，
  且要求 paddle ≥ 3.0.0.dev20250426，因此 Windows 的 `.venv-onnx`
  （paddle 2.6.2）**不能本地转换**，只能走官方仓库）。当前 `models/` 下
  7 个模型目录均已含 `inference.onnx`（server det/rec 为本地转换，layout /
  doc_ori / DocBlockLayout 及 mobile det/rec 为官方仓库下载），与 paddle
  格式并存，两种引擎共用目录。
- **实测性能**：布局检测推理 CPU 0.497 s/page → GPU 0.251 s/page。
- **显存管理（重要）**：onnxruntime 的 CUDA arena 对动态 shape（rec 文本行宽度
  每次不同）按 2 的幂扩展且不归还显存，每个模型 session 会膨胀到 2-4 GB。
  当初 `LayoutDetection` + `PaddleOCR` + `PPStructureV3` 三批 session 在同一进程
  叠加，在 6 GB 显存（如 RTX 3060 Laptop）上会撞墙——structure 阶段每页从
  ~2s 退化到 13-24s（总时间 158s vs paddle 40s，与官网 A100 上 onnxruntime
  更快的结论相反）。修复：阶段化释放（布局检测完 `del layout_model;
  gc.collect()`，OCR 用完再 `del ocr_model; gc.collect()`，onnxruntime session
  销毁会归还显存）+ 移除 PPStructureV3 后，完整 pipeline onnxruntime GPU
  冷缓存 29s < paddle GPU 40s，热缓存 ~11s。

代码侧：`--engine` 通过 `bookmark_pdf(..., engine=...)` → `_engine_kwargs` 传给两个
模型实例（`LayoutDetection`、`PaddleOCR`，PPStructureV3 已随页码扫描重构移除），
`None` 时保持 PaddleX 默认。`_engine_kwargs` 还承载 `cpu_threads` 与
`enable_mkldnn`（见 GUI 章节）。

## ONNX Runtime engine — Windows 差异

- **GPU 不可用，静默降级 CPU**：Windows 的 `.venv-onnx` 未装 `nvidia-cudnn-cu13`
  / `nvidia-cublas`（Linux 环境才有，见上节），onnxruntime 创建
  `CUDAExecutionProvider` 失败（日志：`Failed to create CUDAExecutionProvider.
  Require cuDNN 9.* and CUDA 13.*`）后自动回退 CPU，pipeline 照常完成。
  实测 Windows CPU 全流程 ~3min15s（无缓存）。要启用 Windows GPU 需 pip 装
  上述两个包 + 运行时 `os.add_dll_directory()`（打包场景还需 PyInstaller
  显式收集 nvidia DLL，见 gui_app 打包注意事项）。
- **`.venv-onnx`（Windows）**：paddleocr 3.7.0 + paddlex 3.7.2 +
  onnxruntime-gpu 1.27（1.27 的要求同 Linux）+ paddlepaddle-gpu 2.6.2（与
  paddleocr 3.7 不兼容，只能走 onnxruntime 引擎）。onnxruntime-gpu 会把
  CUDA/nvidia DLL 拉进 PyInstaller 包（~1 GB 纯浪费，见 GUI 章节），**打包
  请用 `.venv-onnx-cpu`** 而不是它。
- **`.venv-onnx-cpu`（Windows）**：python 3.12 + onnxruntime 1.27 **CPU 版**
  + paddlex 3.7.2 + paddle 3.3.1。**GUI 打包推荐构建环境**：CPU 版无
  nvidia 依赖，包体积正常；paddle 引擎因 MKLDNN 无法关闭不可用（3.7.2 已
  移除开关），只能跑 onnxruntime 引擎（GUI 正好是）。
- **`.venv`（Windows）**：paddle 3.3.1 + paddleocr 3.5.0。要跑 onnxruntime
  引擎（CLI `--engine onnxruntime` 或 GUI）需先 `pip install onnxruntime`
  （1.27）；onnxruntime 引擎不依赖 paddle 推理，与 paddle 引擎共存于同一
  环境没问题。GUI 若坚持在 `.venv` 里跑，装上即可。
- **常见报错**：`ValueError: No valid model files were found for engine
  'onnxruntime'.` —— 模型目录缺 `inference.onnx`（布局/方向/区域模型常被
  漏掉），从 ModelScope `PaddlePaddle/{name}_onnx` 下载补齐即可（layout
  ~124 MB、doc_ori ~6.5 MB、DocBlockLayout ~123 MB）。

## Logging

Logs go to `log/toc_forge.log` via `setup_logger`. Uses `logging.DEBUG` level, file-only (no console handler). LLM calls also log via `logger.info` / `logger.warning`.

Stage timing is logged at DEBUG level in `ocr_engine.py` (per stage, covering both
cached and inference paths — `layout detection cost: Xs (N pages, cached|inference)`,
`OCR toc pages cost: Xs (N pages)`, `OCR number pages cost: Xs (N pages)`).
