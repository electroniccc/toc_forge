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
| `gui_app.py` | Desktop GUI (Tkinter + sv_ttk): strategy selection, model download (3 sources), settings persistence; runs `bookmark_pdf` in a worker thread |
| `build_gui.ps1` / `build_gui_pyinstaller.ps1` | Nuitka / PyInstaller packaging scripts for the GUI (console-less exe) |

### OCR models used

The pipeline downloads/caches five PaddleOCR models via `make_sure_model_exists()`:

| Model | Stage |
|---|---|
| `PP-DocLayout_plus-L` | Layout detection |
| `PP-LCNet_x1_0_doc_ori` | Document orientation classification |
| `PP-OCRv5_server_det` | Text detection |
| `PP-OCRv5_server_rec` | Text recognition |
| `PP-DocBlockLayout` | Region detection — no longer used by the pipeline (was for the removed PPStructureV3 page-number scan); files may still exist in `models/` |

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

- **CPU-only, and slow.** The GUI always passes `device="cpu"` (`gui_app.py`, `_process_pdf`): paddle 3.x's unified wheel bundles CUDA kernels, so on machines with an NVIDIA driver installed, device auto-detection picks `gpu` and tries to load `cudnn64_9.dll` — which the packaged app does not ship, crashing with error code 126. Consequence: OCR runs on CPU only, which is slow for large PDFs.
- **The UI can still hang or fail to render.** Paddle CPU inference spawns OpenMP threads that saturate all cores (paddlex defaults to 10 threads), starving the tkinter main loop. Mitigation (already applied): `cpu_threads = max(2, os.cpu_count() - 2)` plus `OMP_NUM_THREADS` / `PADDLE_PDX_CPU_NUM_THREADS` env vars, both set in the worker thread **before** the first paddle import; the value is threaded through `bookmark_pdf(..., cpu_threads=...)` to both inference models (`LayoutDetection`, `PaddleOCR` via `_thread_kwargs` in `pipeline.py`). This reduces but does not fully eliminate jank — a fully responsive UI would require moving OCR to a separate process (not done).
- **MKLDNN is enabled by default again.** `enable_mkldnn=False` was previously hard-coded in `pipeline.py` (paddle 3.3.1's oneDNN executor crashes on Windows with "ConvertPirAttribute2RuntimeAttribute does not support ArrayAttribute<DoubleAttribute>") but was removed so Linux runs MKLDNN normally. If Windows CPU inference crashes again, re-disable MKLDNN (ideally behind a flag).
- **noconsole packaging.** Both build scripts produce console-less exes; `toc_forge/__init__.py` replaces `sys.stdout`/`sys.stderr` with `StringIO` when they are `None`, so `print()` is a no-op instead of crashing.
- `.gui_settings.json` stores the API key in plaintext — it is gitignored, never commit it.

## ONNX Runtime engine (optional, Linux)

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
- **GPU 依赖**：onnxruntime CUDA provider 需要 `libcudnn.so.9` / `libcublas.so.13`，
  由 pip 包 `nvidia-cudnn-cu13`（9.24.0.43）和 `nvidia-cublas`（13.6.1.10）提供，
  装在 `.venv-onnx` 内。`.set_onnx_env.sh` 在运行前扩展 `LD_LIBRARY_PATH` 指向
  `site-packages/nvidia/{cudnn,cublas}/lib`，否则 provider 加载失败。
- **模型转换**：每个模型目录需含 `inference.onnx`（模型文件前缀为 `inference`）。
  用 `paddlex --paddle2onnx -m <model_dir> -s <output_dir>` 转换（需
  `paddle2onnx==2.0.2rc3`）。当前 `models/` 下 5 个模型均已转换
  （PP-DocLayout_plus-L、PP-LCNet_x1_0_doc_ori、PP-OCRv5_server_det、
  PP-OCRv5_server_rec、PP-DocBlockLayout）。
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

代码侧：`--engine` 通过 `bookmark_pdf(..., engine=...)` → `_engine_kwargs` 传给全部
4 个模型实例（`LayoutDetection`、`PaddleOCR`×2、`PPStructureV3`），`None` 时保持
PaddleX 默认。

## Logging

Logs go to `log/toc_forge.log` via `setup_logger`. Uses `logging.DEBUG` level, file-only (no console handler). LLM calls also log via `logger.info` / `logger.warning`.
