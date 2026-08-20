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
- `--toc_detect_max_page`: max pages to scan for TOC detection (default: `25*3=75`, also capped by the total page count). See "Layout detection" below for the batched scan behavior.

Output: `{output}/{input_stem}_bookmarked.pdf` with injected PDF outline.

## Architecture

The pipeline has five stages (`bookmark_pdf` in `toc_forge/pipeline.py:118`):

1. **Page image extraction + batched TOC scan** (`scan_toc_page_range` / `layout_pages_in_range`, both in `ocr_engine.py`): pages are rendered to numpy arrays via PyMuPDF at 2x zoom (falling back to 1x if the image exceeds 2000px in either dimension) and layout-detected **in batches of 25**, up to `min(toc_detect_max_page, page_count)` (default `25*3=75`). Once a TOC page (a layout `content` box) is seen in a batch, the batch size shrinks to **10** and scanning continues until **3 consecutive non-TOC pages**; if no TOC page is ever seen, scanning runs to the cap. Per-page layout results are cached under the `layout` stage, so the follow-up `layout_pages_in_range(0, scan_end)` pass over the scanned range is a cache hit. **Page-number sampling**: after the TOC pages are finalized (incl. the keyword/continuity supplements below), `bookmark_pdf` renders + layout-detects up to **20 more pages after the last TOC page** (the sampling range) and appends their `number` boxes to the number-box set used for the page-offset computation; the layout model is kept alive until this extension is done, then released.

2. **Layout detection** (`extract_toc_and_number_pages` on the scanned range; `get_toc_pages` remains as a wrapper for callers with pre-rendered images): runs `PP-DocLayout_plus-L` on the page images to find "content" boxes (TOC blocks) and "number" boxes (page numbers). On pages that have `content` boxes, `paragraph_title` boxes sitting above each content box are also collected — these are often chapter/part headings (e.g. "第一篇", "第1章") that the layout model didn't label as "content". Deduplicates overlapping boxes via `deduplicate_content_boxes`. Returns the raw per-page boxes as a third value so the keyword supplement can reuse them. **TOC page supplement + denoise** (in `bookmark_pdf`, after the OCR model is created): `detect_toc_pages_by_keyword` recovers TOC pages the layout model missed — a page whose upper area has a `paragraph_title`, or whose top-left/top-right corner has a `header`, OCR-reading "Contents"/"目录", is added as a TOC page even with no `content` box (English textbooks' running "CONTENTS" headers). Such pages get a synthetic full-page content box (`_synthetic_content_box`) that excludes the running-header band (top 8%), footer boxes (30px margin — `filter_toc_result`'s 20px tolerance otherwise lets the bottom copyright line through), and the matched keyword boxes themselves, so "CONTENTS v" header lines never become bogus entries. Candidate-box OCR is tiny and upscaled 2x before recognition (pad must be ≥12px or small header glyphs OCR as garbage, e.g. "CONTENTS" → "SSNESEEE"), cached as `toc_keyword` per page. **Continuity propagation** (`detect_toc_pages_by_continuity` / `_is_toc_style_page`): English TOCs often span several pages but only the first carries a "Contents" heading (verified on OpenStax Calculus 1-3 — the layout model found no `content` box on *any* TOC page, and the keyword supplement matched only the heading page). Each page following a confirmed TOC page is checked for "TOC style" — most text lines end with a page number (e.g. "4.10 Antiderivatives 419") — via full-page OCR (cached as `toc_continuity` per page, header band top 8% and area below the first footer box excluded); passing pages are added with a synthetic content box and propagation continues, the first non-TOC-style page (e.g. the Preface body) stops the run. Finally `keep_longest_contiguous_pages` keeps only the longest run of consecutive TOC page indices (e.g. [5, 7, 8, 9] → [7, 8, 9] — 5 was a stray), which also re-sorts the list into ascending page order (the supplement appends out of order and the page-tree merge expects ascending order).

3. **OCR on TOC pages** (`ocr_toc_pages`): runs PaddleOCR on each detected TOC page, then filters results to only keep text inside the content boxes (`filter_toc_result`).

4. **TOC tree reconstruction** — three strategies exist, auto-selected in `cli.py:main()`:
   - **`local_ocr` (default)**: uses gap-clustering + indentation parsing (`reconstruct_toc1`). No API needed.
   - **`llm`**: runs local OCR first, then sends OCR JSON to a text LLM (`build_toc_llm`). Selected when `--api_base_url`, `--api_key`, and `--llm_name` are set but `--vllm_name` is not.
   - **`vllm`**: sends TOC page images directly to a vision LLM, skipping local OCR (`build_toc_vllm`). Selected when all three of `--api_base_url`, `--api_key`, and `--vllm_name` are set.

   For `local_ocr`, one heuristic parser exists:
   - `reconstruct_toc1` (**the active/default strategy**): per-page level detection + cross-page merge. Level assignment is **geometry-driven and language-agnostic**:
     1. **Gap-tree clustering**: x-coordinate gaps determine baseline indentation levels for all entries on a page.
     2. **Paragraph-title depth**: consecutive `paragraph_title` entries above a content box form a nesting chain (first → level 0, second → level 1, …). Content entries are shifted by the number of consecutive paragraph_titles above them.
     3. **Semantic floors** (supplementary): chapter-like patterns (e.g. `第X章`) are forced to level 0; section/subsection patterns set minimum levels — applied only to entries from `content` boxes, never to `paragraph_title` entries.
     4. **KMeans fallback**: if gap-tree produces a single bin but x-variance is large, force a 2-cluster split.
   - Per-page mini-trees are built via `_build_tree` (same-level entries are siblings, only strictly-deeper entries nest).
   - `_merge_page_trees` merges per-page trees: recognizes section-like entries (sections, exercises, numbered entries) and attaches them under the last chapter, with exercises matched to their parent section by number. `_is_chapter_like` accepts English chapter entries too (module-level `_EN_CHAPTER_PAT`: "1. Introduction" (Kibble style, digit+dot), "1 Functions and Graphs", "Chapter 1", "Part I", "A Table of Integrals", "Preface", "Answer Key"…) and per-chapter English back matter ("Chapter Review", "Key Terms"…) plus letter-numbered appendix sections ("A.2") are section-like — without these, continuation pages of English TOCs left "4.10 Antiderivatives"/"Chapter Review" at the root, and "1. Introduction" got nested under front matter ("List of Symbols").
   - `_assign_levels` semantic floors use the number-led subset of `_EN_CHAPTER_PAT` (`_EN_CHAPTER_FLOOR`) as a level-0 floor — otherwise "1. Introduction" is raised to level ≥1 by the `^\d+\s*\S` section floor and nests under a level-0 front-matter entry. Front-matter words (Preface, Introduction…) are deliberately excluded from the floor: OpenStax's "Introduction 7" is a per-chapter subheading that must stay nested.
   - Structural repair preserves OCR/source order until missing page numbers have been inherited. `restore_toc_order` then repairs locally inverted child lists only when every sibling has a resolvable page number in the same numbering system; equal-page entries remain stable, while root lists, unresolved siblings, and mixed Roman/Arabic systems are left untouched.
   - `inherit_page_numbers` fills **only** `None` page numbers (common for `paragraph_title` entries) by inheriting from descendants (max depth 3) or the next sibling — string page numbers ("VII", "I-1") are meaningful and are never overwritten.
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
   - Parses each line into (title, page_num) by detecting page number fragments at the rightmost end — handles plain digits, Roman numerals, parenthesized digits, trailed dots, dot-leader patterns (e.g. `"…………6"`), "X-n" Roman-chapter numbers, and English "Title N" trailing numbers. Page numbers are only accepted on the **rightmost** fragment of a line — a leading standalone number ("1 Functions and Graphs 7" starts with "1") is a chapter number, not a page number, and stays in the title. When most rightmost fragments of a page end in a 1-3 digit number (English TOCs without dot leaders, e.g. OpenStax), a trailing-number mode splits the page number off even when it shares the OCR fragment with the title ("Preface 1", "1.1 Review of Functions 8") or is glued to it ("The Limit Laws140" — title part must end in a letter/CJK char so "1.1" numbering is not split), and the gap check is relaxed for rightmost fragments (OpenStax page-number OCR boxes often overlap the title box horizontally, e.g. "Limits" + "105").
   - Strips leading bullet markers (`*`, `•`, `·`) from titles

5. **Page offset + bookmark injection** (`get_page_offset2`, `add_bookmarks_to_pdf`): computes the offset between printed-page and PDF-page indexing **without PPStructureV3**. Layout detection already locates "number" blocks (printed page numbers) per page; `get_page_offset2` filters them by parity-group position consistency (odd pages' number boxes should mostly overlap, even pages likewise — accidental boxes are dropped), crops each kept page around its number box, OCRs just the crop with the existing `PaddleOCR` model, parses the printed number, and takes the mode of `pdf_page_idx - page_num`. Unresolved or out-of-range entries are skipped (resolvable children are promoted) rather than silently targeting PDF page 1; an entirely empty/unresolvable outline raises `EmptyTocError`. The output PDF is written to a same-directory temporary file and atomically replaces the destination. The page-number logic is split into three independently-testable stages: `get_number_box_pages` (collect + parity-consistency filter), `ocr_number_boxes` (crop + OCR + cache), `compute_page_offset` (parse + mode), with `get_page_offset2` as the wrapper. `bookmark_pdf` calls the three stages explicitly (not the wrapper) so the number OCR results are available for the Roman page-map check below. **Roman-numeral page map** (module `toc_forge/page_map.py`): some books number body pages "I-1", "I-2", …, "II-1", … (chapter Roman numeral + dash + within-chapter Arabic number; e.g. Morin's "Introduction to Classical Mechanics"). When `detect_roman_arabic_format` finds ≥60% of number-box OCR texts in that form, `build_page_map` scans body pages from right after the TOC (rendering pages on demand, OCR'ing only the number-box crop regions derived by clustering the layout-detected number-box positions — top/bottom × left/right bands, median centers — so odd/even mirroring and chapter-title-page bottom centers both work; per-page OCR cached as `page_map_ocr`) and builds {"I-1": 1, "I-2": 2, …, "II-1": <ch-I length + 1>, …}: the first matching page is the anchor (cumulative 1), a page whose chapter changes with n==1 starts a new chapter (cumulative = physical distance from anchor + 1), later pages of the same chapter map to chapter_start + (n-1) so single-page OCR misreads cannot corrupt the rest; a chapter change with n≠1 does not start a chapter immediately — the page's number infers a chapter start (idx − (n−1)) and a second page of the same chapter with a consistent inference confirms it (handles chapter-start pages whose OCR failed, e.g. Morin's XIV-1); when a chapter ends, all page numbers 1..max_n seen in it are backfilled so bookmarks pointing at pages whose OCR failed still resolve. Parsing tolerates OCR noise: trailing-search match (a crop glued to a word, e.g. "…NUMERICALLYXIV-15"), underscores/spaces around the dash, lowercase "l" for "I". Scanning stops after 40 consecutive pages without a matching number. `map_page_num` converts a bookmark's "X-n" page number through the map before the offset is applied; without the map (format absent or key missing) the old handling stays. `_parse_toc_lines` in parsing.py also recognizes trailing "X-n" fragments in TOC lines and keeps them as string page numbers (e.g. "1.1 Balancing forces" + "I-1"), which the map then converts. **Front-matter offset** (`build_front_matter_offset` in page_map.py): books whose front matter numbers itself in Roman numerals ("vii") use a separate numbering system from the Arabic body pages — e.g. Kibble: body offset 20, front matter "vii" printed on PDF page 7 (offset 0). When the TOC tree contains a Roman string page number, `bookmark_pdf` scans the pages between the TOC end and the first body page (printed 1 sits at `page_offset + 1`), OCR'ing front-matter number positions (top corners + bottom-center band — these differ from body-page positions, which is why the layout number-box flow drops them; cached as `front_matter_ocr`), and `_page_num_to_pdf` converts Roman page numbers with `front_offset` instead of the body offset. **Segmented offset** (page_map.py): some books' printed page numbers are not a single linear function of the PDF index (Shankar's "Principles of Quantum Mechanics" runs offset 13 for pages 1-73, offset 12 from 76 on, and drifts further — the book skips printed numbers at chapter boundaries, and the Answers/Index section shifts again; a single mode offset lands chapter bookmarks 1-2 pages late). `detect_segmented_offset` samples every root entry's int page number: it estimates the PDF page (`X + page_offset`), reads the printed number actually printed there (text layer first — nearly free — else the number-box OCR crops), and any mismatch triggers a full scan. `build_arabic_page_map` then scans every page (text layer preferred, `page_map_ocr`-cached OCR fallback) and segments runs of equal offset into `[{printed_start, pdf_start, printed_end}, ...]`, dropping single-page segments (isolated OCR misreads); `map_arabic_page` maps a printed number inside a segment linearly and resolves numbers in gaps between segments (printed numbers the book skips, e.g. Shankar's 74/75) from whichever segment end is closer. `_page_num_to_pdf` consults the segment table before the plain-offset formula. Single-offset books cost only the sampling reads.

### Module organization

| File | Responsibility |
|---|---|
| `toc_forge/__init__.py` | Package version, `sys.stdout` encoding setup, re-exports `bookmark_pdf` and `main` |
| `toc_forge/cli.py` | CLI entry point: argparse config, strategy auto-detection, calls `bookmark_pdf` |
| `toc_forge/pipeline.py` | Top-level orchestration: `bookmark_pdf`, `build_toc_local_ocr`, `add_bookmarks_to_pdf`, `get_page_offset` |
| `toc_forge/ocr_engine.py` | PaddleOCR calls: `scan_toc_page_range` (batched TOC-page scan) / `layout_pages_in_range` (render + layout-detect a page range with per-page caching) / `extract_toc_and_number_pages` (layout) / `get_toc_pages` (layout wrapper for pre-rendered images), `ocr_toc_pages` (text), `detect_toc_pages_by_keyword` / `_synthetic_content_box` / `detect_toc_pages_by_continuity` / `_is_toc_style_page` / `keep_longest_contiguous_pages` (layout-missed TOC page supplement + continuity propagation + denoise), `get_number_box_pages` / `ocr_number_boxes` / `compute_page_offset` (page-offset stages, wrapped by `get_page_offset2`) |
| `toc_forge/parsing.py` | Heuristic TOC tree reconstruction: `_parse_toc_lines`, `_build_tree`, `reconstruct_toc1`, `_merge_page_trees`, `_merge_content_box_trees`, `repair_toc_tree`, `_fix_pian_structure`, `_fix_zhang_sections`, `inherit_page_numbers`, `restore_toc_order` |
| `toc_forge/page_map.py` | Page-number mapping: `detect_roman_arabic_format`, `_page_number_regions`, `_ocr_page_numbers`, `build_page_map`, `map_page_num`, `_ocr_front_matter_page_numbers`, `build_front_matter_offset`, `_text_layer_page_number`, `detect_segmented_offset`, `build_arabic_page_map`, `map_arabic_page` — detects "X-n" (Roman chapter + Arabic within-chapter) body page numbering and builds the {"I-1": 1, …} cumulative map; computes the separate front-matter offset for Roman-numeral front matter ("vii"); samples chapter starts and (if the offset is segmented) scans the whole book into a segment table for books whose printed page numbers skip (Shankar) |
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

Layout, OCR, and structure results are cached per-PDF and per-page directly under `--cache_dir/{pdf_hash}/`. Cache keys are `{stage}_page_{idx}.json` (or `{stage}.json` for non-per-page results). OCR model size, engine and device are deliberately not part of the cache path because normal usage does not switch those configurations for the same cache. Writes use a same-directory temporary file plus `os.replace`, so interrupted writes cannot leave a partial JSON at the final cache path. Caching uses `_cache_load` / `_cache_save` with the `CachedResult` wrapper class that mimics PaddleX result objects. Legacy wrapped format `{"res": {...}}` is unwrapped via `_unwrap_legacy_cache`.

The final TOC tree produced by the `llm`/`vllm` strategies is cached in the same PDF directory as `toc_tree_llm.json` / `toc_tree_vllm.json`. Repeat runs skip the LLM call regardless of model or endpoint changes; callers use `--no_toc_cache` when they intentionally want to refresh the result. The fresh result overwrites the fixed cache file. Handled by `_load_toc_tree_cache` / `_save_toc_tree_cache` in `toc_forge/llm.py`.

### LLM integration

LLM strategies use the `openai` SDK via `_build_llm_client()` and `_call_llm()`. Both system prompts are defined as module-level constants:
- `_TOC_LLM_SYSTEM_PROMPT` — instructs the LLM to parse OCR JSON into a TOC tree
- `_TOC_VLLM_SYSTEM_PROMPT` — instructs the vision LLM to parse page images directly

`build_toc_llm` and `build_toc_vllm` require non-empty `llm_model` and `llm_base_url` arguments. They do not infer either value from environment variables; configuration must be resolved by the caller before crossing this interface.

Both prompts are **language-agnostic** (not limited to Chinese academic textbooks): hierarchy is inferred from numbering depth (llm) or visual cues like font size, boldness, and indentation (vllm), with Chinese/English/roman-numeral patterns all treated as examples of the same structural rules. Front/back matter (Preface, Appendix, Bibliography, Index) is recognized as top-level entries.

Both prompts explicitly forbid inline LaTeX (`$...$`) in titles — PDF bookmarks cannot render it.  The model is instructed to use Unicode for all mathematical notation: Greek letters and math symbols (`α`, `β`, `∫`, `∑`, `∇`, `∞`, `ℏ`), superscripts (`x²`, `zⁿ`), subscripts (`x₁`, `aₙ`), and simple expressions (`w=zⁿ`, `f(z)=u+iv`).  As a safety net, `_sanitize_math_in_title` / `_sanitize_toc_tree` post-process every returned TOC tree before caching — it converts any remaining `$...$` to their closest Unicode equivalents (LaTeX commands → Unicode glyphs, `^x` → superscript chars, `_x` → subscript chars, stripping `\mathrm{}` etc.).  The mapping tables cover all common Greek letters, math operators, and superscript/subscript digits and Latin letters.

`_call_llm` always passes `extra_body={"enable_thinking": False}` — TOC extraction is a structured parsing task that does not benefit from reasoning mode, and disabling it avoids wasted latency.  It strips markdown fences from the response before JSON parsing, and logs response length + preview at INFO level.  `_build_llm_client` uses `httpx.Timeout` with the configured `llm_timeout` (default 600 s) for both connect and read phases, and sets `max_retries=0` — timeout retries are pointless.  Set `--llm_timeout` higher (e.g. `1200`) for exceptionally large documents.

### TOC tree data structure

```python
class TocNode(TypedDict):
    title: str
    page_num: int | str | None  # Arabic, Roman/front-matter, X-n, or missing
    children: list[TocNode]
```

Expected extraction failures use the top-level `TocForgeError` hierarchy. `TocNotFoundError` means no TOC pages were detected; `EmptyTocError` means parsing/writing produced no resolvable entries. The CLI exits with status 2 for these failures instead of printing a blank success path.

## GUI (gui_app.py) — Windows adaptations and known limitations

The desktop GUI works on Windows, but has known limitations that are accepted for now (fixing them is deferred):

- **CPU-only, and runs the onnxruntime engine.** The GUI always passes `device="cpu"` + `engine="onnxruntime"` (`gui_app.py`, `_process_pdf`). CPU-only because paddle 3.x's unified wheel bundles CUDA kernels — on machines with an NVIDIA driver installed, device auto-detection would pick `gpu` and try to load `cudnn64_9.dll`, which the packaged app does not ship, crashing with error code 126. The onnxruntime engine was adopted because its CPU inference is substantially faster than the paddle engine (measured ~37 s end-to-end, cold cache, for a full textbook with mobile models; the paddle engine took ~3 min 15 s for the same flow). OCR is still CPU-only, so large PDFs remain slow.
- **The UI can still hang or fail to render.** Paddle CPU inference spawns OpenMP threads that saturate all cores (paddlex defaults to 10 threads), starving the tkinter main loop. Mitigation (already applied): `cpu_threads = max(2, os.cpu_count() - 2)` plus `OMP_NUM_THREADS` / `PADDLE_PDX_CPU_NUM_THREADS` env vars, both set in the worker thread **before** the first paddle import; the value is threaded through `bookmark_pdf(..., cpu_threads=...)` to both inference models (`LayoutDetection`, `PaddleOCR` via `_engine_kwargs` in `pipeline.py`). This reduces but does not fully eliminate jank — a fully responsive UI would require moving OCR to a separate process (not done).
- **GUI uses mobile OCR models.** CPU-only means server det/rec are painfully slow (measured ~89 s per TOC page vs ~28 s for mobile, i.e. 3.2× faster on the same page). The GUI passes `ocr_model_size="mobile"` (`PP-OCRv5_mobile_det`/`PP-OCRv5_mobile_rec`), and `_MODEL_NAMES` in `gui_app.py` downloads the mobile pair. Slightly lower accuracy than server — if TOC parsing quality drops, switch back via the `ocr_model_size` param (CLI: `--ocr_model_size server`). CLI default stays `"server"`.
- **MKLDNN is irrelevant for the GUI now (onnxruntime engine).** The oneDNN executor crash (`ConvertPirAttribute2RuntimeAttribute`, paddle 3.3.1, Windows CPU) only affects the paddle engine — and the `enable_mkldnn` kwarg is a **no-op in paddlex 3.5.2 and 3.7.2** (verified by grepping the installed packages; the parameter no longer exists). The real switch is the `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT` env var, which the CLI sets under `--disable_mkldnn` (`cli.py`). The GUI still passes `enable_mkldnn=False` — harmless; it just doesn't do anything. All engine-related kwargs (`engine`, `cpu_threads`, `enable_mkldnn`) are collected into `_engine_kwargs` in `pipeline.py` and forwarded to the two model constructors (`LayoutDetection`, `PaddleOCR`).
- **PDF multi-select.** `_browse_pdf` uses `askopenfilenames`; multiple paths are joined with `os.pathsep` (`;` on Windows — NTFS filenames can't contain it) in the `pdf_path_var` StringVar, which also makes manual `;`-separated input work. `_process_pdf` validates each path (exists + `.pdf`) and processes files **serially** in a worker thread; the first failure aborts with the offending filename in the error message. `_on_done` shows `done` for one file or `done_multi` (`已完成 {n}/{total} 个文件`) for several. Inputs with the same stem receive deterministic path-hash suffixes, so files from different directories cannot overwrite each other. OCR cache is per-file hash, so re-runs are fast.
- **Action-button reset on PDF change.** After a successful run the button becomes "打开输出目录" (pointing at the old document's output). A `trace_add("write")` on `pdf_path_var` resets it to "生成书签" whenever the path changes (browse or manual edit), unless processing is running.
- **Model downloads are onnx-only and atomic.** Because the GUI runs the onnxruntime engine, every model directory needs non-trivial `inference.onnx` and `inference.yml` files. `_MODEL_FILES` in `gui_app.py` downloads exactly these two files from the official `PaddlePaddle/{name}_onnx` repos (ModelScope / HuggingFace). Each file is written to `.part`, flushed, length-checked and atomically installed; an interrupted download cannot replace an existing model. `all_models_exist` validates both files and their minimum sizes, so partial or paddle-only directories are re-downloaded.
- **PyInstaller packaging collects onnxruntime.** `build_gui_pyinstaller.ps1` adds `--collect-all=onnxruntime` (the capi `.pyd`/`.dll` are loaded by path at runtime). The `--copy-metadata` package name is auto-detected at build time — `.venv-onnx` installs `onnxruntime-gpu` (dist-info `onnxruntime_gpu-*.dist-info`), `.venv` installs `onnxruntime`; PyInstaller's `copy_metadata` needs the exact metadata name, so the script probes both and passes whichever exists (paddlex's dependency check reads `importlib.metadata`). Build environment must have onnxruntime installed (1.27 recommended — 1.28's `get_available_providers()` misreports CUDA/TensorRT).
- **Build output carries the version.** The script reads `toc_forge.__version__` (`toc_forge/__init__.py`) and names every artifact `TOC-Forge-{version}` (exe / onedir / zip). Reading failure degrades to plain `TOC-Forge` with a warning. Bump the version in `__init__.py` to change the artifact name. PyInstaller's generated `TOC-Forge-{version}.spec` is gitignored (`TOC-Forge*.spec`).
- **onnxruntime-gpu inflates the package by ~1 GB.** Building with `.venv-onnx` (onnxruntime-gpu) pulls in nvidia CUDA DLLs (cublasLt 435 MB + cufft 277 MB + cublas 49 MB at the dist root) plus `onnxruntime_providers_cuda.dll` 233 MB — the zip grew from 0.9 GB to 1.4 GB — none of which the CPU-only GUI ever uses. Build with `.venv-onnx-cpu` (CPU-only onnxruntime 1.27) instead; see the ONNX section below.
- **noconsole packaging.** Both build scripts produce console-less exes; `toc_forge/__init__.py` replaces `sys.stdout`/`sys.stderr` with `StringIO` when they are `None`, so `print()` is a no-op instead of crashing.
- GUI settings live in the platform user configuration directory (`%APPDATA%/toc-forge` on Windows, `$XDG_CONFIG_HOME/toc-forge` on Linux) and are written atomically. A legacy `.gui_settings.json` next to the app is still read for migration. The API key remains plaintext in that per-user file.
- **TOC max-scan-pages setting + versioned window title.** The Input/Output group has a collapsible **"▶ 高级设置 / Advanced settings"** button (collapsed by default) that reveals a "目录最大探测页数 / Max TOC scan pages" Spinbox (`toc_detect_max_page`). It is **pre-filled with 75** (the `bookmark_pdf` default) — an empty Spinbox would jump to 1 when the user clicks its increment/decrement arrows, so empty / non-numeric / ≤0 persisted values all fall back to "75" in `_load_settings_to_ui`. It persists with the settings and is passed through as `toc_detect_max_page=...` in `_process_pdf`. The window title is `TOC Forge v{__version__}` — the version is read by parsing `toc_forge/__init__.py` (`_app_version` in `gui_app.py`), which neither triggers the paddleocr import on the UI thread nor goes stale when the editable dist-info metadata lags behind `pyproject.toml` (PyInstaller's `--collect-all=toc_forge` ships the source file, so the packaged exe resolves it too).
- **Startup update check (GitHub, silent).** ~1.5 s after launch a daemon thread queries `https://api.github.com/repos/electroniccc/toc_forge/releases/latest` (`_latest_release_tag` in `gui_app.py`, 8 s timeout); if the release `tag_name` is newer than `_app_version()` (compared via `_version_tuple`), a `messagebox.showinfo` "发现新版本 / Update available" prompt appears that the user dismisses with OK. **Any failure is silent** — network error, non-200, missing tag, or equal/older version all return None and nothing is shown (GitHub being unreachable in China must not produce any error or prompt). The check is triggered by `root.after(1500, self._check_for_update)` in `TocForgeApp.__init__`, and `_show_update_prompt` marshals the result to the UI thread via `root.after(0, ...)`.

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
