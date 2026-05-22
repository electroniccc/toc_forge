# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`toc_forge` extracts a table of contents from a PDF using PaddleOCR layout/OCR models and injects the resulting TOC tree back into the PDF as navigable bookmarks (outline). Everything lives in a single file: `toc_forge.py`.

## Environment setup

Source the env script before running:
```powershell
.\.set_paddlex_env.ps1
```
This sets `PADDLE_HUB_HOME`, `PADDLE_PDX_CACHE_HOME`, and disables model source checks. The script checks for `PADDLE_PDX_CACHE_HOME` at startup and refuses to run without it.

The project uses a local `.venv`. Install dependencies with pip (no `requirements.txt` exists — dependencies are what's installed in `.venv`). Key packages: `paddleocr`, `paddlex`, `opencv-python` (cv2), `PyMuPDF` (fitz), `Pillow`, `numpy`, `scikit-learn`.

## Running

```powershell
python toc_forge.py --input <pdf_path> --output <output_dir> [--model_dir ./models] [--debug]
```

- `--input`: path to the source PDF
- `--output`: output directory (default: `output directory`)
- `--model_dir`: directory containing PaddleOCR models (default: `./models`); looks for `PP-DocLayout_plus-L` subdirectory
- `--log_dir`: log directory (default: `log`)
- `--debug`: saves intermediate layout/OCR/parsing results as images and JSON

Output: `{output}/{input_stem}_bookmarked.pdf` with injected PDF outline.

## Architecture

The pipeline has five stages (`bookmark_pdf` in `toc_forge.py:965`):

1. **Page image extraction** (`image_from_page`): renders the first 30 PDF pages to numpy arrays via PyMuPDF at 2x zoom, falling back to 1x if the image exceeds 2000px in either dimension.

2. **Layout detection** (`get_toc_pages`): runs `PP-DocLayout_plus-L` on the page images to find "content" boxes (TOC blocks) and "number" boxes (page numbers). Deduplicates overlapping boxes via `deduplicate_content_boxes`.

3. **OCR on TOC pages** (`ocr_toc_pages`, `build_toc_local_ocr`): runs PaddleOCR on each detected TOC page, then filters results to only keep text inside the content boxes (`filter_toc_result`).

4. **TOC tree reconstruction** — the core parsing logic. Three strategies exist:
   - `reconstruct_toc` (semantic): uses regex patterns to detect chapter/section/subsection levels, with KMeans fallback on x-coordinates for unclassified entries.
   - `reconstruct_toc_indent`: pure indentation clustering — splits x-coordinates into boundaries based on gap threshold, then builds a tree.
   - `reconstruct_toc1` (**the active/default strategy**): hybrid — runs sematic + gap-clustering level detection *per page* to handle x-coordinate shifts between pages, then merges per-page mini-trees with `_merge_page_trees` (which re-parents section-like entries under preceding chapters). Also handles multi-column pages by processing each content box independently then merging left-to-right by column (`_merge_content_box_trees`).
   - `repair_toc_tree` post-processes to fix misplaced entries and sort children by section number.

   The shared parsing pipeline (`_parse_toc_lines`):
   - Flattens OCR text boxes across pages with cumulative y-offsets
   - Groups items into lines by y-overlap
   - Parses each line into (title, page_num) by detecting page number fragments at the rightmost end — handles plain digits, Roman numerals, parenthesized digits, trailed dots, and dot-leader patterns.

5. **Page offset + bookmark injection** (`get_page_offset`, `add_bookmarks_to_pdf`): uses `PPStructureV3` to OCR a few pages *after* the last TOC page, detecting printed page numbers to calculate the offset between printed-page and PDF-page indexing. Then calls `doc.set_toc()` + `doc.save()` to write the PDF outline.

### TOC tree data structure

```python
class TocNode(TypedDict):
    title: str
    page_num: int | None  # printed page number, may be inherited from children
    children: list[TocNode]
```

## Stubs and unfinished work

- `build_toc_llm()` and `build_toc_vllm()` are empty stubs (lines 900-911) — future plans for LLM-based TOC extraction.
- Only `reconstruct_toc1` is active in `build_toc_local_ocr`; the other two strategies are commented out.

## Logging

Logs go to `log/toc_forge.log` via `setup_logger`. Uses `logging.DEBUG` level, file-only (no console handler).
