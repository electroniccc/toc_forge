## TOC Forge
**[中文版 README](README_zh.md)** | **[English README](README.md)**
**One-click automatic PDF bookmark generator.**  
Just drop in a PDF with a table of contents page — get a fully bookmarked PDF instantly. No manual editing, no complex configuration.

Built for scanned books, eBooks, research papers, and any PDF that has a directory page but lacks clickable bookmarks.

## ✨ Features

- Fully automatic: OCR + intelligent clustering to extract TOC
- Multiple modes: Pure local OCR, Local OCR + LLM, Vision LLM
- Extremely simple one-command workflow
- Supports Chinese and English PDFs
- Fast and lightweight

## Usage
**Prepare python env**
```python
uv venv .venv
.\.venv\bin\active
uv pip install -r requirements.txt
```
**Run CLI**
```Python
# build TOC with PaddleOCR running on local machine
python toc_forge.py --input <pdf_file> --output <output_folder>
# build TOC with local OCR + text LLM
python toc_forge.py --input <pdf_file> --api_base_url --api_key <your api key> --llm_name deepseek-v4-flash
# build TOC with vision LLM
python toc_forge.py --input <pdf_file> --api_base_url --api_key <your api key> --vllm_name qwen3.6-flash
```
**Run Web APP**
```python
python web_app.py
```
