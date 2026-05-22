## TOC Forge
An automatic tool for adding bookmarks to PDF files. Just input a PDF and get a bookmarked PDF without any extra configuration.

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
