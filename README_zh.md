# TOC Forge — PDF 目录转书签工具

中文 | [English README](README.md)

TOC Forge 可以自动识别 PDF 中已有的**目录页，并将目录转换为可点击、可跳转的 PDF 层级书签**。它使用 PaddleOCR 或可选的 LLM 提取标题和印刷页码，恢复章节层级，校正印刷页码与 PDF 页序之间的偏移，最后输出带有导航大纲的新 PDF。

适用于“**PDF 有目录页但没有书签**”的扫描版书籍、电子书、教材、论文和技术文档，也可以用于扫描 PDF 自动添加书签、OCR 识别 PDF 目录、让 PDF 目录可点击，以及从目录页生成 PDF 大纲。

## 添加书签前后对比

| 添加前：PDF 没有书签 | 添加后：TOC Forge 生成层级书签 |
| --- | --- |
| ![扫描版 PDF 添加书签前，阅读器左侧只有页面缩略图](doc/imgs/before_bookmark.jpg) | ![TOC Forge 识别目录后，为扫描版 PDF 生成可点击的层级书签](doc/imgs/after_bookmark.jpg) |

程序不会覆盖原始 PDF。默认输出为 `output/<原文件名>_bookmarked.pdf`。

## 功能特点

- **自动检测目录页：**无需手动填写目录页码，可以识别单页或连续多页目录。
- **目录转 PDF 书签：**提取标题、页码、缩进和层级，并写入原生 PDF Outline。
- **支持扫描版和电子版 PDF：**以页面图像为输入，不要求 PDF 已有可搜索文本层。
- **支持中文和英文目录：**主要依据版面位置与缩进恢复结构，并兼容常见章节编号。
- **自动校正页码：**处理印刷页码与 PDF 页序的偏移，也支持前言罗马数字页码以及 `I-1`、`II-3` 等格式。
- **自动跳过无页码空白页：**识别正文中插入的空白页或无页码页，并重新校正其后所有书签的页码偏移。
- **三种提取方式：**纯本地 OCR、本地 OCR + 文本 LLM、视觉 LLM。
- **本地隐私模式：**默认模式不会把 PDF 内容发送到外部 API。
- **提供 CLI、Windows 桌面 GUI 和本地 Web UI。**
- **结果缓存：**重复处理时可复用版面分析、OCR 和可选 LLM 结果。

## 工作流程

```text
PDF → 检测目录页 → OCR 或视觉 LLM → 恢复目录层级
    → 映射印刷页码 → 写入可点击的 PDF 书签
```

TOC Forge 不会根据全文凭空生成目录。输入 PDF 必须已经包含一页或多页可见目录。

## 环境要求

- Python 3.10 或更高版本
- Windows 或 Linux；打包版桌面工作流主要面向 Windows
- 第一次运行需要联网下载 PaddleOCR/PaddleX 模型
- 只有使用文本 LLM 或视觉 LLM 时才需要 API Key

OCR 模型体积较大，本地推理会占用一定 CPU 和内存。版面与 OCR 结果会被缓存，因此同一文件重复运行通常会更快。

## 安装

克隆仓库并创建虚拟环境：

```bash
git clone https://github.com/electroniccc/toc_forge.git
cd toc_forge
uv venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

安装命令行工具：

```bash
uv pip install ".[onnx-cpu]"
```

这会安装推荐的 ONNX Runtime CPU 依赖。需要其他推理运行时时，改用 `.[onnx-gpu]` 或 `.[paddle-gpu]`。不使用 `uv` 时可执行 `pip install ".[onnx-cpu]"`。

## 快速开始：纯本地 OCR

不需要 LLM 或 API Key：

```bash
toc-forge --input book.pdf --output output
```

也可以使用模块方式运行：

```bash
python -m toc_forge --input book.pdf --output output
```

CPU 环境可以选择较小的 OCR 模型以缩短处理时间：

```bash
toc-forge --input book.pdf --output output --device cpu --ocr_model_size mobile
```

Windows 下如果 Paddle 引擎报告 oneDNN/MKLDNN 错误，可以执行：

```powershell
toc-forge --input book.pdf --output output --device cpu --disable_mkldnn
```

## 可选 LLM 模式

TOC Forge 支持 OpenAI-compatible API。程序会根据传入的模型选项自动选择处理模式。

### 本地 OCR + 文本 LLM

先在本地识别目录，再把 OCR 文本和位置发给文本模型恢复层级：

```powershell
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
$env:OPENAI_API_KEY = "your-api-key"
toc-forge --input book.pdf --output output --llm_name your-text-model
```

### 视觉 LLM

将检测出的目录页图片直接发送给支持视觉输入的模型：

```powershell
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
$env:OPENAI_API_KEY = "your-api-key"
toc-forge --input book.pdf --output output --vllm_name your-vision-model
```

Linux/macOS 请使用 `export OPENAI_BASE_URL=...` 和 `export OPENAI_API_KEY=...`。

| 模式 | 是否调用外部 API | 适用场景 |
| --- | --- | --- |
| 纯本地 OCR | 否 | 隐私优先、模型下载后离线处理、常见目录版式 |
| OCR + 文本 LLM | 是，发送 OCR 文本与位置 | OCR 有噪声或本地规则无法恢复层级 |
| 视觉 LLM | 是，发送目录页图片 | 复杂视觉排版或多模态模型 |

## Windows 桌面 GUI

安装 GUI 依赖并启动：

```powershell
uv pip install ".[gui]"
python .\gui_app.py
```

选择 PDF 和输出目录，第一次使用时点击“下载 / 检查模型”。GUI 使用 ONNX Runtime 和移动版 OCR 模型在 CPU 上推理，也支持一次选择多个 PDF。

## 本地 Web UI

```bash
uv pip install ".[web,onnx-cpu]"
python web_app.py  # 默认：--engine onnxruntime --device cpu
```

浏览器会打开 `http://127.0.0.1:8000`，上传的文件由本机启动的服务处理。
如需使用其他已安装的运行时，可通过 `--engine` 和 `--device` 指定，例如
`python web_app.py --engine onnxruntime --device gpu`。

### 部署到 Vercel

安装并登录 Vercel CLI，在仓库中首次执行 `vercel link` 关联项目，然后在 Linux、macOS 或 WSL 的 Bash 中部署：

```bash
bash ./deploy_vercel.sh       # 预览部署
bash ./deploy_vercel.sh --prod
```

Vercel 入口固定使用 ONNX Runtime CPU 和 mobile OCR 模型。模型会在首次请求时下载到函数临时目录；冷启动后可能需要重新下载。Vercel Functions 的请求体和响应体均限制为 4.5 MB，因此上传的 PDF 与生成的带书签 PDF 都必须小于此限制。
部署脚本会根据 `pyproject.toml` 临时生成 Linux CPU 版 `requirements.txt`，部署结束后删除；仓库不再保存固定依赖清单。

## 常用命令行参数

| 参数 | 说明 |
| --- | --- |
| `--input PATH` | 输入 PDF，必填 |
| `--output DIR` | 输出目录，默认为 `output` |
| `--device cpu|gpu|gpu:0` | PaddleOCR 推理设备 |
| `--engine onnxruntime` | 使用 ONNX Runtime 推理引擎 |
| `--ocr_model_size server|mobile` | 精度较高的 server 模型或速度较快的 mobile 模型 |
| `--toc_detect_max_page N` | 最多从 PDF 开头扫描多少页寻找目录 |
| `--cache_dir DIR` | 版面分析与 OCR 缓存目录 |
| `--no_toc_cache` | 忽略已缓存的目录树，重新调用 LLM |
| `--debug` | 保存中间图片和 JSON，便于排查问题 |

执行 `toc-forge --help` 可以查看全部参数。

## 适用范围与限制

TOC Forge 面向目录中包含标题和页码的书籍、电子书、论文、手册等 PDF。它支持扫描页、多页目录、中英文文本、阿拉伯数字页码、前言罗马数字页码和部分非常规页码格式。

识别结果仍会受到扫描质量和目录版式影响。手写目录、页面严重裁切、装饰性很强的版式、目录缺少页码或没有可见目录的文档，可能需要改用 LLM 模式或人工修正。发布 PDF 前建议检查生成的书签。

## 常见问题

### 扫描版 PDF 怎么自动添加书签？

只要 PDF 中已有可见的目录页，就可以运行 TOC Forge。程序通过 OCR 提取目录，并另存为带可点击书签的新 PDF，不要求原文件具有文本层。

### 能把 PDF 中现有的目录页转换成书签吗？

可以。这正是 TOC Forge 的主要用途：把印刷或扫描目录中的标题、层级和页码转换为原生 PDF 大纲。

### 不使用 ChatGPT、Claude、DeepSeek、豆包等 LLM 可以运行吗？

可以。默认的纯本地 OCR 模式不会调用外部 LLM。只有本地方式难以解析某些目录时，才需要选择兼容 OpenAI API 的文本或视觉模型。

### 会修改原始 PDF 吗？

不会。程序会在输出目录新建 `<原文件名>_bookmarked.pdf`。

### 为什么书签有时会跳到错误页面？

封面、前言以及章节之间插入的空白页或无页码页，都会导致印刷页码与 PDF 页序不同。TOC Forge 会自动估算偏移，跳过无页码空白页，修正后续书签位置，并支持罗马数字和分段页码；非常规排版或模糊页码仍建议人工检查。

## 故障排查

- 没有检测到目录时，使用 `--debug` 查看生成的中间图片和 JSON。
- 目录出现在文档较后位置时，增大 `--toc_detect_max_page`。
- 需要更高精度时使用 `--ocr_model_size server`，CPU 较慢时使用 `mobile`。
- 更换 LLM 模型或 API 地址后，使用 `--no_toc_cache` 重新生成目录树。
- 默认运行日志位于 `log/toc_forge.log`。

## 开源协议

[MIT License](LICENSE)
