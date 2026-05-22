# TOC Forge
**[English README](README.md)** | **[中文 README](README_zh.md)**

**一键自动为 PDF 添加书签工具**  

只需输入一个带有目录页的 PDF，即可快速输出带完整书签的 PDF。  
无需手动编辑目录、无需复杂配置，真正傻瓜式操作。

专为扫描版书籍、电子书、研究论文、技术文档等设计，解决“有目录但没书签”的痛点。

## ✨ 特性

- 全自动：OCR + 智能聚类提取目录结构
- 支持多种模式：纯本地 OCR、本地 OCR + LLM、Vision LLM
- 对中文 PDF 支持良好
- 操作极简，一条命令即可完成
- 轻量快速

## 安装步骤

```bash
git clone https://github.com/electroniccc/toc_forge.git
cd toc_forge

# 创建虚拟环境
uv venv .venv
source .venv/bin/activate        # Windows 用户请执行: .venv\Scripts\activate

# 安装依赖
uv pip install -r requirements.txt
```

## 基础使用
```bash
# 1. 基础模式（仅本地 PaddleOCR，推荐新手使用）
python toc_forge.py --input your_file.pdf --output ./output/

# 2. 使用文本 LLM 增强（目前推荐）
python toc_forge.py --input your_file.pdf --output ./output/ \
  --api_base_url https://api.deepseek.com \
  --api_key sk-xxxxxxxxxxxxxxxx \
  --llm_name deepseek-chat-v4

# 3. 使用 Vision LLM（对复杂排版和扫描件效果更强）
python toc_forge.py --input your_file.pdf --output ./output/ \
  --api_base_url https://api.qwen.ai \
  --api_key sk-xxxxxxxxxxxxxxxx \
  --vllm_name qwen3.6-flash
常用参数说明：
```

## Web APP
```bash
python web_app.py
```