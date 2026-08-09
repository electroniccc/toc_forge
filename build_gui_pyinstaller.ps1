param(
    [switch]$OneFile,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

$mainScript = "gui_app.py"

# 版本号取自 toc_forge.__version__（toc_forge/__init__.py），拼进产物名
# （exe/目录/zip 一并带上），便于区分不同版本的分发包
$version = (& python -c 'from toc_forge import __version__; print(__version__)' 2>$null | Select-Object -First 1)
if ($version) {
    $version = "-$($version.Trim())"
} else {
    Write-Host "WARNING: 读取 toc_forge.__version__ 失败，产物名将不带版本号" -ForegroundColor Yellow
    $version = ""
}
$outputName = "TOC-Forge$version"

# 清理上次构建产物。PyInstaller 的 --noconfirm 只会删 dist\TOC-Forge 目录本身，
# 残留的旧 onefile exe / zip 需要手动清掉，避免 onefile/onedir 切换时混淆
if (Test-Path "dist\$outputName")     { Remove-Item "dist\$outputName" -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path "dist\$outputName.exe") { Remove-Item "dist\$outputName.exe" -Force -ErrorAction SilentlyContinue }
if (Test-Path "dist\$outputName.zip") { Remove-Item "dist\$outputName.zip" -Force -ErrorAction SilentlyContinue }

$PyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--name=$outputName",
    "--distpath=dist",
    "--workpath=build\pyinstaller",
    "--noupx",
    # paddle/paddleocr/paddlex 大量懒加载子模块 + 散落的 .pyd/.dll，
    # 静态分析会漏，整包收集最稳
    "--collect-all=paddle",
    "--collect-all=paddleocr",
    "--collect-all=paddlex",
    # GUI 固定用 onnxruntime 引擎（gui_app.py 里 engine="onnxruntime"）：
    # capi 下的 onnxruntime_pybind11_state.pyd + onnxruntime.dll 等二进制
    # 依赖运行时目录，collect-all 整包收集最稳。注意构建环境必须已安装
    # onnxruntime（建议 1.27，1.28 的 get_available_providers() 有 bug）
    "--collect-all=onnxruntime",
    # sv_ttk 的 tcl 主题文件（sv.tcl / theme/）是数据文件，导入分析不自动收集
    "--collect-all=sv_ttk",
    # 本地包：editable 安装下 PyInstaller 的查找偶尔不稳，显式给路径
    "--paths=.",
    "--collect-all=toc_forge",
    # ---- paddlex 的依赖检查（paddlex/utils/deps.py）用 importlib.metadata.version()
    #      判断依赖是否可用，PyInstaller 默认不打包各依赖的 .dist-info 元数据，
    #      导致打包环境里全部判为"不可用"，报
    #      "A dependency error occurred during predictor/pipeline creation"。
    #      下面覆盖两类检查：predictor 组件检查 + paddlex[ocr]/[ocr-core] extra 门槛
    "--copy-metadata=opencv-contrib-python",
    "--copy-metadata=pyclipper",
    "--copy-metadata=python-bidi",
    "--copy-metadata=shapely",
    "--copy-metadata=pypdfium2",
    "--copy-metadata=lxml",
    "--copy-metadata=openpyxl",
    "--copy-metadata=scikit-learn",
    "--copy-metadata=imagesize",
    "--copy-metadata=beautifulsoup4",
    "--copy-metadata=einops",
    "--copy-metadata=ftfy",
    "--copy-metadata=Jinja2",
    "--copy-metadata=latex2mathml",
    "--copy-metadata=premailer",
    "--copy-metadata=regex",
    "--copy-metadata=safetensors",
    "--copy-metadata=scipy",
    "--copy-metadata=sentencepiece",
    "--copy-metadata=tiktoken",
    "--copy-metadata=tokenizers",
    # ---- web-app deps (unused by GUI) ----
    "--exclude-module=fastapi",
    "--exclude-module=starlette",
    "--exclude-module=uvicorn",
    "--exclude-module=watchfiles",
    "--exclude-module=websockets",
    "--exclude-module=python_multipart",
    # ---- langchain 相关 ----
    # 注意：paddlex 在 import 时就引入 langchain_core / langchain_text_splitters /
    # langsmith / langchain_community（retriever 组件），这几个不能排除；
    # langchain / langchain_openai 未用到，仍可排除
    "--exclude-module=langchain",
    "--exclude-module=langchain_openai",
    "--exclude-module=sqlalchemy",
    "--exclude-module=greenlet",
    "--exclude-module=marshmallow",
    # ---- CLI / formatting deps ----
    "--exclude-module=rich",
    "--exclude-module=typer",
    "--exclude-module=click",
    "--exclude-module=shellingham",
    "--exclude-module=pygments",
    # ---- misc ----
    # 注意：pandas / openpyxl / prettytable / tokenizers 是 paddlex 的 import 期硬依赖，不能排除
    "--exclude-module=sentry_sdk",
    "--exclude-module=latex2mathml",
    "--exclude-module=pkg_resources"
)

# onnxruntime 引擎：paddlex 的依赖检查以 importlib.metadata 判定 onnxruntime
# 是否可用，缺 .dist-info 会被判为不可用。注意 .venv-onnx 装的是
# onnxruntime-gpu（dist-info 目录名 onnxruntime_gpu-*.dist-info），而 .venv 装
# 的是 onnxruntime —— PyInstaller 的 copy_metadata 按元数据包名查找，两个
# 环境包名不同，这里动态检测取实际名字
$ortMeta = & python -c @"
import importlib.metadata as m
for name in ('onnxruntime-gpu', 'onnxruntime'):
    try:
        m.version(name)
        print(name)
        break
    except Exception:
        pass
"@
if ($ortMeta) {
    $PyInstallerArgs += "--copy-metadata=$ortMeta"
}

if ($Debug) {
    $PyInstallerArgs += "--console"
    $PyInstallerArgs += "--debug=all"
} else {
    # GUI 程序：不带控制台窗口（print 输出将被丢弃）
    $PyInstallerArgs += "--noconsole"
}

if ($OneFile) {
    $PyInstallerArgs += "--onefile"
}

Write-Host "=== TOC Forge — PyInstaller Build ===" -ForegroundColor Cyan
Write-Host "  OneFile : $OneFile" -ForegroundColor Gray
Write-Host "  Debug   : $Debug" -ForegroundColor Gray
Write-Host ""

$joined = $PyInstallerArgs -join " "
Write-Host "pyinstaller $joined $mainScript" -ForegroundColor Yellow
Write-Host ""

& python -m PyInstaller @PyInstallerArgs $mainScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Build complete ===" -ForegroundColor Green
    if ($OneFile) {
        Write-Host "  dist/$outputName.exe" -ForegroundColor White
    } else {
        Write-Host "  dist/$outputName/$outputName.exe" -ForegroundColor White

        # onedir：把整个 dist 文件夹打成 zip，便于分发（解压一次即"安装"）
        $zipPath = "dist\$outputName.zip"
        if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
        Write-Host "  Packaging dist/$outputName -> $zipPath ..." -ForegroundColor Yellow
        if (Get-Command tar.exe -ErrorAction SilentlyContinue) {
            # Windows 11 自带的 bsdtar，对几百 MB 的大目录比 Compress-Archive 快得多
            tar -a -c -f $zipPath -C "dist" $outputName
        } else {
            Compress-Archive -Path "dist\$outputName" -DestinationPath $zipPath
        }
        if ($LASTEXITCODE -eq 0) {
            $sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
            Write-Host "  $zipPath ($sizeMB MB)" -ForegroundColor White
        } else {
            Write-Host "  Zip failed (exit $LASTEXITCODE)" -ForegroundColor Red
        }
    }
} else {
    Write-Host ""
    Write-Host "=== Build FAILED ===" -ForegroundColor Red
    exit $LASTEXITCODE
}
