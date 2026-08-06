param(
    [switch]$OneFile,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

$mainScript = "gui_app.py"
$outputName = "TOC-Forge"
$iconFile = ""
$NuitkaArgs = @(
    "--standalone",
    "--windows-console-mode=disable",
    "--enable-plugin=tk-inter",
    "--include-package=toc_forge",
    "--include-package=sv_ttk",
    "--include-package=requests",
    # ---- web-app deps (unused by GUI) ----
    "--nofollow-import-to=fastapi",
    "--nofollow-import-to=starlette",
    "--nofollow-import-to=uvicorn",
    "--nofollow-import-to=watchfiles",
    "--nofollow-import-to=websockets",
    "--nofollow-import-to=python_multipart",
    # ---- langchain 相关 ----
    # 注意：paddlex 在 import 时就引入 langchain_core / langchain_text_splitters /
    # langsmith / langchain_community（retriever 组件），这几个不能排除；
    # langchain / langchain_openai 未用到，仍可排除
    "--nofollow-import-to=langchain",
    "--nofollow-import-to=langchain_openai",
    "--nofollow-import-to=sqlalchemy",
    "--nofollow-import-to=greenlet",
    "--nofollow-import-to=marshmallow",
    # ---- CLI / formatting deps (unused by GUI) ----
    "--nofollow-import-to=rich",
    "--nofollow-import-to=typer",
    "--nofollow-import-to=click",
    "--nofollow-import-to=shellingham",
    "--nofollow-import-to=pygments",
    # ---- misc ----
    # 注意：pandas / openpyxl / prettytable / tokenizers 是 paddlex 的 import 期硬依赖，不能排除
    "--nofollow-import-to=sentry_sdk",
    "--nofollow-import-to=latex2mathml",
    "--nofollow-import-to=pkg_resources",
    "--noinclude-numba-mode=nofollow",
    "--output-dir=build",
    "--output-filename=$outputName",
    # ---- build speed ----
    # 编译缓存：首次全量编译后，后续只重编改动过的模块
    "--cache-dir=$env:LOCALAPPDATA\Nuitka\Cache",
    # 关闭链接期优化：链接阶段能省不少时间，对非热点代码无感
    "--lto=no",
    $mainScript
)

if ($OneFile) {
    $NuitkaArgs += "--onefile"
    # 跳过 zstd 压缩：打包更快、启动更快，代价是 exe 体积变大
    $NuitkaArgs += "--no-compression"
}

if ($Debug) {
    $NuitkaArgs += "--enable-console"
}

Write-Host "=== TOC Forge — Nuitka Build ===" -ForegroundColor Cyan
Write-Host "  OneFile : $OneFile" -ForegroundColor Gray
Write-Host "  Debug   : $Debug" -ForegroundColor Gray
Write-Host ""

$joined = $NuitkaArgs -join " "
Write-Host "nuitka $joined" -ForegroundColor Yellow
Write-Host ""

& python -m nuitka @NuitkaArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Build complete ===" -ForegroundColor Green
    if ($OneFile) {
        Write-Host "  build/$outputName.exe" -ForegroundColor White
    } else {
        Write-Host "  build/$outputName.dist/$outputName.exe" -ForegroundColor White
    }
} else {
    Write-Host ""
    Write-Host "=== Build FAILED ===" -ForegroundColor Red
    exit $LASTEXITCODE
}
