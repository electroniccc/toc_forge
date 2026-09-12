#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
requirements_path="$script_dir/requirements.txt"
generated_requirements=0
production=0

usage() {
  printf 'Usage: %s [--prod]\n' "$(basename -- "$0")" >&2
}

if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi
if [[ $# -eq 1 ]]; then
  if [[ "$1" == "--prod" ]]; then
    production=1
  else
    usage
    exit 2
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  printf 'Error: uv was not found. Install uv before deploying.\n' >&2
  exit 1
fi
if ! command -v vercel >/dev/null 2>&1; then
  printf "Error: Vercel CLI was not found. Install it with 'npm install --global vercel', then run 'vercel login' and 'vercel link'.\n" >&2
  exit 1
fi
if [[ -e "$requirements_path" ]]; then
  printf 'Error: remove or rename %s before deploying; this script creates a temporary CPU-only file there.\n' \
    "$requirements_path" >&2
  exit 1
fi

cleanup() {
  local status=$?
  trap - EXIT
  if (( generated_requirements )) && [[ -e "$requirements_path" ]]; then
    rm -f -- "$requirements_path" || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd -- "$script_dir"
generated_requirements=1
uv pip compile pyproject.toml \
  --extra web \
  --extra onnx-cpu \
  --python-version 3.12 \
  --python-platform linux \
  --no-emit-package opencv-python \
  --no-header \
  --no-annotate \
  --output-file "$requirements_path"

deploy_args=(deploy --yes)
if (( production )); then
  deploy_args+=(--prod)
fi
vercel "${deploy_args[@]}"
