#!/usr/bin/env bash
# Source this script before running toc-forge with the ONNX Runtime engine:
#   source .set_onnx_env.sh
#
# onnxruntime's CUDA provider dlopens libcudnn.so.9 / libcublas.so.13 from
# the nvidia-* pip packages inside .venv-onnx; those libs are not on the
# default search path, so we extend LD_LIBRARY_PATH to find them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv-onnx"
SITE="$VENV/lib/python3.12/site-packages"

export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH extended with cuDNN/cuBLAS from $VENV"
