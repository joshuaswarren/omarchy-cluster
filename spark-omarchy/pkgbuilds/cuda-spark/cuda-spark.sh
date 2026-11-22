# shellcheck shell=bash
export PATH="$PATH:/opt/cuda/bin"
export CUDA_HOME=/opt/cuda
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/opt/cuda/lib64"
