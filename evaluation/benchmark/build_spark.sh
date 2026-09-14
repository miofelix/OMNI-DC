#!/usr/bin/env bash
# 构建本仓库的 DGX Spark 基准镜像并打印镜像 ID。
# 用法：bash evaluation/benchmark/build_spark.sh [image-tag]；需要 Linux ARM64、Docker 和网络。
set -euo pipefail
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "Build this CUDA benchmark image on DGX Spark (Linux aarch64), not on the Mac." >&2
  exit 2
fi
image="${1:-omni-dc-forward:spark}"
docker build --platform linux/arm64 --build-arg "BASE_IMAGE=${SPARK_BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}" -f evaluation/benchmark/Dockerfile.spark -t "$image" .
docker image inspect "$image" --format '{{.Id}}'
