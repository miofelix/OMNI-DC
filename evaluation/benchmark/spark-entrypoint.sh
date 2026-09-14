#!/usr/bin/env bash
# 在已挂载的当前 checkout 运行基准；所有装配发生在计时前。
# 用法由 Dockerfile.spark 定义；依赖容器 Python 和挂载的仓库。
set -euo pipefail
exec "$@"
