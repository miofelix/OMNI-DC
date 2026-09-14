"""独立测速入口：python -m evaluation.benchmark；FPS 需要 CUDA。"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
