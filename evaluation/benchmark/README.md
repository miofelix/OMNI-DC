# OMNI-DC 原生 forward 基准

本基准入口需要 Python 3.10+（Spark 模板使用 Python 3.12）。本目录在本仓库内完整、独立运行，不依赖其他项目的测速代码。协议为
`forward-benchmark-v1`，只统计模型 `forward`。AS-Depth 的 suite 可以调度本入口。

## 本地自检

```bash
python -m evaluation.benchmark --help
python -m evaluation.benchmark run --config evaluation/benchmark/example.json --dry-run
python -m pytest tests/test_forward_benchmark.py -q
```

Mac 只验证配置、输入、计时逻辑和模型装配契约。dry-run 的参数量与 FPS 均为 null。
`params --device cpu` 只适用于原生模型允许 CPU 构造的情况；不会模拟 CUDA 构造。

## 固定实验口径

- batch=1；合成 RGB 为 uint8；深度为 float32 meter、0.1–6.0 m、100% 有效且非恒定。
- 合成图像/目标图像为宽 640、高 480；固定 PCG64 seed=0。
- 保留原生内部 resize、patch/token 设置。报告显示实际 forward 输入和输出尺寸；不同内部尺寸不声称相同计算量。
- 外层不启用 AMP，保留模型 forward 内已有的 autocast。记录参数 dtype、观察到的内部 AMP、TF32 和 attention 环境。
- 每轮预热 50 次，测量 200 次，共 3 轮。模型 eval，torch.inference_mode，输入和模型提前驻留 GPU。
- CUDA Events 只包住一次完整原生 forward。外部准备/传输/后处理不计时；forward 内的全部算子计时。
- FPS = 测量次数 × 1000 / 总毫秒数。保存每次耗时、每轮统计、mean、median、P95。
- 全部去重 Parameter 计入，包括冻结和辅助网络；buffer 单列。checkpoint 加载与哈希在计时前完成。
- 合成负载结果不表示真实数据集准确率、端到端速度或相机帧率。

## Spark 单项目运行

填写 `example.json` 的本地权重路径，另存为自己的配置。相对路径相对于配置所在目录；也可用下列绝对路径覆盖。

```bash
python -m evaluation.benchmark doctor --config evaluation/benchmark/example.json --checkpoint /path/to/checkpoint --mde-checkpoint /path/to/depth_anything_v2_vitl.pth --probe --output outputs/forward-probe
python -m evaluation.benchmark params --config evaluation/benchmark/example.json --checkpoint /path/to/checkpoint --mde-checkpoint /path/to/depth_anything_v2_vitl.pth --output outputs/forward-params
python -m evaluation.benchmark run --config evaluation/benchmark/example.json --checkpoint /path/to/checkpoint --mde-checkpoint /path/to/depth_anything_v2_vitl.pth --output outputs/forward-run
python -m evaluation.benchmark summarize outputs/forward-run --output outputs/forward-summary
```

每次选择新的输出目录，避免把旧结果误当作当前结果。`doctor` 不加 `--probe` 仅检查环境与资源，
不证明模型 kernel 可运行。权重或依赖缺失、非有限输出、OOM、设备错误均保留失败状态，FPS 留空。
不会下载权重、换随机权重、改精度、降尺寸或自动回退 CPU。

产物：`benchmark.json`、`run.log`、`summary.json`、`summary.csv`、`summary.md`。
JSON 保存源码版本和工作区状态、测速源码哈希、权重指纹、合成输入指纹、真实尺寸和运行环境。
汇总输入/计时/硬件/基础运行环境不一致时，报告明确标记不兼容并返回非零退出码。

## 独立 Spark 环境

```bash
bash evaluation/benchmark/build_spark.sh
```

镜像模板固定基于 `nvcr.io/nvidia/pytorch:25.11-py3`，为 Linux ARM64 准备。
依赖安装保留镜像内 Torch/torchvision 的精确版本，并安装独立基准依赖；不运行原项目的旧版 uv 锁。
代码从挂载的当前 checkout 运行。Mac 阶段未构建该 CUDA 镜像，必须在 Spark 上通过 `doctor --probe` 验证。
构建脚本输出镜像 ID；运行时通过 `BENCHMARK_CONTAINER_IMAGE` 记录这个 ID，保留基础镜像 digest。

```bash
# 在仓库根目录运行。权重、配置、输出放在挂载路径下；其他权重目录另加只读挂载。
docker run --rm --gpus all \
  -v "$PWD:$PWD" -w "$PWD" \
  -e BENCHMARK_CONTAINER_IMAGE="$(docker image inspect omni-dc-forward:spark --format '{{.Id}}')" \
  omni-dc-forward:spark python -m evaluation.benchmark run \
  --config evaluation/benchmark/example.json --output outputs/forward-run
```

只有 Prior 镜像从固定 `torch-cluster==1.6.3` 源码编译 GB10 kernel；其 ARM64/CUDA 兼容性属于服务器门禁。
xFormers 不作为新增必装项；实际是否存在及是否使用以报告和原生实现为准。基础镜像、驱动或算子不兼容时先修复该环境，再重新运行。
不应通过更换一个模型的基础 Torch 版本后继续声称所有模型运行环境相同。

服务器自动门禁：

```bash
FORWARD_BENCHMARK_CONFIG=/absolute/path/to/your-config.json \
  python -m pytest tests/test_forward_benchmark_server.py -q
```

该门禁使用完整真实模型和默认查询尺寸，只缩短预热/迭代次数检查可运行性，不生成正式性能结论。
