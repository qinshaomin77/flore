# FLORE

**中文** | [English](README.en.md)

## 目录

- `code/`：Grid36 核心算法、训练、逐回合评估和窗口评估。
- `code/kunshan/`：昆山空间尺度实现。观测范围、阈值 schema、随机种子规则与 Grid36 不同，保留独立实现以兼容原权重。
- `code/presslight/`：PressLight 基线。
- `configs/`：完整解析后的实验参数，按 `grid36/`、`kunshan/`、`presslight/` 分类；同级的 `thresholds/` 保存各分类阈值。
- `data/`：Grid36、昆山和排放因子，保持 SUMO 输入的相对引用。
- `models/`：原始模型权重，未经重写；来源和 SHA256 见 `data/manifest.json`。
- `scripts/`：标定、场景生成、批量实验、校验和汇总。
- `results/`：本机新结果，Git 忽略。

## 安装

验证环境：Python 3.11.17、PyTorch 2.6.0 CPU、SUMO 原生程序 1.20.0；Python traci/sumolib 1.26.0。安装 SUMO 并将 `sumo` 加入 PATH；推荐设置 SUMO_HOME 为 SUMO 安装目录。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
sumo --version
python run.py doctor --hashes
```

Linux 激活命令为 `source .venv/bin/activate`，其余 Python 命令相同。GPU 用户按自己的 CUDA 环境安装 PyTorch；正式对比需保持 SUMO 和输入数据版本一致。绘图可另装 `requirements-plot.txt`。

## 最短完整运行

```powershell
python run.py quickstart
python run.py quickstart --network kunshan
```

每次创建唯一结果目录：2 个 300 秒训练回合，检查优化器确实更新，再用新权重评估 1 回合。此配置用于检查流程，不能代替正式论文训练。

## 训练与评估

```powershell
python run.py train --config configs/grid36/flore.yaml --seed 42 --run-id grid36_flore_seed42
python run.py evaluate --checkpoint models/grid36/multi_objective.pt --eval-episodes 20 --workers 3 --run-id flore
python run.py train --network kunshan --config configs/kunshan/flore.yaml --seed 42 --run-id kunshan_flore_seed42
python run.py evaluate --network kunshan --checkpoint models/kunshan_spatial/kunshan_w300_mo.pt --eval-episodes 20 --parallel --max-workers 3 --run-id kunshan_flore
```

评估自己训练的模型时，将 `--checkpoint` 替换为对应训练结果目录中的 `checkpoint_best.pt`。配置必须与权重的观测维度、奖励和阈值匹配；错误会直接报告，不默认放宽兼容检查。重复运行时选择不同 run-id / 输出目录。

训练支持 `--resume <checkpoint_latest.pt> --load-optimizer`：恢复模型、优化器和计数器，再执行指定回合数。原实现没有保存经验回放与完整随机状态，因此这是继续训练，不是逐步完全一致的断点恢复。

## 需求场景和批量实验

```powershell
python scripts/run_experiments.py --suite train-grid36-demand --dry-run
python scripts/run_experiments.py --suite grid36-main --dry-run
python scripts/run_experiments.py --suite grid36-demand --dry-run
python scripts/run_experiments.py --suite kunshan-spatial --dry-run
```

去掉 `--dry-run` 执行；其他套件：`grid36-ablation`、`grid36-sensitivity`、`grid36-multiseed`。`configs/grid36/experiments.yaml` 和 `configs/kunshan/experiments.yaml` 明确每个命令。需求实验包含 D20/D40/D50/D60/D70/D80，每个需求 5 个货车比例场景；D50/D70 是否纳入最终论文由作者决定。敏感性和多种子套件会从头训练，再自动找到相应的最佳权重进行评估；旧权重因缺少原阈值已排除。任务多且运行时间长，先做 quickstart。批量记录保存在 results/batches；`--resume <batch目录>` 仅跳过代码、配置、资产清单和任务指纹一致且上次成功的任务，失败重跑使用新的输出目录。

需求训练输入每个需求有 160 个场景，4200 秒仿真；默认固定路网对比为 3600 秒。窗口默认分析 300–3300 秒；需求套件显式使用 600–3600 秒。论文使用的最终窗口必须以论文方法和最终实验记录确认，不应凭目录名称猜测。

## 标定

Grid36 标定流程：运行全轿车自适应信号场景，保存 lane_step.csv，然后按需求独立计算暖机后的正 NOx 样本 P80，向上取整。保留原标定规则。

```powershell
python code/evaluate_actuated.py --config configs/grid36/calibration/gap_actuated_calibration.yaml --calibration-all --calibration-root data/grid36/calibration_sumocfg --calibration-output-root results/calibration/grid36 --eval-episodes 12 --parallel --max-workers 3
python scripts/calibrate_nox_max.py --run-dir results/calibration/grid36 --output-dir results/calibration/grid36/thresholds --quantile 0.8 --warmup-seconds 300
```

先输出到 results 核对，避免覆盖公开阈值；确认后用训练的 `--thresholds-json` 指向新文件。

昆山分别采集 w100、w300、w500：

```powershell
python code/kunshan/evaluate_actuated.py --config configs/kunshan/calibration/gap_actuated_w100.yaml --calibration --eval-episodes 20 --parallel --log-root results/calibration/kunshan --run-id kunshan_actuated_w100
python code/kunshan/evaluate_actuated.py --config configs/kunshan/calibration/gap_actuated_w300.yaml --calibration --eval-episodes 20 --parallel --log-root results/calibration/kunshan --run-id kunshan_actuated_w300
python code/kunshan/evaluate_actuated.py --config configs/kunshan/calibration/gap_actuated_w500.yaml --calibration --eval-episodes 20 --parallel --log-root results/calibration/kunshan --run-id kunshan_actuated_w500
python scripts/calibrate_spatial_nox_max.py --batch-dir results/calibration/kunshan --output-dir results/calibration/kunshan/thresholds
```

昆山标定输出 schema 4，不向上取整，校验回合、决策间隔和观测尺度。Grid36 和昆山阈值不能混用。

## 基线与参数变体

```powershell
python run.py baseline --controller actuated --eval-episodes 20 --workers 3
python run.py baseline --controller max_pressure --eval-episodes 20 --workers 3
python run.py baseline --controller truck_weighted_max_pressure --eval-episodes 20 --workers 3
python run.py baseline --network kunshan --controller actuated --eval-episodes 20 --parallel
python run.py presslight-train
python run.py presslight-evaluate --checkpoint <PressLight权重路径> --episodes 20 --max-workers 3
```

详细选项：`python code/train.py --help`、`python code/evaluate.py --help`、`python code/evaluate_windowed.py --help`。`scripts/make_config.py` 可从完整配置生成变体，支持 `--set emission_risk.lambda_e=0.5` 等点分字段；不修改源码即可重新训练变体。

## 结果与发布

```powershell
python scripts/summarize_results.py --input results --output results/summary.csv
pip install -r requirements-plot.txt
python scripts/plot_results.py --input results/summary.csv --metric avg_delay_s --filter grid36 --ylabel "Delay (s)"
```

汇总保留源文件身份，不把不同方法或不同时间窗口混为一个均值。输出只是复现结果；论文表格与图的最终对应关系见 RELEASE_NOTES.md 的发布事项。

完整权重预检：`python scripts/check_project.py --models`。测试：`python scripts/test_batch.py` 和 `python scripts/test_calibration.py`。

窗口指标默认不保存逐车 Parquet。该可选输出要求 route 文件的车型显式使用代码规定的 HBEFA4 类；旧标定/需求输入未全部设置该属性。仅在输入匹配时开启 `--save-vehicle-state`，不用为了保存轨迹改变正式实验的排放模型。

`models/presslight/` 内为可信本项目的历史完整权重，评估这些文件需显式加 `--trusted-checkpoint`。新训练同时输出 `eval_checkpoint_ep_XXXX.pt`，可直接用于默认评估。

本地验证详情见 VERIFICATION.md；机器可读报告位于 results/verification.json。

## 场景生成与配置结构

场景生成工具已统一，默认写入 `results/generated/`，不会覆盖分发数据：

```bash
python scripts/generate_grid36.py --kind all
python scripts/generate_kunshan.py --mode base
python scripts/generate_kunshan.py --mode enhanced
python scripts/check_project.py --models
python scripts/check_project.py --update-manifest --hashes
```

Grid36 支持 `--kind train/evaluation/calibration/all`，涵盖 D20/D40/D50/D60/D70/D80；训练路由自动去重。两个生成器均支持 `--output-dir`。Kunshan 可用 `--check` 仅执行静态验证。标定入口与批量实验入口保持原命令。

配置按 `configs/grid36`、`configs/kunshan`、`configs/presslight` 分类，默认配置为各网络的 `flore.yaml`。Grid36 标定配置位于 `grid36/calibration`；阈值统一位于 `configs/thresholds/grid36`、`configs/thresholds/kunshan`、`configs/thresholds/presslight`。实验清单分别位于 grid36/kunshan 的 `experiments.yaml`，批量入口自动查找；自定义清单可用 `--manifest`。重复配置与未使用的详细日志、历史全尺度标定、分位数阈值已移除。

阈值目录 `configs/thresholds` 与 grid36、kunshan、presslight 配置目录同级，11 个阈值文件仅调整位置，内容保持一致。

## 更新 GitHub

项目仓库：[qinshaomin77/flore](https://github.com/qinshaomin77/flore)。在项目根目录执行以下命令发布本地修改：

```powershell
git add .
git commit -m "Update FLORE"
git push
```

修改分发输入后，提交前运行 `python scripts/check_project.py --update-manifest --hashes` 更新并核对资产清单。

## 策略名称

配置、注释与实验任务统一采用表 3 的七种策略名称，详见 [策略配置说明](configs/README.md)。FLORE-TS 为状态消融，FLORE-TO 为目标消融；其他 NOx、奖励和参数变体单独标注为扩展实验。
