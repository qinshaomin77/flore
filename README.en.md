# FLORE

[中文](README.md) | **English**

## Directory layout

- `code/`: Grid36 algorithms, training, episode evaluation and windowed evaluation.
- `code/kunshan/`: Kunshan spatial-scale implementation. Observation scopes, threshold schemas and seed rules differ from Grid36; separate implementations preserve compatibility with the original weights.
- `code/presslight/`: PressLight baseline.
- `configs/`: Complete experiment configurations, organized under `grid36/`, `kunshan/` and `presslight/`. The sibling `thresholds/` directory contains thresholds for each category.
- `data/`: Grid36 and Kunshan inputs and emission factors, preserving relative references between SUMO input files.
- `models/`: Original model weights. Provenance and SHA256 hashes are recorded in `data/manifest.json`.
- `scripts/`: Calibration, scenario generation, batch experiments, validation and result summaries.
- `results/`: Locally generated results, excluded from Git.

## Installation

Validated environment: Python 3.11.17, PyTorch 2.6.0 on CPU, native SUMO 1.20.0, and Python traci/sumolib 1.26.0. Install SUMO separately and add `sumo` to PATH. Setting SUMO_HOME to the SUMO installation directory is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
sumo --version
python run.py doctor --hashes
```

On Linux, activate the environment with `source .venv/bin/activate`; the remaining Python commands are the same. For GPU use, install PyTorch for your CUDA environment. Keep SUMO and input data versions consistent for formal comparisons. Plotting dependencies are provided separately in `requirements-plot.txt`.

## Minimal end-to-end run

```powershell
python run.py quickstart
python run.py quickstart --network kunshan
```

Each run creates a unique result directory, trains for two 300-second episodes, verifies that optimizer updates occurred, and evaluates the new weight for one episode. This checks the workflow; it does not replace the full training used for manuscript experiments.

## Training and evaluation

```powershell
python run.py train --config configs/grid36/flore.yaml --seed 42 --run-id grid36_flore_seed42
python run.py evaluate --checkpoint models/grid36/multi_objective.pt --eval-episodes 20 --workers 3 --run-id flore
python run.py train --network kunshan --config configs/kunshan/flore.yaml --seed 42 --run-id kunshan_flore_seed42
python run.py evaluate --network kunshan --checkpoint models/kunshan_spatial/kunshan_w300_mo.pt --eval-episodes 20 --parallel --max-workers 3 --run-id kunshan_flore
```

To evaluate your own trained model, replace `--checkpoint` with `checkpoint_best.pt` from its training output. The configuration must match the checkpoint's observation dimensions, reward and thresholds. Compatibility errors are reported rather than silently bypassed. Use distinct run IDs or output directories for repeated runs.

Training supports `--resume <checkpoint_latest.pt> --load-optimizer`, which restores model weights, the optimizer and counters before running the requested additional episodes. Replay memory and the complete random state were not saved by the original implementation, so this continues training without guaranteeing an identical step-by-step continuation.

## Demand scenarios and batch experiments

```powershell
python scripts/run_experiments.py --suite train-grid36-demand --dry-run
python scripts/run_experiments.py --suite grid36-main --dry-run
python scripts/run_experiments.py --suite grid36-demand --dry-run
python scripts/run_experiments.py --suite kunshan-spatial --dry-run
```

Remove `--dry-run` to execute. Additional suites include `grid36-ablation`, `grid36-sensitivity` and `grid36-multiseed`. Commands are defined in `configs/grid36/experiments.yaml` and `configs/kunshan/experiments.yaml`. Demand experiments cover D20/D40/D50/D60/D70/D80, each with five truck-share scenarios. Inclusion of D50/D70 in the final manuscript is an author decision.

Sensitivity and multiple-seed suites train from scratch, then locate the corresponding best checkpoint for evaluation. Old weights without their original thresholds are excluded. These suites can take considerable time; run quickstart first. Batch records are saved under `results/batches`. `--resume <batch_directory>` skips only previously successful tasks whose code, configuration, asset manifest and task fingerprints still match. Failed tasks are rerun in new output directories.

Each demand has 160 training scenarios with 4,200-second simulations. Default fixed-network comparisons run for 3,600 seconds. Windowed evaluation normally analyzes 300–3,300 seconds; demand suites explicitly use 600–3,600 seconds. Confirm the final manuscript window against its methods and final experiment records.

## Calibration

Grid36 calibration runs all-sedan scenarios under actuated control, saves `lane_step.csv`, and calculates P80 from positive NOx samples after warm-up separately for each demand, rounding upward. This preserves the original calibration rules.

```powershell
python code/evaluate_actuated.py --config configs/grid36/calibration/gap_actuated_calibration.yaml --calibration-all --calibration-root data/grid36/calibration_sumocfg --calibration-output-root results/calibration/grid36 --eval-episodes 12 --parallel --max-workers 3
python scripts/calibrate_nox_max.py --run-dir results/calibration/grid36 --output-dir results/calibration/grid36/thresholds --quantile 0.8 --warmup-seconds 300
```

Write calibration outputs to `results` for inspection before replacing distributed thresholds. After checking them, point training's `--thresholds-json` to the new file.

For Kunshan, collect w100, w300 and w500 separately:

```powershell
python code/kunshan/evaluate_actuated.py --config configs/kunshan/calibration/gap_actuated_w100.yaml --calibration --eval-episodes 20 --parallel --log-root results/calibration/kunshan --run-id kunshan_actuated_w100
python code/kunshan/evaluate_actuated.py --config configs/kunshan/calibration/gap_actuated_w300.yaml --calibration --eval-episodes 20 --parallel --log-root results/calibration/kunshan --run-id kunshan_actuated_w300
python code/kunshan/evaluate_actuated.py --config configs/kunshan/calibration/gap_actuated_w500.yaml --calibration --eval-episodes 20 --parallel --log-root results/calibration/kunshan --run-id kunshan_actuated_w500
python scripts/calibrate_spatial_nox_max.py --batch-dir results/calibration/kunshan --output-dir results/calibration/kunshan/thresholds
```

Kunshan calibration uses schema 4, does not round upward, and validates episodes, decision intervals and observation scales. Grid36 and Kunshan thresholds are not interchangeable.

## Baselines and parameter variants

```powershell
python run.py baseline --controller actuated --eval-episodes 20 --workers 3
python run.py baseline --controller max_pressure --eval-episodes 20 --workers 3
python run.py baseline --controller truck_weighted_max_pressure --eval-episodes 20 --workers 3
python run.py baseline --network kunshan --controller actuated --eval-episodes 20 --parallel
python run.py presslight-train
python run.py presslight-evaluate --checkpoint <PressLight_checkpoint_path> --episodes 20 --max-workers 3
```

For full options, run `python code/train.py --help`, `python code/evaluate.py --help` or `python code/evaluate_windowed.py --help`. `scripts/make_config.py` creates complete configuration variants using dotted fields such as `--set emission_risk.lambda_e=0.5`, allowing variant training without source-code edits.

## Results and release

```powershell
python scripts/summarize_results.py --input results --output results/summary.csv
pip install -r requirements-plot.txt
python scripts/plot_results.py --input results/summary.csv --metric avg_delay_s --filter grid36 --ylabel "Delay (s)"
```

Summaries preserve source-file identity and do not pool different methods or time windows into one average. These outputs are reproduction results; see the release items in [RELEASE_NOTES.md](RELEASE_NOTES.md) for mapping final results to manuscript tables and figures.

Check all distributed RL weights with `python scripts/check_project.py --models`. Run tests with `python scripts/test_batch.py` and `python scripts/test_calibration.py`.

Windowed metrics do not save per-vehicle Parquet by default. This optional output requires explicit HBEFA4 vehicle emission classes matching the code; some historical calibration and demand inputs do not specify them. Enable `--save-vehicle-state` only for matching inputs. Do not change the formal experiment's emission model merely to save trajectories.

Files in `models/presslight/` are trusted historical full checkpoints from this project. Evaluating them requires `--trusted-checkpoint`. New training also writes `eval_checkpoint_ep_XXXX.pt`, which can be evaluated with the default loading mode.

Local validation details are in [VERIFICATION.md](VERIFICATION.md). The local machine-readable report is `results/verification.json`; `results/` is excluded from Git.

## Scenario generation and configuration layout

Generators write to `results/generated/` by default without overwriting distributed inputs:

```bash
python scripts/generate_grid36.py --kind all
python scripts/generate_kunshan.py --mode base
python scripts/generate_kunshan.py --mode enhanced
python scripts/check_project.py --models
python scripts/check_project.py --update-manifest --hashes
```

Grid36 supports `--kind train/evaluation/calibration/all` for D20/D40/D50/D60/D70/D80, with automatic training-route deduplication. Both generators support `--output-dir`. Kunshan supports `--check` for static validation without writing outputs.

Configuration categories are `configs/grid36`, `configs/kunshan` and `configs/presslight`. Each network's main configuration is `flore.yaml`. Grid36 calibration configurations are in `configs/grid36/calibration`. Thresholds are in the sibling directory `configs/thresholds`, organized under `grid36/`, `kunshan/` and `presslight/`. The 11 threshold files were relocated without changing their contents.

The batch runner automatically discovers the Grid36 and Kunshan `experiments.yaml` files; use `--manifest` for a custom manifest. Duplicate configurations, unused detailed-log configurations, historical full-scale calibration configurations and unused quantile thresholds have been removed.

## Updating GitHub

The repository is available at [qinshaomin77/flore](https://github.com/qinshaomin77/flore). To publish local changes, run these commands from the repository root:

```powershell
git add .
git commit -m "Update FLORE"
git push
```

After intentionally changing distributed inputs, refresh and verify the asset manifest with `python scripts/check_project.py --update-manifest --hashes` before committing.

## Strategy names

Configuration names, comments and task labels follow the seven strategies in Table 3. See [strategy configuration guide](configs/README.md). FLORE-TS is the state ablation and FLORE-TO is the objective ablation. Additional NOx, reward and parameter variants are explicitly labeled as supplemental experiments.
