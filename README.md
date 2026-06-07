# PASSION for Dermatology
This repository contains the code to reproduce all evaluations in the paper "PASSION for Dermatology: Bridging the Diversity Gap with Pigmented Skin Images from Sub-Saharan Africa".

## Usage
Run `make` for a list of possible targets.

## Installation
Install the Python dependencies with:

`pip install -r requirements.txt`

or use:

`make install`

## Reproducibility of the Paper
This section is meant as a minimal guide to rerun the experiments used in the
paper and the fairness thesis workflow.

The PASSION images, `label.csv`, and `PASSION_split.csv` must be available
under `data/PASSION`. Dataset access must be requested via
https://passionderm.github.io/.

For reproducibility, install the dependencies from `requirements.txt` before
running the experiments.

If you want to exclude unsupported Fitzpatrick groups globally, set
`dataset.passion.exclude_fitzpatrick_values` in `configs/default.yaml`.
The filter is applied before split generation, training, and fairness
evaluation.

### Before You Run Anything

Set the seeds you want in `configs/default.yaml`:

```yaml
seeds: [32]
```

`seeds` can also be a list such as `seeds: [32, 42]`. In that case, the
selected experiment is run once per seed.

All results are written to `assets/evaluation/seed_<seed>/`.

If a run is interrupted, rerun the same command with `--append_results`.

Completed folds are skipped automatically and existing checkpoints are reused.

### Main Paper Runs

Run the original paper experiments with:

```bash
python -m src.evaluate_experiments --config_path configs/default.yaml --exp1 --exp2
python -m src.evaluate_experiments --config_path configs/default.yaml --exp3 --exp4
```

### Fairness Thesis Runs

Run the fairness baseline and split-generation experiments with:

```bash
python -m src.evaluate_experiments --config_path configs/default.yaml --exp5 --exp6 --exp7
```

Default behavior:

- `exp5`: no cross-validation folds
- `exp6`: 5 folds
- `exp7`: no cross-validation folds; retrains on `TRAIN+VALIDATION` and evaluates on `TEST`

Run the mitigation experiments with:

```bash
python -m src.evaluate_experiments --config_path configs/default.yaml --exp8
python -m src.evaluate_experiments --config_path configs/default.yaml --exp9
python -m src.evaluate_experiments --config_path configs/default.yaml --exp10
python -m src.evaluate_experiments --config_path configs/default.yaml --exp11
python -m src.evaluate_experiments --config_path configs/default.yaml --exp12
```

These runs also use 5 folds by default unless `--mitigation_n_folds` is set explicitly.

Experiment mapping:

- `exp8`: color jitter + oversampling
- `exp9`: instance reweighting
- `exp10`: Group DRO
- `exp11`: Fairlearn ThresholdOptimizer equalized odds
- `exp12`: MIFair fairness regularized loss

Useful options:

- `--split_ids 1 4 6` runs only selected 1-based stratified splits.
- `--mitigation_strengths 0.33 0.67 1.0` runs the mitigation sweep used in the thesis.
- `--mitigation_n_folds 0` switches a mitigation run to a single validation split instead of cross-validation.

Example:

```bash
python -m src.evaluate_experiments --config_path configs/default.yaml --exp10 --split_ids 1 --mitigation_n_folds 0 --mitigation_strengths 1.0
```

The stratified splits can also be generated on their own with:

```bash
cd src/utils/
python -m stratified_split_generator
```

The color-jitter variants can be inspected with:

```bash
jupyter notebook notebooks/inspect_color_jitter.ipynb
```

### Aggregate Results Across Seeds

After the runs finish, aggregate the fairness and performance outputs with:

```bash
python -m src.compare_fairness_across_seeds --seeds 32 --experiments exp5 exp6 exp8 exp9 exp10 exp11 exp12 --model imagenet_tiny
```

The main summary files are written to `results/fairness_comparison/`.
For the thesis workflow, the most useful file is:

- `{exp}_comparison_summary.csv`

This compact summary combines overall performance, Fitzpatrick fairness, and
worst Fitzpatrick subgroup accuracy.

### Fairness Reporting Defaults

The fairness reporting setup is configured in `configs/default.yaml` under
`fairness_evaluation`.

By default, the main thesis outputs use:

- reporting mode, not raw audit mode
- no intersectional subgroup combinations
- Fitzpatrick equalized odds difference as the main fairness comparison
- worst Fitzpatrick subgroup balanced accuracy as the main subgroup comparison

## Code and test conventions
- `black` for code style
- `isort` for import sorting
- docstring style: `sphinx`
- `pytest` for running tests

### Development installation and configurations
To set up your dev environment run:
```bash
pip install -r requirements.txt
# install pre-commit hooks
pre-commit install
```
