# autoresearch — Traffic Congestion Prediction

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for the ECE1513 traffic speed prediction project.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr1`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
3. **Read the in-scope files**:
   - `program.md` — this file (experiment protocol).
   - `config.py` — path constants. Do not modify.
   - `train.py` — **the file you modify**. Features, model, hyperparameters.
4. **Verify data exists**: Check that `data/processed/final_traffic_weather_merged.csv` exists. If not, tell the human to run `python 02_process_weather.py && python 03_process_traffic.py && python 04_merge_data.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**.

## Experimentation

Each experiment modifies `train.py` and runs it: `python train.py > run.log 2>&1`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - XGBoost hyperparameters (learning_rate, max_depth, n_estimators, regularization, etc.)
  - Feature engineering (add new features, remove features, transform features)
  - Feature selection (try subsets, add interaction features, etc.)
  - Model architecture (try LightGBM, ensemble methods, stacking, etc.)
  - Target transformation (log transform, deviation-based target, etc.)
  - Data filtering (minimum data quality thresholds, outlier removal, etc.)

**What you CANNOT do:**
- Modify `config.py` or the raw data pipeline scripts (01-04).
- Change the evaluation metric definitions (MAE, RMSE, R²).
- Change the train/val/test split logic (must remain 70/10/20 chronological).
- Install new packages beyond what's available (pandas, numpy, scikit-learn, xgboost, lightgbm).

**The goal: minimize test_mae (lower is better).** Secondary goals: minimize RMSE, maximize R².

**Simplicity criterion**: All else being equal, simpler is better. A tiny MAE improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Always establish the baseline first by running `train.py` as-is.

## Output format

The script prints a summary block:

```
---
test_mae:     X.XXXXXX
test_rmse:    X.XXXXXX
test_r2:      X.XXXXXX
train_time_s: X.X
best_iter:    N
n_features:   N
n_train:      N
n_test:       N
---
```

Extract the key metric: `grep "^test_mae:" run.log`

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	test_mae	test_rmse	test_r2	status	description
```

1. git commit hash (short, 7 chars)
2. test_mae achieved — use 0.000000 for crashes
3. test_rmse achieved — use 0.000000 for crashes
4. test_r2 achieved — use 0.000000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	test_mae	test_rmse	test_r2	status	description
a1b2c3d	5.123456	7.654321	0.850000	keep	baseline XGBoost
b2c3d4e	4.987654	7.432100	0.860000	keep	increase max_depth to 8
c3d4e5f	5.234567	7.890123	0.840000	discard	remove weather features
d4e5f6g	0.000000	0.000000	0.000000	crash	try stacking (import error)
```

## The experiment loop

LOOP FOREVER:

1. Look at git state and results.tsv to understand where you are.
2. Modify `train.py` with an experimental idea.
3. `git commit -m "experiment: <short description>"`
4. Run: `python train.py > run.log 2>&1`
5. Extract results: `grep "^test_mae:\|^test_rmse:\|^test_r2:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the traceback.
7. Record in `results.tsv` (do NOT commit results.tsv).
8. If test_mae improved (lower), keep the commit.
9. If test_mae is equal or worse, `git reset --hard HEAD~1` to revert.

**Ideas to try** (in rough priority order):
- Hyperparameter tuning: learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda
- Feature engineering: interaction terms (speed × volume), time-of-day buckets, rush hour flags, seasonal features
- Additional lag windows: lag 12, 16, 24 (6-hour lookback)
- Target engineering: predict speed_deviation instead of raw speed, then add back historical avg
- Feature selection: remove low-importance features, try ablation
- Try LightGBM as a drop-in replacement
- Ensemble: average XGBoost + LightGBM predictions
- Outlier filtering: remove extreme speed values before training
- Per-segment normalization or grouping strategies

**Timeout**: Each run should complete in under 5 minutes. If it exceeds 10 minutes, kill and revert.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. If you run out of ideas, re-read `train.py`, think harder about feature engineering, try combining near-misses, try more radical changes. Loop until manually interrupted.
