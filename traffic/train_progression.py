"""
ECE1513 — Model improvement progression
Target: next 15-min avg_speed_kph
Split: 80/10/10 chronological
Steps: XGBoost baseline → tuned XGBoost → ResNet MLP
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from time import perf_counter

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import torch
import torch.nn as nn

from config import MERGED_DATA, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42
GROUP_COLS = ["centreline_id", "direction"]

# ── 1. Load & Prepare ────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(MERGED_DATA)
df["time_start"] = pd.to_datetime(df["time_start"])
df["time_end"] = pd.to_datetime(df["time_end"])

for col in ["is_raining", "is_snowing", "is_foggy"]:
    if col in df.columns:
        df[col] = df[col].replace({True: 1, False: 0}).fillna(0).astype(int)

for col in ["longitude", "latitude", "volume_15min", "avg_speed_kph",
            "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
            "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(GROUP_COLS + ["time_start"]).reset_index(drop=True)

# Time features
df["hour"] = df["time_start"].dt.hour
df["minute"] = df["time_start"].dt.minute
df["day_of_week"] = df["time_start"].dt.dayofweek
df["month"] = df["time_start"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 9) & (df["is_weekend"] == 0)).astype(int)
df["is_evening_rush"] = ((df["hour"] >= 16) & (df["hour"] <= 18) & (df["is_weekend"] == 0)).astype(int)

# Lag features
for lag in [1, 2, 4, 8, 12, 16]:
    df[f"speed_lag_{lag}"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(lag)
    df[f"volume_lag_{lag}"] = df.groupby(GROUP_COLS)["volume_15min"].shift(lag)

for window in [4, 8]:
    grp = df.groupby(GROUP_COLS)
    df[f"speed_roll_mean_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    df[f"speed_roll_std_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std())
    df[f"volume_roll_mean_{window}"] = grp["volume_15min"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())

# Target: next 15-min speed
df["target_speed"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(-1)
df = df.dropna(subset=["target_speed"]).copy()

# ── 2. Split (80/10/10) ─────────────────────────────────────
df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = df_sorted.iloc[:train_end].copy()
val_df = df_sorted.iloc[train_end:val_end].copy()
test_df = df_sorted.iloc[val_end:].copy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

TARGET = "target_speed"

# ── 3. Feature Sets ──────────────────────────────────────────
# Baseline: same as notebook cell 10
FEAT_BASE = [
    "centreline_id", "longitude", "latitude",
    "volume_15min", "avg_speed_kph",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    "hour", "minute", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "speed_lag_1", "speed_lag_2", "speed_lag_4", "speed_lag_8",
    "volume_lag_1", "volume_lag_2", "volume_lag_4", "volume_lag_8",
    "speed_roll_mean_4", "speed_roll_mean_8",
    "speed_roll_std_4", "speed_roll_std_8",
    "volume_roll_mean_4", "volume_roll_mean_8",
    "direction",
]

# Enhanced: + rush hour + more lags
FEAT_ENHANCED = FEAT_BASE + [
    "is_morning_rush", "is_evening_rush",
    "speed_lag_12", "speed_lag_16",
    "volume_lag_12", "volume_lag_16",
]

results = []

# ── Helper ───────────────────────────────────────────────────
def make_preprocessor(feat_list, scale=False):
    num = [f for f in feat_list if f != "direction"]
    cat = ["direction"] if "direction" in feat_list else []
    if scale:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())])
    else:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(transformers=[
        ("num", num_pipe, num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def run_xgb(name, feat_list, xgb_params):
    pre = make_preprocessor(feat_list)
    mask_tr = train_df[feat_list + [TARGET]].notna().all(axis=1)
    mask_val = val_df[feat_list + [TARGET]].notna().all(axis=1)
    mask_te = test_df[feat_list + [TARGET]].notna().all(axis=1)

    X_tr = pre.fit_transform(train_df[mask_tr][feat_list])
    X_v = pre.transform(val_df[mask_val][feat_list])
    X_te = pre.transform(test_df[mask_te][feat_list])
    y_tr = train_df[mask_tr][TARGET]
    y_v = val_df[mask_val][TARGET]
    y_te = test_df[mask_te][TARGET]

    model = XGBRegressor(**xgb_params)
    t0 = perf_counter()
    model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=0)
    t = perf_counter() - t0

    yp = model.predict(X_te)
    mae = mean_absolute_error(y_te, yp)
    rmse = np.sqrt(mean_squared_error(y_te, yp))
    r2 = r2_score(y_te, yp)
    results.append({"step": name, "MAE": mae, "RMSE": rmse, "R2": r2, "time_s": t})
    print(f"  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  time={t:.1f}s")


# ── Step 1: XGBoost Baseline ─────────────────────────────────
print("\n--- Step 1: XGBoost Baseline ---")
run_xgb("1. XGB Baseline", FEAT_BASE, dict(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", device="cuda",
    eval_metric="rmse", early_stopping_rounds=30,
    n_jobs=-1, random_state=RANDOM_STATE))

# ── Step 2: + Rush Hour + More Lags ──────────────────────────
print("\n--- Step 2: + Rush Hour + Lag 12/16 ---")
run_xgb("2. + Features", FEAT_ENHANCED, dict(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", device="cuda",
    eval_metric="rmse", early_stopping_rounds=30,
    n_jobs=-1, random_state=RANDOM_STATE))

# ── Step 3: MAE Loss + Deeper + Lower LR ─────────────────────
print("\n--- Step 3: MAE Loss + depth=8 + lr=0.01 ---")
run_xgb("3. Tuned XGB", FEAT_ENHANCED, dict(
    n_estimators=5000, learning_rate=0.01, max_depth=8,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, objective="reg:absoluteerror",
    tree_method="hist", device="cuda",
    eval_metric="mae", early_stopping_rounds=50,
    n_jobs=-1, random_state=RANDOM_STATE))

# ── Step 4: ResNet MLP ───────────────────────────────────────
print("\n--- Step 4: ResNet MLP (GPU) ---")

pre_mlp = make_preprocessor(FEAT_ENHANCED, scale=True)
mask_tr = train_df[FEAT_ENHANCED + [TARGET]].notna().all(axis=1)
mask_val = val_df[FEAT_ENHANCED + [TARGET]].notna().all(axis=1)
mask_te = test_df[FEAT_ENHANCED + [TARGET]].notna().all(axis=1)

X_tr_np = pre_mlp.fit_transform(train_df[mask_tr][FEAT_ENHANCED]).astype(np.float32)
X_val_np = pre_mlp.transform(val_df[mask_val][FEAT_ENHANCED]).astype(np.float32)
X_te_np = pre_mlp.transform(test_df[mask_te][FEAT_ENHANCED]).astype(np.float32)
y_tr = train_df[mask_tr][TARGET].values
y_v = val_df[mask_val][TARGET].values
y_te = test_df[mask_te][TARGET].values

device = torch.device("cuda")
X_tr_t = torch.tensor(X_tr_np, device=device)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
X_val_t = torch.tensor(X_val_np, device=device)
X_te_t = torch.tensor(X_te_np, device=device)
n_feat = X_tr_t.shape[1]


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.BatchNorm1d(dim), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.SiLU(), nn.BatchNorm1d(dim))

    def forward(self, x):
        return x + self.block(x)


class ResNetMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(n_in, 512), nn.SiLU(), nn.BatchNorm1d(512), nn.Dropout(0.3))
        self.res1 = ResBlock(512, 0.2)
        self.res2 = ResBlock(512, 0.2)
        self.down = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.BatchNorm1d(256))
        self.res3 = ResBlock(256, 0.1)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.down(x)
        x = self.res3(x)
        return self.head(x)


t0 = perf_counter()
model_nn = ResNetMLP(n_feat).to(device)
optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
loss_fn = nn.L1Loss()

batch_size = 4096
best_val_mae = float("inf")
patience_counter = 0
best_epoch = 0

for epoch in range(300):
    model_nn.train()
    perm = torch.randperm(len(X_tr_t))
    for i in range(0, len(X_tr_t), batch_size):
        idx = perm[i:i + batch_size]
        pred = model_nn(X_tr_t[idx])
        loss = loss_fn(pred, y_tr_t[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_nn.parameters(), max_norm=1.0)
        optimizer.step()

    model_nn.eval()
    with torch.no_grad():
        vp = model_nn(X_val_t).cpu().numpy().flatten()
        vm = mean_absolute_error(y_v, vp)
    scheduler.step(vm)

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d} | Val MAE: {vm:.4f}")

    if vm < best_val_mae:
        best_val_mae = vm
        best_state = {k: v.clone() for k, v in model_nn.state_dict().items()}
        patience_counter = 0
        best_epoch = epoch
    else:
        patience_counter += 1
        if patience_counter >= 20:
            break

model_nn.load_state_dict(best_state)
model_nn.eval()
with torch.no_grad():
    y_pred_mlp = model_nn(X_te_t).cpu().numpy().flatten()
t_mlp = perf_counter() - t0

mae = mean_absolute_error(y_te, y_pred_mlp)
rmse = np.sqrt(mean_squared_error(y_te, y_pred_mlp))
r2 = r2_score(y_te, y_pred_mlp)
results.append({"step": "4. ResNet MLP", "MAE": mae, "RMSE": rmse, "R2": r2, "time_s": t_mlp})
print(f"  ResNet MLP: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  time={t_mlp:.1f}s  epoch={best_epoch}")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Step':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Time':>8}")
print("=" * 70)
for r in results:
    print(f"{r['step']:<22} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f} {r['time_s']:>7.1f}s")
print("=" * 70)

base_mae = results[0]["MAE"]
best_mae = min(r["MAE"] for r in results)
print(f"\nImprovement: {(base_mae - best_mae) / base_mae * 100:.1f}% MAE reduction")

pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "progression.csv"), index=False)
