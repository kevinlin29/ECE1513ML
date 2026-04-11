"""
ECE1513 — Classification metrics for speed prediction models.
Bins continuous speed predictions into congestion levels,
then computes precision, recall, F1 per class.

Run after train_all.py or train_progression.py to evaluate saved predictions.
Can also be imported and called directly.
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
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, f1_score,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
import torch
import torch.nn as nn

from config import MERGED_DATA, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42
GROUP_COLS = ["centreline_id", "direction"]

# ── Congestion level bins ────────────────────────────────────
BINS = [0, 20, 35, 50, np.inf]
LABELS = ["Congested", "Slow", "Normal", "Fast"]


def speed_to_class(speeds):
    return pd.cut(speeds, bins=BINS, labels=LABELS, right=False)


# ── 1. Prepare data (same pipeline as train_all.py) ──────────
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

df["target_speed"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(-1)
df = df.dropna(subset=["target_speed"]).copy()

df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = df_sorted.iloc[:train_end]
val_df = df_sorted.iloc[train_end:val_end]
test_df = df_sorted.iloc[val_end:]

TARGET = "target_speed"
FEATURES = [
    "centreline_id", "longitude", "latitude",
    "volume_15min", "avg_speed_kph",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    "hour", "minute", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_morning_rush", "is_evening_rush",
    "speed_lag_1", "speed_lag_2", "speed_lag_4", "speed_lag_8",
    "speed_lag_12", "speed_lag_16",
    "volume_lag_1", "volume_lag_2", "volume_lag_4", "volume_lag_8",
    "volume_lag_12", "volume_lag_16",
    "speed_roll_mean_4", "speed_roll_mean_8",
    "speed_roll_std_4", "speed_roll_std_8",
    "volume_roll_mean_4", "volume_roll_mean_8",
    "direction",
]

num_feats = [f for f in FEATURES if f != "direction"]
cat_feats = ["direction"]

# Tree preprocessor
tree_pre = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_feats),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])

# Scaled preprocessor (for KNN + MLP)
scaled_pre = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]), num_feats),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])

mask_tr = train_df[FEATURES + [TARGET]].notna().all(axis=1)
mask_val = val_df[FEATURES + [TARGET]].notna().all(axis=1)
mask_te = test_df[FEATURES + [TARGET]].notna().all(axis=1)

X_tr = tree_pre.fit_transform(train_df[mask_tr][FEATURES])
X_val = tree_pre.transform(val_df[mask_val][FEATURES])
X_te = tree_pre.transform(test_df[mask_te][FEATURES])
y_tr = train_df[mask_tr][TARGET].values
y_v = val_df[mask_val][TARGET].values
y_te = test_df[mask_te][TARGET].values

X_tr_s = scaled_pre.fit_transform(train_df[mask_tr][FEATURES])
X_val_s = scaled_pre.transform(val_df[mask_val][FEATURES])
X_te_s = scaled_pre.transform(test_df[mask_te][FEATURES])

y_true_class = speed_to_class(y_te)

print(f"Train: {len(y_tr)} | Val: {len(y_v)} | Test: {len(y_te)}")
print(f"\nTest set class distribution:")
print(y_true_class.value_counts().sort_index())

# ── 2. Train & evaluate each model ───────────────────────────
all_results = []


def eval_model(name, y_pred):
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)

    y_pred_class = speed_to_class(y_pred)
    f1_macro = f1_score(y_true_class, y_pred_class, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_class, y_pred_class, average="weighted", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f" {name}")
    print(f"{'=' * 60}")
    print(f"  Regression: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    print(f"  Classification: F1-macro={f1_macro:.4f}  F1-weighted={f1_weighted:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true_class, y_pred_class, zero_division=0))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_true_class, y_pred_class, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"True:{l}" for l in LABELS], columns=[f"Pred:{l}" for l in LABELS])
    print(cm_df.to_string())

    all_results.append({
        "model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
        "F1_macro": f1_macro, "F1_weighted": f1_weighted,
    })


# ── XGBoost ──────────────────────────────────────────────────
print("\nTraining XGBoost...")
xgb = XGBRegressor(
    n_estimators=5000, learning_rate=0.01, max_depth=8,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, objective="reg:absoluteerror",
    tree_method="hist", device="cuda",
    eval_metric="mae", early_stopping_rounds=50,
    n_jobs=-1, random_state=RANDOM_STATE,
)
xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_v)], verbose=0)
eval_model("XGBoost (tuned)", xgb.predict(X_te))

# ── LightGBM ─────────────────────────────────────────────────
print("\nTraining LightGBM...")
lgbm = LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=8,
    num_leaves=64, subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_v)])
eval_model("LightGBM", lgbm.predict(X_te))

# ── KNN ──────────────────────────────────────────────────────
print("\nTraining KNN...")
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(X_tr_s.shape[0], size=min(120000, X_tr_s.shape[0]), replace=False)
knn = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
knn.fit(X_tr_s[idx], y_tr[idx])
eval_model("KNN (k=15)", knn.predict(X_te_s))

# ── ResNet MLP ───────────────────────────────────────────────
print("\nTraining ResNet MLP...")
device = torch.device("cuda")
X_tr_t = torch.tensor(X_tr_s.astype(np.float32), device=device)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
X_val_t = torch.tensor(X_val_s.astype(np.float32), device=device)
X_te_t = torch.tensor(X_te_s.astype(np.float32), device=device)
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


model_nn = ResNetMLP(n_feat).to(device)
optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
loss_fn = nn.L1Loss()

best_val_mae = float("inf")
patience_counter = 0

for epoch in range(300):
    model_nn.train()
    perm = torch.randperm(len(X_tr_t))
    for i in range(0, len(X_tr_t), 4096):
        idx = perm[i:i + 4096]
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

    if vm < best_val_mae:
        best_val_mae = vm
        best_state = {k: v.clone() for k, v in model_nn.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 20:
            break

model_nn.load_state_dict(best_state)
model_nn.eval()
with torch.no_grad():
    y_pred_mlp = model_nn(X_te_t).cpu().numpy().flatten()

eval_model("ResNet MLP", y_pred_mlp)

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'Model':<18} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'F1-macro':>9} {'F1-wt':>7}")
print("=" * 80)
for r in all_results:
    print(f"{r['model']:<18} {r['MAE']:>7.4f} {r['RMSE']:>7.4f} {r['R2']:>7.4f} {r['F1_macro']:>9.4f} {r['F1_weighted']:>7.4f}")
print("=" * 80)

pd.DataFrame(all_results).to_csv(os.path.join(RESULTS_DIR, "classification_metrics.csv"), index=False)
print(f"\nSaved to {RESULTS_DIR}/classification_metrics.csv")
