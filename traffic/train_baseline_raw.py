"""
ECE1513 — XGBoost with ZERO feature engineering (raw columns only)
Target: next 15-min avg_speed_kph
Split: 80/10/10 chronological
This is the true baseline — no lags, no rolling, no time encoding.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from time import perf_counter

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import MERGED_DATA, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42
GROUP_COLS = ["centreline_id", "direction"]

# ── 1. Load raw data ─────────────────────────────────────────
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

# Only extract hour/dow from timestamp (minimal, not "engineering")
df["hour"] = df["time_start"].dt.hour
df["day_of_week"] = df["time_start"].dt.dayofweek
df["month"] = df["time_start"].dt.month

# Target: next 15-min speed
df["target_speed"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(-1)
df = df.dropna(subset=["target_speed"]).copy()

# ── 2. Raw features only — no lags, no rolling, no encoding ──
RAW_FEATURES = [
    "centreline_id", "longitude", "latitude",
    "volume_15min", "avg_speed_kph",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    "hour", "day_of_week", "month",
    "direction",
]

TARGET = "target_speed"

# ── 3. Split (80/10/10) ─────────────────────────────────────
df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = df_sorted.iloc[:train_end]
val_df = df_sorted.iloc[train_end:val_end]
test_df = df_sorted.iloc[val_end:]

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Features: {len(RAW_FEATURES)} (raw only, zero engineering)")

# ── 4. Preprocessor ──────────────────────────────────────────
num_feats = [f for f in RAW_FEATURES if f != "direction"]
cat_feats = ["direction"]

pre = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_feats),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])

mask_tr = train_df[RAW_FEATURES + [TARGET]].notna().all(axis=1)
mask_val = val_df[RAW_FEATURES + [TARGET]].notna().all(axis=1)
mask_te = test_df[RAW_FEATURES + [TARGET]].notna().all(axis=1)

X_train = pre.fit_transform(train_df[mask_tr][RAW_FEATURES])
X_val = pre.transform(val_df[mask_val][RAW_FEATURES])
X_test = pre.transform(test_df[mask_te][RAW_FEATURES])
y_train = train_df[mask_tr][TARGET].values
y_val = val_df[mask_val][TARGET].values
y_test = test_df[mask_te][TARGET].values

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn

results = []


def evaluate(name, y_true, y_pred, t):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "time_s": t})
    print(f"  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  time={t:.1f}s")


# ── 5a. XGBoost ──────────────────────────────────────────────
print("\n--- XGBoost ---")
xgb = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", device="cuda",
    eval_metric="rmse", early_stopping_rounds=30,
    n_jobs=-1, random_state=RANDOM_STATE,
)
t0 = perf_counter()
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
evaluate("XGBoost", y_test, xgb.predict(X_test), perf_counter() - t0)

# ── 5b. LightGBM ────────────────────────────────────────────
print("\n--- LightGBM ---")
lgbm = LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
t0 = perf_counter()
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
evaluate("LightGBM", y_test, lgbm.predict(X_test), perf_counter() - t0)

# ── 5c. KNN (subsampled) ────────────────────────────────────
print("\n--- KNN ---")
# Need scaled data for KNN
pre_knn = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]), num_feats),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
X_tr_knn = pre_knn.fit_transform(train_df[mask_tr][RAW_FEATURES])
X_te_knn = pre_knn.transform(test_df[mask_te][RAW_FEATURES])

KNN_MAX = 120000
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(X_tr_knn.shape[0], size=min(KNN_MAX, X_tr_knn.shape[0]), replace=False)

knn = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
t0 = perf_counter()
knn.fit(X_tr_knn[idx], y_train[idx])
evaluate("KNN (k=15)", y_test, knn.predict(X_te_knn), perf_counter() - t0)

# ── 5d. ResNet MLP ───────────────────────────────────────────
print("\n--- ResNet MLP ---")
pre_mlp = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]), num_feats),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
X_tr_mlp = pre_mlp.fit_transform(train_df[mask_tr][RAW_FEATURES]).astype(np.float32)
X_val_mlp = pre_mlp.transform(val_df[mask_val][RAW_FEATURES]).astype(np.float32)
X_te_mlp = pre_mlp.transform(test_df[mask_te][RAW_FEATURES]).astype(np.float32)

device = torch.device("cuda")
X_tr_t = torch.tensor(X_tr_mlp, device=device)
y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
X_val_t = torch.tensor(X_val_mlp, device=device)
X_te_t = torch.tensor(X_te_mlp, device=device)
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
        vm = mean_absolute_error(y_val, vp)
    scheduler.step(vm)

    if epoch % 20 == 0:
        print(f"    Epoch {epoch:3d} | Val MAE: {vm:.4f}")

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

evaluate("ResNet MLP", y_test, y_pred_mlp, perf_counter() - t0)

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  RAW BASELINE — {len(RAW_FEATURES)} features, zero engineering")
print("=" * 65)
print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Time':>10}")
print("-" * 65)
for r in results:
    print(f"{r['model']:<15} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f} {r['time_s']:>9.1f}s")
print("=" * 65)

pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "raw_baseline_comparison.csv"), index=False)
