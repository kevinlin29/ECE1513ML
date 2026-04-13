"""
ECE1513 Project — Model Comparison
Target: next 15-min avg_speed_kph (shift -1)
Models: Random Forest, XGBoost, LightGBM, KNN, ResNet MLP
Split: 80/10/10 chronological
Based on baseline_model_randomForest_with_training_time.ipynb
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import os
from time import perf_counter

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import torch
import torch.nn as nn

from config import MERGED_DATA, MODELS_DIR, RESULTS_DIR

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────
TARGET_HORIZON = 1   # predict 1 step (15 min) ahead
RANDOM_STATE = 42
KNN_TRAIN_MAX = 120000
KNN_N_NEIGHBORS = 15
GROUP_COLS = ["centreline_id", "direction"]

# ── 1. Load Data ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(MERGED_DATA)
df["time_start"] = pd.to_datetime(df["time_start"], errors="coerce")
df["time_end"] = pd.to_datetime(df["time_end"], errors="coerce")
df = df.dropna(subset=["time_start"])

for col in ["is_raining", "is_snowing", "is_foggy"]:
    if col in df.columns:
        df[col] = df[col].replace({True: 1, False: 0}).fillna(0).astype(int)

num_cols = [
    "longitude", "latitude", "volume_15min", "avg_speed_kph",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos",
]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(GROUP_COLS + ["time_start"]).reset_index(drop=True)
print(f"Total rows: {len(df)}")

# ── 2. Time Features ─────────────────────────────────────────
df["hour"] = df["time_start"].dt.hour
df["minute"] = df["time_start"].dt.minute
df["day_of_week"] = df["time_start"].dt.dayofweek
df["month"] = df["time_start"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# ── 3. Lag & Rolling Features ────────────────────────────────
lag_steps = [1, 2, 4, 8]
rolling_windows = [4, 8]

for lag in lag_steps:
    df[f"speed_lag_{lag}"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(lag)
    df[f"volume_lag_{lag}"] = df.groupby(GROUP_COLS)["volume_15min"].shift(lag)

for window in rolling_windows:
    grp = df.groupby(GROUP_COLS)
    df[f"speed_roll_mean_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df[f"speed_roll_std_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )
    df[f"volume_roll_mean_{window}"] = grp["volume_15min"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# ── 4. Target: next 15-min speed ─────────────────────────────
df["target_speed"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(-TARGET_HORIZON)
df = df.dropna(subset=["target_speed"]).copy()

# ── 5. Feature List ──────────────────────────────────────────
numeric_features = [
    "centreline_id", "longitude", "latitude",
    "volume_15min", "avg_speed_kph",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    "hour", "minute", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
] + [f"speed_lag_{l}" for l in lag_steps] \
  + [f"volume_lag_{l}" for l in lag_steps] \
  + [f"speed_roll_mean_{w}" for w in rolling_windows] \
  + [f"speed_roll_std_{w}" for w in rolling_windows] \
  + [f"volume_roll_mean_{w}" for w in rolling_windows]

numeric_features = [c for c in numeric_features if c in df.columns]
categorical_features = ["direction"]
all_features = numeric_features + categorical_features

print(f"Features: {len(all_features)} ({len(numeric_features)} numeric + {len(categorical_features)} categorical)")

# ── 6. Split (80/10/10) ─────────────────────────────────────
model_df = df[["time_start", "target_speed"] + all_features].copy()
model_df = model_df.sort_values("time_start").reset_index(drop=True)

n = len(model_df)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = model_df.iloc[:train_end].copy()
val_df = model_df.iloc[train_end:val_end].copy()
test_df = model_df.iloc[val_end:].copy()

X_train, y_train = train_df[all_features], train_df["target_speed"]
X_val, y_val = val_df[all_features], val_df["target_speed"]
X_test, y_test = test_df[all_features], test_df["target_speed"]

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train: {train_df['time_start'].min()} to {train_df['time_start'].max()}")
print(f"Val:   {val_df['time_start'].min()} to {val_df['time_start'].max()}")
print(f"Test:  {test_df['time_start'].min()} to {test_df['time_start'].max()}")

# ── 7. Preprocessors ─────────────────────────────────────────
tree_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

knn_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

results = []


def evaluate(y_true, y_pred, name, train_time):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "time_s": train_time})
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  Time={train_time:.1f}s")


# ── 8. Random Forest (small, fast) ────────────────────────────
print("\n" + "=" * 50)
print(" RANDOM FOREST")
print("=" * 50)

rf_pipe = Pipeline([
    ("preprocessor", tree_preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE))
])

t0 = perf_counter()
rf_pipe.fit(X_train, y_train)
t_rf = perf_counter() - t0

evaluate(y_test, rf_pipe.predict(X_test), "Random Forest", t_rf)

# ── 9. XGBoost ───────────────────────────────────────────────
print("\n" + "=" * 50)
print(" XGBOOST")
print("=" * 50)

X_tr_tree = tree_preprocessor.fit_transform(X_train)
X_val_tree = tree_preprocessor.transform(X_val)
X_te_tree = tree_preprocessor.transform(X_test)

xgb = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=8,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    device="cuda", eval_metric="rmse", early_stopping_rounds=30,
    n_jobs=-1, random_state=RANDOM_STATE,
)

t0 = perf_counter()
xgb.fit(X_tr_tree, y_train, eval_set=[(X_val_tree, y_val)], verbose=100)
t_xgb = perf_counter() - t0

evaluate(y_test, xgb.predict(X_te_tree), "XGBoost", t_xgb)

# ── 10. LightGBM ─────────────────────────────────────────────
print("\n" + "=" * 50)
print(" LIGHTGBM")
print("=" * 50)

lgbm = LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=8,
    num_leaves=64, subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)

t0 = perf_counter()
lgbm.fit(X_tr_tree, y_train, eval_set=[(X_val_tree, y_val)])
t_lgbm = perf_counter() - t0

evaluate(y_test, lgbm.predict(X_te_tree), "LightGBM", t_lgbm)

# ── 11. KNN ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print(" KNN")
print("=" * 50)

X_tr_knn = knn_preprocessor.fit_transform(X_train)
X_te_knn = knn_preprocessor.transform(X_test)

# Subsample for KNN (too slow on full data)
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(X_tr_knn.shape[0], size=min(KNN_TRAIN_MAX, X_tr_knn.shape[0]), replace=False)
X_tr_knn_sub = X_tr_knn[idx]
y_tr_knn_sub = y_train.iloc[idx]
print(f"  KNN using {len(y_tr_knn_sub)} train samples (subsampled)")

knn = KNeighborsRegressor(n_neighbors=KNN_N_NEIGHBORS, weights="distance", n_jobs=-1)

t0 = perf_counter()
knn.fit(X_tr_knn_sub, y_tr_knn_sub)
t_knn = perf_counter() - t0

evaluate(y_test, knn.predict(X_te_knn), "KNN", t_knn)

# ── 12. ResNet MLP ───────────────────────────────────────────
print("\n" + "=" * 50)
print(" RESNET MLP (GPU)")
print("=" * 50)

# Use the knn_preprocessor (has scaler) for the MLP
mlp_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

X_tr_mlp = mlp_preprocessor.fit_transform(X_train).astype(np.float32)
X_val_mlp = mlp_preprocessor.transform(X_val).astype(np.float32)
X_te_mlp = mlp_preprocessor.transform(X_test).astype(np.float32)

device = torch.device("cuda")
X_tr_t = torch.tensor(X_tr_mlp, device=device)
y_tr_t = torch.tensor(y_train.values, dtype=torch.float32, device=device).unsqueeze(1)
X_val_t = torch.tensor(X_val_mlp, device=device)
X_te_t = torch.tensor(X_te_mlp, device=device)

n_feat = X_tr_t.shape[1]


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.BatchNorm1d(dim), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.SiLU(), nn.BatchNorm1d(dim),
        )

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
        val_pred = model_nn(X_val_t).cpu().numpy().flatten()
        val_mae = mean_absolute_error(y_val.values, val_pred)
    scheduler.step(val_mae)

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d} | Val MAE: {val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
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
print(f"  Best epoch: {best_epoch}")

evaluate(y_test, y_pred_mlp, "ResNet MLP", t_mlp)

torch.save(best_state, os.path.join(MODELS_DIR, "resnet_mlp.pt"))

# ── 13. Summary ──────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'Model':<18} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Time':>10}")
print("=" * 65)
for r in results:
    print(f"{r['model']:<18} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f} {r['time_s']:>9.1f}s")
print("=" * 65)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
print(f"\nSaved to {RESULTS_DIR}/model_comparison.csv")
