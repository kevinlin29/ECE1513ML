"""Quick benchmark: training time for each model type."""
import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from config import MERGED_DATA

# ── Load + feature engineer (same as train.py) ───────────────
df = pd.read_csv(MERGED_DATA)
df["time_start"] = pd.to_datetime(df["time_start"])
df["time_end"] = pd.to_datetime(df["time_end"])
df = df.sort_values(["centreline_id", "direction", "time_start"]).reset_index(drop=True)

hour_trunc = df["time_start"].dt.floor("h")
hours_per_loc = hour_trunc.groupby(df["centreline_id"]).nunique()
valid_locs = hours_per_loc[hours_per_loc >= 72].index
df = df[df["centreline_id"].isin(valid_locs)].reset_index(drop=True)

df["hour"] = df["time_start"].dt.hour
df["dayofweek"] = df["time_start"].dt.dayofweek
df["month"] = df["time_start"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 9) & (df["is_weekend"] == 0)).astype(int)
df["is_evening_rush"] = ((df["hour"] >= 16) & (df["hour"] <= 18) & (df["is_weekend"] == 0)).astype(int)

historical_avg = df.groupby(["centreline_id", "direction", "hour", "dayofweek"])["avg_speed_kph"].transform("mean")
df["speed_deviation"] = df["avg_speed_kph"] - historical_avg
historical_vol_avg = df.groupby(["centreline_id", "direction", "hour", "dayofweek"])["volume_15min"].transform("mean")
df["volume_deviation"] = df["volume_15min"] - historical_vol_avg

group_keys = ["centreline_id", "direction"]
for lag in [1, 2, 4, 8, 12, 16]:
    df[f"speed_dev_lag_{lag}"] = df.groupby(group_keys)["speed_deviation"].shift(lag)
    df[f"volume_dev_lag_{lag}"] = df.groupby(group_keys)["volume_deviation"].shift(lag)
for lag in [2, 4, 8, 12, 16]:
    df[f"speed_lag_{lag}"] = df.groupby(group_keys)["avg_speed_kph"].shift(lag)
    df[f"volume_lag_{lag}"] = df.groupby(group_keys)["volume_15min"].shift(lag)

for window in [4, 8]:
    grp = df.groupby(group_keys)
    df[f"speed_dev_roll_mean_{window}"] = grp["speed_deviation"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    df[f"speed_dev_roll_std_{window}"] = grp["speed_deviation"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
    df[f"volume_dev_roll_mean_{window}"] = grp["volume_deviation"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    df[f"speed_roll_mean_{window}"] = grp["avg_speed_kph"].transform(lambda x: x.shift(2).rolling(window, min_periods=1).mean())

grp = df.groupby(group_keys)
df["speed_dev_roll_mean_16"] = grp["speed_deviation"].transform(lambda x: x.shift(1).rolling(16, min_periods=1).mean())
df["speed_roll_mean_16"] = grp["avg_speed_kph"].transform(lambda x: x.shift(2).rolling(16, min_periods=1).mean())

df["hist_avg_speed"] = historical_avg
df["hist_avg_volume"] = historical_vol_avg
df["speed_lag1_x_volume"] = df.groupby(group_keys)["avg_speed_kph"].shift(1) * df["volume_15min"]

TARGET = "speed_deviation"
FEATURES = [
    "longitude", "latitude", "centreline_id",
    "hour", "dayofweek", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_morning_rush", "is_evening_rush",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    "hist_avg_speed", "hist_avg_volume",
    "speed_dev_lag_1", "speed_dev_lag_2", "speed_dev_lag_4", "speed_dev_lag_8",
    "speed_dev_lag_12", "speed_dev_lag_16",
    "volume_dev_lag_1", "volume_dev_lag_2", "volume_dev_lag_4", "volume_dev_lag_8",
    "volume_dev_lag_12", "volume_dev_lag_16",
    "speed_dev_roll_mean_4", "speed_dev_roll_mean_8",
    "speed_dev_roll_std_4", "speed_dev_roll_std_8",
    "volume_dev_roll_mean_4", "volume_dev_roll_mean_8",
    "speed_lag_2", "speed_lag_4", "speed_lag_8", "speed_lag_12", "speed_lag_16",
    "volume_lag_2", "volume_lag_4", "volume_lag_8", "volume_lag_12", "volume_lag_16",
    "speed_roll_mean_4", "speed_roll_mean_8", "speed_roll_mean_16",
    "speed_dev_roll_mean_16",
    "speed_lag1_x_volume",
    "volume_15min",
]

df_model = df.dropna(subset=FEATURES + [TARGET]).copy()
n = len(df_model)
sorted_df = df_model.sort_values("time_start").reset_index(drop=True)
train_end = int(n * 0.70)
val_end = int(n * 0.80)
train_df = sorted_df.iloc[:train_end]
val_df = sorted_df.iloc[train_end:val_end]
test_df = sorted_df.iloc[val_end:]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val, y_val = val_df[FEATURES], val_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

def evaluate(name, y_true_dev, y_pred_dev):
    y_pred_speed = y_pred_dev + test_df["hist_avg_speed"].values
    y_true_speed = test_df["avg_speed_kph"].values
    mae = mean_absolute_error(y_true_speed, y_pred_speed)
    rmse = np.sqrt(mean_squared_error(y_true_speed, y_pred_speed))
    r2 = r2_score(y_true_speed, y_pred_speed)
    return mae, rmse, r2

results = []

# ── KNN ───────────────────────────────────────────────────────
print("\nKNN...")
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train.values)
X_te_s = scaler.transform(X_test.values)
t0 = time.time()
knn = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
knn.fit(X_tr_s, y_train)
y_pred = knn.predict(X_te_s)
t_knn = time.time() - t0
mae, rmse, r2 = evaluate("KNN", y_test, y_pred)
results.append(("KNN (k=10)", mae, rmse, r2, t_knn))
print(f"  MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} time={t_knn:.1f}s")

# ── XGBoost baseline ──────────────────────────────────────────
print("XGBoost baseline...")
t0 = time.time()
xgb_base = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
    tree_method="hist", device="cuda", early_stopping_rounds=30, eval_metric="rmse")
xgb_base.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
y_pred = xgb_base.predict(X_test)
t_xgb_base = time.time() - t0
mae, rmse, r2 = evaluate("XGB base", y_test, y_pred)
results.append(("XGBoost Baseline", mae, rmse, r2, t_xgb_base))
print(f"  MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} time={t_xgb_base:.1f}s")

# ── XGBoost tuned ─────────────────────────────────────────────
print("XGBoost tuned...")
t0 = time.time()
xgb_tuned = XGBRegressor(n_estimators=15000, learning_rate=0.008, max_depth=8,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, objective="reg:absoluteerror",
    random_state=42, n_jobs=-1, tree_method="hist", device="cuda",
    early_stopping_rounds=50, eval_metric="mae")
xgb_tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
y_pred = xgb_tuned.predict(X_test)
t_xgb_tuned = time.time() - t0
mae, rmse, r2 = evaluate("XGB tuned", y_test, y_pred)
results.append(("XGBoost Tuned", mae, rmse, r2, t_xgb_tuned))
print(f"  MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} time={t_xgb_tuned:.1f}s")

# ── ResNet MLP ────────────────────────────────────────────────
print("ResNet MLP...")
import torch
import torch.nn as nn

device = torch.device("cuda")
scaler2 = StandardScaler()
X_tr_np = scaler2.fit_transform(X_train.values)
X_val_np = scaler2.transform(X_val.values)
X_te_np = scaler2.transform(X_test.values)

X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32, device=device)
y_tr_t = torch.tensor(y_train.values, dtype=torch.float32, device=device).unsqueeze(1)
X_val_t = torch.tensor(X_val_np, dtype=torch.float32, device=device)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32, device=device).unsqueeze(1)
X_te_t = torch.tensor(X_te_np, dtype=torch.float32, device=device)

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

class MLP(nn.Module):
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

t0 = time.time()
model_nn = MLP(n_feat).to(device)
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
        idx = perm[i:i+batch_size]
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
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_state = {k: v.clone() for k, v in model_nn.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 20:
            break

model_nn.load_state_dict(best_state)
model_nn.eval()
with torch.no_grad():
    y_pred = model_nn(X_te_t).cpu().numpy().flatten()
t_mlp = time.time() - t0
mae, rmse, r2 = evaluate("ResNet MLP", y_test, y_pred)
results.append(("ResNet MLP (final)", mae, rmse, r2, t_mlp))
print(f"  MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} time={t_mlp:.1f}s epochs={epoch+1}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'R2':>7} {'Time (s)':>10}")
print("=" * 65)
for name, mae, rmse, r2, t in results:
    print(f"{name:<22} {mae:>7.3f} {rmse:>7.3f} {r2:>7.3f} {t:>10.1f}")
print("=" * 65)
