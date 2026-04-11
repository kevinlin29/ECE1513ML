"""
ECE1513 — ResNet MLP Traffic Speed Prediction

Target: speed_deviation (diff from historical avg)
Historical avg computed on TRAIN data only (no leakage).
Deviation lags computed AFTER applying train-only avg to all data.

Run: python train.py
"""
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from config import MERGED_DATA

# ── 1. Load Data ─────────────────────────────────────────────
df = pd.read_csv(MERGED_DATA)
df["time_start"] = pd.to_datetime(df["time_start"])
df["time_end"] = pd.to_datetime(df["time_end"])
df = df.sort_values(["centreline_id", "direction", "time_start"]).reset_index(drop=True)

# Filter low-quality locations (< 72 hours of data)
hour_trunc = df["time_start"].dt.floor("h")
hours_per_loc = hour_trunc.groupby(df["centreline_id"]).nunique()
valid_locs = hours_per_loc[hours_per_loc >= 72].index
df = df[df["centreline_id"].isin(valid_locs)].reset_index(drop=True)

# ── 2. Time Features ─────────────────────────────────────────
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

# ── 3. Identify train rows BEFORE computing hist avg ─────────
df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.70)

train_rows = df_sorted.iloc[:train_end]

# ── 4. Historical avg from TRAIN ONLY ────────────────────────
hist_keys = ["centreline_id", "direction", "hour", "dayofweek"]
hist_speed_map = train_rows.groupby(hist_keys)["avg_speed_kph"].mean()
hist_volume_map = train_rows.groupby(hist_keys)["volume_15min"].mean()
global_speed_mean = train_rows["avg_speed_kph"].mean()
global_volume_mean = train_rows["volume_15min"].mean()

# Apply train-only hist avg to ALL data
idx = pd.MultiIndex.from_arrays([df_sorted[k] for k in hist_keys])
df_sorted["hist_avg_speed"] = idx.map(hist_speed_map).fillna(global_speed_mean).values
df_sorted["hist_avg_volume"] = idx.map(hist_volume_map).fillna(global_volume_mean).values

# ── 5. Deviation (using train-only hist avg) ─────────────────
df_sorted["speed_deviation"] = df_sorted["avg_speed_kph"] - df_sorted["hist_avg_speed"]
df_sorted["volume_deviation"] = df_sorted["volume_15min"] - df_sorted["hist_avg_volume"]

# Re-sort by segment+time for correct lag computation
df_sorted = df_sorted.sort_values(["centreline_id", "direction", "time_start"]).reset_index(drop=True)

# ── 6. Lag Features ──────────────────────────────────────────
group_keys = ["centreline_id", "direction"]

# Deviation lags (core signal)
for lag in [1, 2, 4, 8, 12, 16]:
    df_sorted[f"speed_dev_lag_{lag}"] = df_sorted.groupby(group_keys)["speed_deviation"].shift(lag)
    df_sorted[f"volume_dev_lag_{lag}"] = df_sorted.groupby(group_keys)["volume_deviation"].shift(lag)

# Raw speed lags
for lag in [2, 4, 8]:
    df_sorted[f"speed_lag_{lag}"] = df_sorted.groupby(group_keys)["avg_speed_kph"].shift(lag)
    df_sorted[f"volume_lag_{lag}"] = df_sorted.groupby(group_keys)["volume_15min"].shift(lag)

# ── 7. Rolling Window Features ────────────────────────────────
for window in [4, 8]:
    grp = df_sorted.groupby(group_keys)
    df_sorted[f"speed_dev_roll_mean_{window}"] = grp["speed_deviation"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_sorted[f"speed_dev_roll_std_{window}"] = grp["speed_deviation"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )
    df_sorted[f"volume_dev_roll_mean_{window}"] = grp["volume_deviation"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_sorted[f"speed_roll_mean_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(2).rolling(window, min_periods=1).mean()
    )

grp = df_sorted.groupby(group_keys)
df_sorted["speed_dev_roll_mean_16"] = grp["speed_deviation"].transform(
    lambda x: x.shift(1).rolling(16, min_periods=1).mean()
)
df_sorted["speed_roll_mean_16"] = grp["avg_speed_kph"].transform(
    lambda x: x.shift(2).rolling(16, min_periods=1).mean()
)

# Interaction
df_sorted["speed_lag1_x_volume"] = df_sorted.groupby(group_keys)["avg_speed_kph"].shift(1) * df_sorted["volume_15min"]

# ── 8. Feature / Target Definition ───────────────────────────
TARGET = "speed_deviation"

FEATURES = [
    # Location
    "longitude", "latitude", "centreline_id",
    # Time
    "hour", "dayofweek", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_morning_rush", "is_evening_rush",
    # Weather
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)",
    "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    # Historical context (train-only, no leakage)
    "hist_avg_speed", "hist_avg_volume",
    # Deviation lags
    "speed_dev_lag_1", "speed_dev_lag_2", "speed_dev_lag_4", "speed_dev_lag_8",
    "speed_dev_lag_12", "speed_dev_lag_16",
    "volume_dev_lag_1", "volume_dev_lag_2", "volume_dev_lag_4", "volume_dev_lag_8",
    "volume_dev_lag_12", "volume_dev_lag_16",
    # Deviation rolling stats
    "speed_dev_roll_mean_4", "speed_dev_roll_mean_8", "speed_dev_roll_mean_16",
    "speed_dev_roll_std_4", "speed_dev_roll_std_8",
    "volume_dev_roll_mean_4", "volume_dev_roll_mean_8",
    # Raw lags (>= 2)
    "speed_lag_2", "speed_lag_4", "speed_lag_8",
    "volume_lag_2", "volume_lag_4", "volume_lag_8",
    # Raw rolling (shifted by 2)
    "speed_roll_mean_4", "speed_roll_mean_8", "speed_roll_mean_16",
    # Interaction
    "speed_lag1_x_volume",
    # Current volume
    "volume_15min",
]

# ── 9. Drop NaN and re-split chronologically ─────────────────
df_final = df_sorted.dropna(subset=FEATURES + [TARGET]).copy()
df_final = df_final.sort_values("time_start").reset_index(drop=True)

n = len(df_final)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

train_df = df_final.iloc[:train_end]
val_df = df_final.iloc[train_end:val_end]
test_df = df_final.iloc[val_end:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_val = val_df[FEATURES]
y_val = val_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Features: {len(FEATURES)}")

# ── 10. Train ResNet MLP ─────────────────────────────────────
t0 = time.time()

device = torch.device("cuda")

scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train.values)
X_val_np = scaler.transform(X_val.values)
X_test_np = scaler.transform(X_test.values)

X_train_t = torch.tensor(X_train_np, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32, device=device).unsqueeze(1)
X_val_t = torch.tensor(X_val_np, dtype=torch.float32, device=device)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32, device=device)

n_features = X_train_t.shape[1]


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


model_nn = MLP(n_features).to(device)
optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
loss_fn = nn.L1Loss()

batch_size = 4096
best_val_mae = float("inf")
patience_counter = 0
max_patience = 20

for epoch in range(300):
    model_nn.train()
    perm = torch.randperm(len(X_train_t))
    for i in range(0, len(X_train_t), batch_size):
        idx = perm[i:i + batch_size]
        pred = model_nn(X_train_t[idx])
        loss = loss_fn(pred, y_train_t[idx])
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
        if patience_counter >= max_patience:
            break

model_nn.load_state_dict(best_state)
model_nn.eval()
with torch.no_grad():
    y_pred_dev = model_nn(X_test_t).cpu().numpy().flatten()

train_time = time.time() - t0

# ── 11. Evaluate (convert deviation back to speed) ───────────
y_pred_speed = y_pred_dev + test_df["hist_avg_speed"].values
y_true_speed = test_df["avg_speed_kph"].values

mae = mean_absolute_error(y_true_speed, y_pred_speed)
rmse = np.sqrt(mean_squared_error(y_true_speed, y_pred_speed))
r2 = r2_score(y_true_speed, y_pred_speed)

print("\n---")
print(f"test_mae:     {mae:.6f}")
print(f"test_rmse:    {rmse:.6f}")
print(f"test_r2:      {r2:.6f}")
print(f"train_time_s: {train_time:.1f}")
print(f"best_epoch:   {epoch + 1}")
print(f"n_features:   {len(FEATURES)}")
print(f"n_train:      {len(train_df)}")
print(f"n_test:       {len(test_df)}")
print("---")
