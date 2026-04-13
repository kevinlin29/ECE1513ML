"""
ECE1513 — ResNet MLP for traffic speed prediction
Approach: Baseline + Residual Learning
  1. Compute historical avg speed per (segment, direction, hour, dow) from train set
  2. Train ResNet to predict the RESIDUAL (actual - historical avg)
  3. Final prediction = historical avg + predicted residual
Target: next 15-min avg_speed_kph
Split: 80/10/10 chronological
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from time import perf_counter

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Lag features (on raw speed/volume)
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
# Also need next step's time features for historical avg lookup
df["target_hour"] = df.groupby(GROUP_COLS)["hour"].shift(-1)
df["target_dow"] = df.groupby(GROUP_COLS)["day_of_week"].shift(-1)
df = df.dropna(subset=["target_speed", "target_hour", "target_dow"]).copy()
df["target_hour"] = df["target_hour"].astype(int)
df["target_dow"] = df["target_dow"].astype(int)

# ── 2. Split (80/10/10) ─────────────────────────────────────
df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = df_sorted.iloc[:train_end].copy()
val_df = df_sorted.iloc[train_end:val_end].copy()
test_df = df_sorted.iloc[val_end:].copy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── 3. Historical averages (train set only — no leakage) ────
print("Computing historical averages from training set...")
hist_key = ["centreline_id", "direction", "hour", "day_of_week"]

hist_speed = train_df.groupby(hist_key)["avg_speed_kph"].mean()
hist_volume = train_df.groupby(hist_key)["volume_15min"].mean()
global_mean_speed = train_df["avg_speed_kph"].mean()
global_mean_volume = train_df["volume_15min"].mean()

def map_hist_avg(split_df, col_name, hist_series, fallback, hour_col="hour", dow_col="day_of_week"):
    """Map historical average to rows using given hour/dow columns."""
    keys = list(zip(split_df["centreline_id"], split_df["direction"],
                    split_df[hour_col], split_df[dow_col]))
    vals = [hist_series.get(k, fallback) for k in keys]
    split_df[col_name] = vals

# Current-step historical averages (features)
for split_df in [train_df, val_df, test_df]:
    map_hist_avg(split_df, "hist_avg_speed", hist_speed, global_mean_speed)
    map_hist_avg(split_df, "hist_avg_volume", hist_volume, global_mean_volume)

# Target-step historical average (for residual computation)
for split_df in [train_df, val_df, test_df]:
    map_hist_avg(split_df, "target_hist_avg_speed", hist_speed, global_mean_speed,
                 hour_col="target_hour", dow_col="target_dow")

# Speed/volume deviation from current-step historical average
for split_df in [train_df, val_df, test_df]:
    split_df["speed_deviation"] = split_df["avg_speed_kph"] - split_df["hist_avg_speed"]
    split_df["volume_deviation"] = split_df["volume_15min"] - split_df["hist_avg_volume"]

# Deviation lags
for split_df in [train_df, val_df, test_df]:
    s = split_df.sort_values(GROUP_COLS + ["time_start"])
    for lag in [1, 2, 4]:
        split_df[f"speed_dev_lag_{lag}"] = s.groupby(GROUP_COLS)["speed_deviation"].shift(lag)
        split_df[f"volume_dev_lag_{lag}"] = s.groupby(GROUP_COLS)["volume_deviation"].shift(lag)

# Residual target: how much does next speed deviate from its historical avg?
for split_df in [train_df, val_df, test_df]:
    split_df["target_residual"] = split_df["target_speed"] - split_df["target_hist_avg_speed"]

TARGET = "target_residual"

# ── 4. Encode categoricals for embedding ─────────────────────
cid_map = {v: i for i, v in enumerate(df_sorted["centreline_id"].unique())}
dir_map = {v: i for i, v in enumerate(df_sorted["direction"].unique())}
n_centrelines = len(cid_map)
n_directions = len(dir_map)
print(f"Centreline IDs: {n_centrelines} | Directions: {n_directions}")

for split_df in [train_df, val_df, test_df]:
    split_df["cid_idx"] = split_df["centreline_id"].map(cid_map).fillna(0).astype(int)
    split_df["dir_idx"] = split_df["direction"].map(dir_map).fillna(0).astype(int)

# ── 5. Numeric features ─────────────────────────────────────
NUM_FEATURES = [
    "longitude", "latitude",
    "volume_15min", "avg_speed_kph",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month", "is_weekend",
    "is_morning_rush", "is_evening_rush",
    # Lags
    "speed_lag_1", "speed_lag_2", "speed_lag_4", "speed_lag_8",
    "speed_lag_12", "speed_lag_16",
    "volume_lag_1", "volume_lag_2", "volume_lag_4", "volume_lag_8",
    "volume_lag_12", "volume_lag_16",
    # Rolling
    "speed_roll_mean_4", "speed_roll_mean_8",
    "speed_roll_std_4", "speed_roll_std_8",
    "volume_roll_mean_4", "volume_roll_mean_8",
    # Historical + deviation
    "hist_avg_speed", "hist_avg_volume",
    "speed_deviation", "volume_deviation",
    "speed_dev_lag_1", "speed_dev_lag_2", "speed_dev_lag_4",
    "volume_dev_lag_1", "volume_dev_lag_2", "volume_dev_lag_4",
]

# ── 6. Prepare tensors ──────────────────────────────────────
all_cols = NUM_FEATURES + [TARGET, "target_speed", "target_hist_avg_speed", "cid_idx", "dir_idx"]
mask_tr = train_df[all_cols].notna().all(axis=1)
mask_val = val_df[all_cols].notna().all(axis=1)
mask_te = test_df[all_cols].notna().all(axis=1)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_tr_num = scaler.fit_transform(imputer.fit_transform(train_df.loc[mask_tr, NUM_FEATURES]))
X_val_num = scaler.transform(imputer.transform(val_df.loc[mask_val, NUM_FEATURES]))
X_te_num = scaler.transform(imputer.transform(test_df.loc[mask_te, NUM_FEATURES]))

# Residual targets (what the NN predicts)
y_tr = train_df.loc[mask_tr, TARGET].values
y_val = val_df.loc[mask_val, TARGET].values
y_te = test_df.loc[mask_te, TARGET].values

# Actual speed targets + hist avg (for reconstruction & final eval)
y_te_actual = test_df.loc[mask_te, "target_speed"].values
hist_te = test_df.loc[mask_te, "target_hist_avg_speed"].values
y_val_actual = val_df.loc[mask_val, "target_speed"].values
hist_val = val_df.loc[mask_val, "target_hist_avg_speed"].values

cid_tr = train_df.loc[mask_tr, "cid_idx"].values
cid_val = val_df.loc[mask_val, "cid_idx"].values
cid_te = test_df.loc[mask_te, "cid_idx"].values
dir_tr = train_df.loc[mask_tr, "dir_idx"].values
dir_val = val_df.loc[mask_val, "dir_idx"].values
dir_te = test_df.loc[mask_te, "dir_idx"].values

# Print residual stats
print(f"\nResidual target stats (train):")
print(f"  Mean: {y_tr.mean():.2f}  Std: {y_tr.std():.2f}  "
      f"Min: {y_tr.min():.1f}  Max: {y_tr.max():.1f}")
print(f"Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_te)} | Num features: {X_tr_num.shape[1]}")

# Baseline: just using hist avg (residual = 0)
baseline_mae = mean_absolute_error(y_te_actual, hist_te)
print(f"\nBaseline MAE (hist avg only, no ML): {baseline_mae:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X_tr_t = torch.tensor(X_tr_num, dtype=torch.float32, device=device)
X_val_t = torch.tensor(X_val_num, dtype=torch.float32, device=device)
X_te_t = torch.tensor(X_te_num, dtype=torch.float32, device=device)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
cid_tr_t = torch.tensor(cid_tr, dtype=torch.long, device=device)
cid_val_t = torch.tensor(cid_val, dtype=torch.long, device=device)
cid_te_t = torch.tensor(cid_te, dtype=torch.long, device=device)
dir_tr_t = torch.tensor(dir_tr, dtype=torch.long, device=device)
dir_val_t = torch.tensor(dir_val, dtype=torch.long, device=device)
dir_te_t = torch.tensor(dir_te, dtype=torch.long, device=device)


# ── 7. Model ────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class TrafficResNet(nn.Module):
    def __init__(self, n_num_features, n_centrelines, n_directions,
                 emb_dim_cid=8, emb_dim_dir=4, hidden=256, n_blocks=3, dropout=0.2):
        super().__init__()
        self.cid_emb = nn.Embedding(n_centrelines, emb_dim_cid)
        self.dir_emb = nn.Embedding(n_directions, emb_dim_dir)

        input_dim = n_num_features + emb_dim_cid + emb_dim_dir

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])

        self.down = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(dropout * 0.5),
        )

        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x_num, cid, direction):
        e_cid = self.cid_emb(cid)
        e_dir = self.dir_emb(direction)
        x = torch.cat([x_num, e_cid, e_dir], dim=1)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.down(x)
        return self.head(x)


# ── 8. Train ────────────────────────────────────────────────
n_feat = X_tr_t.shape[1]
model = TrafficResNet(n_feat, n_centrelines, n_directions,
                      emb_dim_cid=8, emb_dim_dir=4,
                      hidden=256, n_blocks=3, dropout=0.2).to(device)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {param_count:,}")

EPOCHS = 500
BATCH_SIZE = 8192
LR_MAX = 5e-4
LR_START = 1e-6
WARMUP_EPOCHS = 20
PATIENCE = 40

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=1e-4)
# After warmup, cosine decay from LR_MAX to 1e-6
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
loss_fn = nn.SmoothL1Loss(beta=2.0)

best_val_mae = float("inf")  # tracked on reconstructed speed, not residual
patience_counter = 0
best_state = None
best_epoch = 0

t0 = perf_counter()

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr_t), device=device)
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, len(X_tr_t), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        pred = model(X_tr_t[idx], cid_tr_t[idx], dir_tr_t[idx])
        loss = loss_fn(pred, y_tr_t[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    # Evaluate on validation: reconstruct speed = hist_avg + predicted_residual
    model.eval()
    with torch.no_grad():
        val_residual_pred = model(X_val_t, cid_val_t, dir_val_t).cpu().numpy().flatten()
    val_speed_pred = hist_val + val_residual_pred
    val_mae = mean_absolute_error(y_val_actual, val_speed_pred)

    if epoch < WARMUP_EPOCHS:
        # Linear warmup from LR_START to LR_MAX
        warmup_lr = LR_START + (LR_MAX - LR_START) * (epoch + 1) / WARMUP_EPOCHS
        for pg in optimizer.param_groups:
            pg["lr"] = warmup_lr
    else:
        cosine_scheduler.step()

    if epoch % 10 == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        res_mae = mean_absolute_error(y_val, val_residual_pred)
        print(f"  Epoch {epoch:3d} | Loss: {epoch_loss/n_batches:.4f} | "
              f"Residual MAE: {res_mae:.4f} | Speed MAE: {val_mae:.4f} | LR: {lr_now:.2e}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        best_epoch = epoch
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch}, best was {best_epoch}")
            break

train_time = perf_counter() - t0

# ── 9. Evaluate on test set ─────────────────────────────────
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    te_residual_pred = model(X_te_t, cid_te_t, dir_te_t).cpu().numpy().flatten()

# Reconstruct: predicted speed = historical avg + predicted residual
y_pred_speed = hist_te + te_residual_pred

mae = mean_absolute_error(y_te_actual, y_pred_speed)
rmse = np.sqrt(mean_squared_error(y_te_actual, y_pred_speed))
r2 = r2_score(y_te_actual, y_pred_speed)

print(f"\n{'='*60}")
print(f"  ResNet (Baseline + Residual Learning)")
print(f"  Best epoch: {best_epoch}")
print(f"  Hist-avg-only MAE:  {baseline_mae:.4f}  (no ML)")
print(f"  ResNet MAE:         {mae:.4f}")
print(f"  ResNet RMSE:        {rmse:.4f}")
print(f"  ResNet R²:          {r2:.4f}")
print(f"  Improvement:        {(baseline_mae - mae) / baseline_mae * 100:.1f}%")
print(f"  Time:               {train_time:.1f}s")
print(f"{'='*60}")

# Save
torch.save(best_state, os.path.join(RESULTS_DIR, "resnet_model.pt"))
pd.DataFrame({
    "y_true": y_te_actual,
    "y_pred": y_pred_speed,
    "hist_avg": hist_te,
    "residual_pred": te_residual_pred,
}).to_csv(os.path.join(RESULTS_DIR, "resnet_predictions.csv"), index=False)
print("Saved model and predictions.")
