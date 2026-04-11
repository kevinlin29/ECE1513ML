"""Plot raw vs engineered comparison for all 4 models."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = ["KNN", "LightGBM", "XGBoost", "ResNet MLP"]

# Raw baseline results (from train_baseline_raw.py)
raw_mae   = [5.0415, 3.9541, 3.8395, 3.8575]
raw_r2    = [0.5586, 0.6871, 0.7049, 0.6964]
raw_time  = [5.1,    119.2,  2.4,    47.9]

# Engineered results (from train_all.py)
eng_mae   = [3.8893, 3.3078, 3.3021, 3.2799]
eng_r2    = [0.7092, 0.7685, 0.7693, 0.7663]
eng_time  = [0.0,    30.8,   3.6,    105.7]

x = np.arange(len(models))
w = 0.35

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── MAE ──────────────────────────────────────────────────────
ax = axes[0]
b1 = ax.bar(x - w/2, raw_mae, w, label="Raw (19 features)", color="#94a3b8", edgecolor="white")
b2 = ax.bar(x + w/2, eng_mae, w, label="Engineered (45 features)", color="#2563eb", edgecolor="white")
ax.set_ylabel("MAE (km/h) — lower is better", fontsize=11)
ax.set_title("MAE Comparison", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 5.5)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

# ── R² ───────────────────────────────────────────────────────
ax = axes[1]
b1 = ax.bar(x - w/2, raw_r2, w, label="Raw", color="#94a3b8", edgecolor="white")
b2 = ax.bar(x + w/2, eng_r2, w, label="Engineered", color="#16a34a", edgecolor="white")
ax.set_ylabel("R² — higher is better", fontsize=11)
ax.set_title("R² Comparison", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0.4, 0.85)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

# ── Training Time ────────────────────────────────────────────
ax = axes[2]
b1 = ax.bar(x - w/2, raw_time, w, label="Raw", color="#94a3b8", edgecolor="white")
b2 = ax.bar(x + w/2, eng_time, w, label="Engineered", color="#f97316", edgecolor="white")
ax.set_ylabel("Training Time (seconds)", fontsize=11)
ax.set_title("Training Time", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=9)
for bars in [b1, b2]:
    for bar in bars:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8)

fig.suptitle("Raw Features vs Engineered Features — All Models", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/raw_vs_engineered.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: results/raw_vs_engineered.png")
