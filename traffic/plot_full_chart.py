"""Full comparison chart — all models, MAE + R² + Training Time."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = [
    "Global Mean",
    "Historical Avg",
    "Persistence",
    "KNN (raw)",
    "LightGBM (raw)",
    "XGBoost (raw)",
    "ResNet MLP (raw)",
    "KNN (eng)",
    "LightGBM (eng)",
    "XGBoost (eng)",
    "ResNet MLP (eng)",
    "XGBoost (tuned)",
    "ResNet MLP (tuned)",
]

mae   = [9.596, 8.403, 4.193, 5.042, 3.954, 3.840, 3.858, 3.889, 3.308, 3.194, 3.280, 3.180, 3.172]
r2    = [-0.341, -0.084, 0.612, 0.559, 0.687, 0.705, 0.696, 0.709, 0.769, 0.783, 0.766, 0.781, 0.781]
times = [0, 0, 0, 5.1, 119.2, 2.4, 47.9, 0, 30.8, 4.2, 105.7, 49.0, 139.1]

# Color by group
group_colors = {
    "naive": "#94a3b8",
    "naive_best": "#64748b",
    "raw": "#a78bfa",
    "eng": "#2563eb",
    "tuned": "#16a34a",
}
color_map = [
    "naive", "naive", "naive_best",
    "raw", "raw", "raw", "raw",
    "eng", "eng", "eng", "eng",
    "tuned", "tuned",
]
colors = [group_colors[c] for c in color_map]

y = np.arange(len(models))

fig, axes = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={"width_ratios": [1.3, 1.0, 0.8]})

# ── MAE ──────────────────────────────────────────────────────
ax = axes[0]
bars = ax.barh(y, mae, color=colors, edgecolor="white", height=0.7)
ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=10)
ax.set_xlabel("MAE (km/h) — lower is better", fontsize=11)
ax.set_title("MAE", fontsize=13, fontweight="bold")
ax.invert_yaxis()
ax.set_xlim(0, 11.5)

for bar, m in zip(bars, mae):
    ax.text(bar.get_width() + 0.12, bar.get_y() + bar.get_height()/2,
            f"{m:.3f}", va="center", fontsize=8.5, color="#374151")

ax.axvline(x=4.193, color="#64748b", linestyle=":", alpha=0.6, lw=1)
ax.axvline(x=3.172, color="#16a34a", linestyle="--", alpha=0.6, lw=1)

# Section dividers
for div_y in [2.5, 6.5, 10.5]:
    ax.axhline(y=div_y, color="#cbd5e1", linestyle="-", lw=1.2)

# ── R² ───────────────────────────────────────────────────────
ax = axes[1]
r2_clipped = [max(r, 0) for r in r2]
bars = ax.barh(y, r2_clipped, color=colors, edgecolor="white", height=0.7)
ax.set_yticks(y)
ax.set_yticklabels([""] * len(models))
ax.set_xlabel("R² — higher is better", fontsize=11)
ax.set_title("R²", fontsize=13, fontweight="bold")
ax.invert_yaxis()
ax.set_xlim(0, 0.9)

for bar, r in zip(bars, r2):
    xpos = max(bar.get_width() + 0.015, 0.03)
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{r:.3f}", va="center", fontsize=8.5, color="#374151")

for div_y in [2.5, 6.5, 10.5]:
    ax.axhline(y=div_y, color="#cbd5e1", linestyle="-", lw=1.2)

# ── Training Time ────────────────────────────────────────────
ax = axes[2]
bars = ax.barh(y, times, color=colors, edgecolor="white", height=0.7)
ax.set_yticks(y)
ax.set_yticklabels([""] * len(models))
ax.set_xlabel("Training Time (seconds)", fontsize=11)
ax.set_title("Training Time", fontsize=13, fontweight="bold")
ax.invert_yaxis()

for bar, t in zip(bars, times):
    if t > 0:
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f"{t:.1f}s", va="center", fontsize=8.5, color="#374151")
    else:
        ax.text(2, bar.get_y() + bar.get_height()/2,
                "—", va="center", fontsize=9, color="#999999")

for div_y in [2.5, 6.5, 10.5]:
    ax.axhline(y=div_y, color="#cbd5e1", linestyle="-", lw=1.2)

# ── Legend ────────────────────────────────────────────────────
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=group_colors["naive"], label="Naive baseline"),
    Patch(facecolor=group_colors["naive_best"], label="Naive (best)"),
    Patch(facecolor=group_colors["raw"], label="ML — raw features (19)"),
    Patch(facecolor=group_colors["eng"], label="ML — engineered features (45)"),
    Patch(facecolor=group_colors["tuned"], label="ML — tuned (best)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

# ── Section labels on far right ──────────────────────────────
ax_r = axes[2]
section_labels = [
    (1.0, "Naive"),
    (4.5, "ML raw"),
    (8.5, "ML eng"),
    (11.5, "ML tuned"),
]
for sy, sl in section_labels:
    ax_r.text(ax_r.get_xlim()[1] * 1.15, sy, sl, va="center", ha="left",
              fontsize=8.5, color="#666666", fontstyle="italic",
              clip_on=False)

fig.suptitle("Full Model Comparison — Naive Baselines vs ML Models",
             fontsize=15, fontweight="bold", y=1.01)

# Summary text at bottom
summary = ("Best naive (Persistence): MAE 4.193   |   "
           "Best ML raw: MAE 3.840 (8.4% improvement)   |   "
           "Best ML engineered: MAE 3.194 (23.8%)   |   "
           "Best ML tuned: MAE 3.172 (24.4%)")
fig.text(0.5, -0.06, summary, ha="center", fontsize=9.5, color="#555555")

plt.tight_layout()
plt.savefig("results/full_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: results/full_comparison.png")
