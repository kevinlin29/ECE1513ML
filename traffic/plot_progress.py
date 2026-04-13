"""Plot autoresearch experiment progress — MAE improvement over iterations."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

df = pd.read_csv("results.tsv", sep="\t")

# Only show kept experiments (the advancing frontier)
kept = df[df["status"] == "keep"].reset_index(drop=True)
kept["step"] = range(len(kept))

fig, ax1 = plt.subplots(figsize=(12, 5))

# MAE progression (kept experiments)
ax1.plot(kept["step"], kept["test_mae"], "o-", color="#2563eb", linewidth=2, markersize=8, label="Test MAE (kept)")
ax1.set_xlabel("Experiment Step (kept only)", fontsize=12)
ax1.set_ylabel("Test MAE (lower is better)", fontsize=12, color="#2563eb")
ax1.tick_params(axis="y", labelcolor="#2563eb")

# Annotate key milestones
annotations = {
    0: "Baseline\nXGBoost",
    4: "+Rush hour\n+More lags",
    7: "Filter noisy\nlocations",
    9: "MAE loss\nobjective",
    10: "Best XGBoost\n(2.018)",
    11: "PyTorch\nMLP",
    13: "ResNet\nMLP",
    14: "Deviation\ntarget",
    15: "+2025\ndata",
    16: "SiLU\n(best)",
}
for step, label in annotations.items():
    if step < len(kept):
        ax1.annotate(label,
            xy=(kept.iloc[step]["step"], kept.iloc[step]["test_mae"]),
            xytext=(0, 20), textcoords="offset points",
            ha="center", fontsize=8, color="#374151",
            arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=0.8))

# R² on secondary axis
ax2 = ax1.twinx()
ax2.plot(kept["step"], kept["test_r2"], "s--", color="#16a34a", linewidth=1.5, markersize=6, alpha=0.7, label="Test R²")
ax2.set_ylabel("Test R² (higher is better)", fontsize=12, color="#16a34a")
ax2.tick_params(axis="y", labelcolor="#16a34a")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

ax1.set_title("Autoresearch: Traffic Speed Prediction — Model Improvement", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Add improvement summary
start_mae = kept.iloc[0]["test_mae"]
end_mae = kept.iloc[-1]["test_mae"]
pct_improvement = (start_mae - end_mae) / start_mae * 100
fig.text(0.5, -0.02,
    f"MAE improved from {start_mae:.3f} → {end_mae:.3f} ({pct_improvement:.1f}% reduction)  |  "
    f"R² improved from {kept.iloc[0]['test_r2']:.4f} → {kept.iloc[-1]['test_r2']:.4f}",
    ha="center", fontsize=11, color="#4b5563")

plt.tight_layout()
plt.savefig("results/autoresearch_progress.png", dpi=150, bbox_inches="tight")
print("Saved: results/autoresearch_progress.png")

# Also make a bar chart of all experiments
fig2, ax = plt.subplots(figsize=(14, 6))
colors = ["#2563eb" if s == "keep" else "#ef4444" if s == "crash" else "#9ca3af" for s in df["status"]]
# Filter out the leakage crash (MAE=0.3 distorts scale)
mask = df["test_mae"] > 0.5
bars = ax.barh(range(mask.sum()), df[mask]["test_mae"], color=[c for c, m in zip(colors, mask) if m])
ax.set_yticks(range(mask.sum()))
ax.set_yticklabels([f"{e}: {d}" for e, d in zip(df[mask]["experiment"], df[mask]["description"])], fontsize=7)
ax.set_xlabel("Test MAE", fontsize=12)
ax.set_title("All Experiments — MAE Comparison", fontsize=14, fontweight="bold")
ax.axvline(x=end_mae, color="#16a34a", linestyle="--", alpha=0.7, label=f"Best: {end_mae:.3f}")
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("results/autoresearch_all_experiments.png", dpi=150, bbox_inches="tight")
print("Saved: results/autoresearch_all_experiments.png")
