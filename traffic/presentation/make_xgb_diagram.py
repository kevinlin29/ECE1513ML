"""Generate XGBoost architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 16)
ax.set_ylim(1, 9.5)
ax.axis("off")

BLUE = "#062958"
LBLUE = "#c7d5ea"
MBLUE = "#8fa8cc"
GREEN = "#2d8a4e"
LGREEN = "#d4edda"
GRAY = "#f0f0f0"
ORANGE = "#f97316"
LORANGE = "#ffedd5"
DARK = "#1e293b"


def box(x, y, w, h, label, sublabel="", color=LBLUE, border=BLUE):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.18, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=DARK)
        ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha="center", va="center",
                fontsize=7.5, color="#555555")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=DARK)


def arrow_right(x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->, head_width=0.2, head_length=0.12",
                               color=BLUE, lw=1.5))


# ── Title ────────────────────────────────────────────────────
ax.text(8.0, 9.0, "XGBoost Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color=BLUE)
ax.text(8.0, 8.5, "Gradient-boosted decision trees — each tree corrects errors of previous trees",
        ha="center", va="center", fontsize=9, color="#666666")

# ── Input ────────────────────────────────────────────────────
box(0.3, 4.3, 2.6, 1.2, "45 Input Features", "speed, weather, lags,\ntime, location", GRAY, "#999999")
arrow_right(3.0, 3.8, 4.9)

# ── Trees ────────────────────────────────────────────────────
tree_x = [4.0, 6.5, 9.5]
tree_labels = ["Tree 1", "Tree 2", f"Tree N"]
tree_subs = ["fits target\nresiduals", "fits Tree 1\nresiduals", "fits Tree N-1\nresiduals"]

for i, (tx, label, sub) in enumerate(zip(tree_x, tree_labels, tree_subs)):
    box(tx, 3.8, 2.0, 2.2, label, sub, MBLUE, BLUE)

    # Small tree icon
    cx = tx + 1.0
    circle = plt.Circle((cx, 7.2), 0.15, color=BLUE, fill=True)
    ax.add_patch(circle)
    circle = plt.Circle((cx - 0.4, 6.6), 0.12, color=MBLUE, ec=BLUE, fill=True, lw=1)
    ax.add_patch(circle)
    circle = plt.Circle((cx + 0.4, 6.6), 0.12, color=MBLUE, ec=BLUE, fill=True, lw=1)
    ax.add_patch(circle)
    ax.plot([cx, cx - 0.4], [7.05, 6.72], color=BLUE, lw=1.2)
    ax.plot([cx, cx + 0.4], [7.05, 6.72], color=BLUE, lw=1.2)
    for lx in [cx - 0.6, cx - 0.2, cx + 0.2, cx + 0.6]:
        rect = mpatches.FancyBboxPatch((lx - 0.1, 6.05), 0.2, 0.2,
                                        boxstyle="round,pad=0.03", facecolor=LGREEN, edgecolor=GREEN, lw=0.8)
        ax.add_patch(rect)
    ax.plot([cx - 0.4, cx - 0.6], [6.48, 6.25], color=BLUE, lw=0.8)
    ax.plot([cx - 0.4, cx - 0.2], [6.48, 6.25], color=BLUE, lw=0.8)
    ax.plot([cx + 0.4, cx + 0.2], [6.48, 6.25], color=BLUE, lw=0.8)
    ax.plot([cx + 0.4, cx + 0.6], [6.48, 6.25], color=BLUE, lw=0.8)

    if i < len(tree_x) - 1:
        arrow_right(tx + 2.1, tree_x[i + 1] - 0.1, 4.9)

# Dots between tree 2 and tree N
ax.text(9.0, 4.9, ". . .", ha="center", va="center", fontsize=16, color=BLUE, fontweight="bold")

# ── Sum ──────────────────────────────────────────────────────
arrow_right(11.6, 12.2, 4.9)
box(12.3, 3.8, 1.6, 2.2, "Sum", "weighted sum\nof all trees", LORANGE, ORANGE)

# ── Output ───────────────────────────────────────────────────
arrow_right(14.0, 14.5, 4.9)
ax.text(15.2, 4.9, "avg_speed\n(km/h)", ha="center", va="center",
        fontsize=11, fontweight="bold", color=GREEN,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=LGREEN, edgecolor=GREEN, lw=1.5))

# ── Config bar at bottom ─────────────────────────────────────
config_text = "max_depth=8  |  lr=0.01  |  ~2600 trees  |  MAE loss  |  early stopping  |  GPU histogram"
ax.text(8.0, 2.5, config_text, ha="center", va="center", fontsize=8, color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#e2e8f0", lw=1))

plt.tight_layout()
plt.savefig("xgb_diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved xgb_diagram.png")
