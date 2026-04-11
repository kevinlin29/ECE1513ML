"""Generate a clean architecture diagram for the ResNet MLP."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(5, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 17)
ax.axis("off")

BLUE = "#062958"
LBLUE = "#c7d5ea"
MBLUE = "#8fa8cc"
GREEN = "#2d8a4e"
LGREEN = "#d4edda"
GRAY = "#f0f0f0"
WHITE = "#ffffff"
DARK = "#1e293b"

BOX_W = 5.6
BOX_H = 1.0
BOX_X = 2.2

def box(y, label, sublabel="", color=LBLUE, border=BLUE):
    rect = mpatches.FancyBboxPatch((BOX_X, y), BOX_W, BOX_H,
                                    boxstyle="round,pad=0.12",
                                    facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    if sublabel:
        ax.text(BOX_X + BOX_W/2, y + BOX_H/2 + 0.15, label,
                ha="center", va="center", fontsize=11, fontweight="bold", color=DARK)
        ax.text(BOX_X + BOX_W/2, y + BOX_H/2 - 0.2, sublabel,
                ha="center", va="center", fontsize=8, color="#555555")
    else:
        ax.text(BOX_X + BOX_W/2, y + BOX_H/2, label,
                ha="center", va="center", fontsize=11, fontweight="bold", color=DARK)

def arrow_down(y_top, y_bottom):
    """Arrow from bottom edge of top box to top edge of bottom box."""
    ax.annotate("", xy=(BOX_X + BOX_W/2, y_bottom + BOX_H),
                xytext=(BOX_X + BOX_W/2, y_top),
                arrowprops=dict(arrowstyle="->, head_width=0.2, head_length=0.12",
                               color=BLUE, lw=1.5))

def skip_arrow(y_top, y_bottom):
    """Skip connection: from right side of top box down to right side of bottom box."""
    x = BOX_X + BOX_W + 0.3
    mid_y = (y_top + y_bottom + BOX_H) / 2
    # Arrow goes from top box down to bottom box (+ sign in residual)
    ax.annotate("", xy=(x, y_bottom + BOX_H),
                xytext=(x, y_top),
                arrowprops=dict(arrowstyle="->, head_width=0.15, head_length=0.1",
                               color="#aaaaaa", lw=1.3, linestyle="dashed"))
    ax.text(x + 0.2, mid_y, "skip", ha="left", va="center", fontsize=7,
            color="#999999", fontstyle="italic")

# Layers from top to bottom
gap = 1.35
y = 15.2
layers = [
    ("49 Input Features", "temporal + weather + lags + historical", GRAY, "#999999"),
    ("Projection", "Linear(512) + SiLU + BN + Dropout(0.3)", LBLUE, BLUE),
    ("ResBlock 512", "2 \u00d7 [Linear + SiLU + BN] + skip", MBLUE, BLUE),
    ("ResBlock 512", "2 \u00d7 [Linear + SiLU + BN] + skip", MBLUE, BLUE),
    ("Downsample", "Linear(256) + SiLU + BN", LBLUE, BLUE),
    ("ResBlock 256", "2 \u00d7 [Linear + SiLU + BN] + skip", MBLUE, BLUE),
    ("Output", "Linear(1) \u2192 avg_speed (km/h)", LGREEN, GREEN),
]

positions = []
for i, (label, sub, color, border) in enumerate(layers):
    cy = y - i * gap
    box(cy, label, sub, color, border)
    positions.append(cy)

# Arrows between consecutive layers
for i in range(len(positions) - 1):
    arrow_down(positions[i], positions[i+1])

# Skip connections: proj->res1, res1->res2, down->res3
skip_arrow(positions[1], positions[2])  # proj -> res1
skip_arrow(positions[2], positions[3])  # res1 -> res2
skip_arrow(positions[4], positions[5])  # down -> res3

plt.tight_layout()
plt.savefig("arch_diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved arch_diagram.png")
