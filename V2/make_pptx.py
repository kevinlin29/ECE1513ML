"""Generate ECE1513 project presentation."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.chart.data import CategoryChartData
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Colors ────────────────────────────────────────────────────
DARK = RGBColor(0x1E, 0x29, 0x3B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x25, 0x63, 0xEB)
LIGHT_BLUE = RGBColor(0xDB, 0xEA, 0xFE)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LIGHT_GRAY = RGBColor(0xF3, 0xF4, 0xF6)
RED = RGBColor(0xEF, 0x44, 0x44)
ORANGE = RGBColor(0xF9, 0x73, 0x16)
ACCENT = RGBColor(0x7C, 0x3A, 0xED)


def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color, corner_radius=None):
    if corner_radius:
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.adjustments[0] = corner_radius
    else:
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def add_text(slide, left, top, width, height, text, font_size=18, color=DARK, bold=False, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = alignment
    return txBox


def add_bullet_slide_text(tf, items, font_size=16, color=DARK, spacing=Pt(8)):
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = spacing
        p.level = 0


# ══════════════════════════════════════════════════════════════
# SLIDE 1: Title
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide, DARK)

# Accent bar
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(1.5), Inches(1.8), Inches(10), Inches(1.2),
         "Traffic Speed Prediction Near\nUniversity of Toronto",
         font_size=44, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

add_text(slide, Inches(1.5), Inches(3.5), Inches(10), Inches(0.6),
         "A Machine Learning Approach Using Urban Midblock Speed Data",
         font_size=22, color=RGBColor(0x93, 0xA3, 0xBF), alignment=PP_ALIGN.CENTER)

# Divider line
add_shape(slide, Inches(5.5), Inches(4.5), Inches(2.3), Inches(0.03), BLUE)

add_text(slide, Inches(1.5), Inches(5.0), Inches(10), Inches(0.5),
         "ECE1513 \u2014 Introduction to Machine Learning",
         font_size=18, color=RGBColor(0x93, 0xA3, 0xBF), alignment=PP_ALIGN.CENTER)

add_text(slide, Inches(1.5), Inches(5.5), Inches(10), Inches(0.5),
         "April 2026",
         font_size=16, color=GRAY, alignment=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════
# SLIDE 2: Problem Definition
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Problem Definition", font_size=36, color=DARK, bold=True)

# Left column - Problem
card1 = add_shape(slide, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), LIGHT_GRAY, 0.03)
add_text(slide, Inches(1.2), Inches(1.7), Inches(4.8), Inches(0.5),
         "The Challenge", font_size=22, color=BLUE, bold=True)

txBox = slide.shapes.add_textbox(Inches(1.2), Inches(2.4), Inches(4.8), Inches(4.0))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_text(tf, [
    "\u2022  Urban traffic congestion costs Toronto $6B+ annually",
    "\u2022  Existing systems (loop detectors, GPS probes) give real-time\n   data but limited prediction capability",
    "\u2022  City planners and commuters need 15-min ahead forecasts\n   to enable proactive decision-making",
    "\u2022  Challenge: traffic patterns vary by road segment, time of\n   day, day of week, weather, and season",
], font_size=16, color=DARK, spacing=Pt(14))

# Right column - Our approach
card2 = add_shape(slide, Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.2), RGBColor(0xEF, 0xF6, 0xFF), 0.03)
add_text(slide, Inches(7.3), Inches(1.7), Inches(4.8), Inches(0.5),
         "Our Approach", font_size=22, color=BLUE, bold=True)

txBox = slide.shapes.add_textbox(Inches(7.3), Inches(2.4), Inches(4.8), Inches(4.0))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_text(tf, [
    "\u2022  Predict average traffic speed (km/h) on city streets\n   within ~1.5 km of U of T St. George campus",
    "\u2022  15-minute interval predictions using historical speed\n   bins, weather, and temporal features",
    "\u2022  Progressive model improvement: XGBoost baseline\n   \u2192 feature engineering \u2192 neural network with\n   residual connections",
    "\u2022  Evaluation: MAE, RMSE, R\u00b2 on chronological\n   test split (no data leakage)",
], font_size=16, color=DARK, spacing=Pt(14))


# ══════════════════════════════════════════════════════════════
# SLIDE 3: Data Overview
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Data Overview", font_size=36, color=DARK, bold=True)

# Three data cards
cards = [
    ("Traffic Speed Data", BLUE, [
        "Toronto Open Data Portal",
        "2,106,533 records (2020\u20132025)",
        "14 speed bins \u2192 weighted avg speed",
        "15-min intervals, 3,500+ road segments",
        "Filtered to 1,947 segments (\u226572h data)",
    ]),
    ("Weather Data", GREEN, [
        "Environment Canada (Stn 6158359)",
        "53,943 hourly observations",
        "Temperature, precipitation, wind,",
        "visibility, humidity",
        "Rain/snow/fog binary flags",
    ]),
    ("Merged Dataset", ACCENT, [
        "Traffic + weather joined on hour",
        "1,578,402 rows after quality filter",
        "49 engineered features",
        "70/10/20 chronological split",
        "Train \u2192 Val \u2192 Test (no shuffling)",
    ]),
]

for i, (title, color, items) in enumerate(cards):
    left = Inches(0.8 + i * 4.1)
    card = add_shape(slide, left, Inches(1.5), Inches(3.7), Inches(5.0), LIGHT_GRAY, 0.03)

    # Color bar on top of card
    add_shape(slide, left, Inches(1.5), Inches(3.7), Inches(0.06), color)

    add_text(slide, left + Inches(0.3), Inches(1.8), Inches(3.1), Inches(0.5),
             title, font_size=20, color=color, bold=True)

    txBox = slide.shapes.add_textbox(left + Inches(0.3), Inches(2.5), Inches(3.1), Inches(3.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    add_bullet_slide_text(tf, [f"\u2022  {item}" for item in items],
                         font_size=14, color=DARK, spacing=Pt(10))


# ══════════════════════════════════════════════════════════════
# SLIDE 4: Feature Engineering
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Feature Engineering", font_size=36, color=DARK, bold=True)

categories = [
    ("Temporal", BLUE, [
        "Hour, day of week, month",
        "Cyclical encoding (sin/cos)",
        "Rush hour flags (AM/PM)",
        "Weekend indicator",
    ]),
    ("Weather", GREEN, [
        "Temperature, wind speed",
        "Precipitation, visibility",
        "Wind direction (sin/cos)",
        "Rain/snow/fog flags",
    ]),
    ("Lag & Rolling", ORANGE, [
        "Speed deviation lags (1,2,4,8,12,16)",
        "Volume deviation lags",
        "Rolling mean/std (4,8,16 window)",
        "Raw speed rolling (shifted by 2)",
    ]),
    ("Historical", ACCENT, [
        "Avg speed per segment/hour/DOW",
        "Avg volume per segment/hour/DOW",
        "Speed deviation from historical",
        "Lagged speed \u00d7 volume interaction",
    ]),
]

for i, (title, color, items) in enumerate(categories):
    left = Inches(0.5 + i * 3.15)
    card = add_shape(slide, left, Inches(1.5), Inches(2.95), Inches(4.8), LIGHT_GRAY, 0.03)
    add_shape(slide, left, Inches(1.5), Inches(2.95), Inches(0.06), color)

    add_text(slide, left + Inches(0.2), Inches(1.8), Inches(2.55), Inches(0.4),
             title, font_size=18, color=color, bold=True)

    txBox = slide.shapes.add_textbox(left + Inches(0.2), Inches(2.4), Inches(2.55), Inches(3.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    add_bullet_slide_text(tf, [f"\u2022  {item}" for item in items],
                         font_size=13, color=DARK, spacing=Pt(8))

add_text(slide, Inches(0.8), Inches(6.5), Inches(11), Inches(0.5),
         "Target: speed_deviation (difference from historical average) \u2192 add back hist_avg_speed at prediction time",
         font_size=14, color=GRAY, alignment=PP_ALIGN.LEFT)


# ══════════════════════════════════════════════════════════════
# SLIDE 5: Model Iteration - XGBoost Phase
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Model Iteration: XGBoost Phase", font_size=36, color=DARK, bold=True)

add_text(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(0.4),
         "Systematic hyperparameter tuning and feature engineering using autoresearch loop",
         font_size=16, color=GRAY)

# Timeline / steps
steps = [
    ("Baseline", "2.246", "Default XGBoost\nlr=0.05, depth=6"),
    ("Hyperparams", "2.229", "depth=8, lr=0.03\n3000 trees"),
    ("+Features", "2.153", "Rush hour flags\nlag 12/16, interactions"),
    ("Data Filter", "2.082", "Remove segments\nwith < 72h data"),
    ("MAE Loss", "2.021", "Switch to MAE\nobjective function"),
    ("Final XGB", "2.018", "lr=0.008, 15k trees\nearly stopping=50"),
]

for i, (label, mae, desc) in enumerate(steps):
    left = Inches(0.6 + i * 2.05)
    # Arrow between steps
    if i > 0:
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left - Inches(0.35), Inches(2.8), Inches(0.3), Inches(0.3))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = RGBColor(0xD1, 0xD5, 0xDB)
        arrow.line.fill.background()

    # Card
    is_final = (i == len(steps) - 1)
    bg = RGBColor(0xEF, 0xF6, 0xFF) if is_final else LIGHT_GRAY
    card = add_shape(slide, left, Inches(1.8), Inches(1.8), Inches(3.8), bg, 0.03)
    if is_final:
        add_shape(slide, left, Inches(1.8), Inches(1.8), Inches(0.06), BLUE)

    add_text(slide, left + Inches(0.1), Inches(2.0), Inches(1.6), Inches(0.4),
             label, font_size=14, color=BLUE, bold=True, alignment=PP_ALIGN.CENTER)

    add_text(slide, left + Inches(0.1), Inches(2.5), Inches(1.6), Inches(0.5),
             f"MAE: {mae}", font_size=22, color=DARK, bold=True, alignment=PP_ALIGN.CENTER)

    add_text(slide, left + Inches(0.1), Inches(3.2), Inches(1.6), Inches(1.5),
             desc, font_size=12, color=GRAY, alignment=PP_ALIGN.CENTER)

# Improvement banner
banner = add_shape(slide, Inches(0.8), Inches(6.0), Inches(11.7), Inches(0.8), RGBColor(0xEC, 0xFD, 0xF5), 0.03)
add_text(slide, Inches(1.2), Inches(6.1), Inches(11), Inches(0.6),
         "XGBoost Phase: MAE 2.246 \u2192 2.018  (10.2% improvement)  |  R\u00b2 0.891 \u2192 0.904",
         font_size=18, color=GREEN, bold=True, alignment=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════
# SLIDE 6: Model Iteration - Neural Network Phase
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Model Iteration: Neural Network Phase", font_size=36, color=DARK, bold=True)

add_text(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(0.4),
         "Switching to PyTorch MLP on GPU (RTX 5090) unlocked further gains",
         font_size=16, color=GRAY)

steps_nn = [
    ("Simple MLP", "1.931", "512-256-128\n3 linear layers"),
    ("Wider MLP", "1.900", "1024-512-256-128\n4 linear layers"),
    ("ResNet MLP", "1.848", "Residual connections\n512\u2192512\u2192256 blocks"),
    ("Deviation\nTarget", "1.842", "Predict deviation\nfrom hist. average"),
    ("+2025 Data", "1.830", "Added 370K rows\nof 2025 speed data"),
    ("SiLU\nActivation", "1.825", "SiLU (Swish)\nreplaces ReLU"),
]

for i, (label, mae, desc) in enumerate(steps_nn):
    left = Inches(0.6 + i * 2.05)
    if i > 0:
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left - Inches(0.35), Inches(2.8), Inches(0.3), Inches(0.3))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = RGBColor(0xD1, 0xD5, 0xDB)
        arrow.line.fill.background()

    is_final = (i == len(steps_nn) - 1)
    bg = RGBColor(0xF5, 0xF3, 0xFF) if is_final else LIGHT_GRAY
    card = add_shape(slide, left, Inches(1.8), Inches(1.8), Inches(3.8), bg, 0.03)
    if is_final:
        add_shape(slide, left, Inches(1.8), Inches(1.8), Inches(0.06), ACCENT)

    add_text(slide, left + Inches(0.1), Inches(2.0), Inches(1.6), Inches(0.4),
             label, font_size=14, color=ACCENT, bold=True, alignment=PP_ALIGN.CENTER)

    add_text(slide, left + Inches(0.1), Inches(2.5), Inches(1.6), Inches(0.5),
             f"MAE: {mae}", font_size=22, color=DARK, bold=True, alignment=PP_ALIGN.CENTER)

    add_text(slide, left + Inches(0.1), Inches(3.2), Inches(1.6), Inches(1.5),
             desc, font_size=12, color=GRAY, alignment=PP_ALIGN.CENTER)

banner = add_shape(slide, Inches(0.8), Inches(6.0), Inches(11.7), Inches(0.8), RGBColor(0xF5, 0xF3, 0xFF), 0.03)
add_text(slide, Inches(1.2), Inches(6.1), Inches(11), Inches(0.6),
         "Neural Network Phase: MAE 1.931 \u2192 1.825  (5.5% improvement)  |  R\u00b2 0.910 \u2192 0.917",
         font_size=18, color=ACCENT, bold=True, alignment=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════
# SLIDE 7: Architecture Diagram
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Final Model Architecture", font_size=36, color=DARK, bold=True)

# Architecture flow
layers = [
    ("49 Features\n(Input)", Inches(0.8), LIGHT_GRAY, DARK),
    ("Linear 512\n+ SiLU + BN\n+ Dropout(0.3)", Inches(2.8), RGBColor(0xDB, 0xEA, 0xFE), BLUE),
    ("ResBlock 512\n(2 layers + skip)\n+ Dropout(0.2)", Inches(4.8), RGBColor(0xDB, 0xEA, 0xFE), BLUE),
    ("ResBlock 512\n(2 layers + skip)\n+ Dropout(0.2)", Inches(6.8), RGBColor(0xDB, 0xEA, 0xFE), BLUE),
    ("Linear 256\n+ SiLU + BN", Inches(8.8), RGBColor(0xE0, 0xE7, 0xFF), ACCENT),
    ("ResBlock 256\n(2 layers + skip)\n+ Dropout(0.1)", Inches(10.8), RGBColor(0xE0, 0xE7, 0xFF), ACCENT),
]

for label, left, bg, text_color in layers:
    card = add_shape(slide, left, Inches(1.8), Inches(1.7), Inches(1.8), bg, 0.03)
    add_text(slide, left + Inches(0.05), Inches(2.1), Inches(1.6), Inches(1.3),
             label, font_size=12, color=text_color, alignment=PP_ALIGN.CENTER)

# Output
card = add_shape(slide, Inches(10.8), Inches(4.2), Inches(1.7), Inches(1.0), RGBColor(0xEC, 0xFD, 0xF5), 0.03)
add_text(slide, Inches(10.85), Inches(4.3), Inches(1.6), Inches(0.8),
         "Linear 1\n\u2192 speed_deviation", font_size=12, color=GREEN, bold=True, alignment=PP_ALIGN.CENTER)

# Arrows
for i in range(len(layers) - 1):
    left = Inches(2.6 + i * 2.0)
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, Inches(2.5), Inches(0.25), Inches(0.25))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = RGBColor(0xD1, 0xD5, 0xDB)
    arrow.line.fill.background()

# Down arrow to output
arrow = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(11.45), Inches(3.7), Inches(0.25), Inches(0.4))
arrow.fill.solid()
arrow.fill.fore_color.rgb = RGBColor(0xD1, 0xD5, 0xDB)
arrow.line.fill.background()

# Training details
details_card = add_shape(slide, Inches(0.8), Inches(4.5), Inches(9.5), Inches(2.5), LIGHT_GRAY, 0.03)
add_text(slide, Inches(1.2), Inches(4.7), Inches(4), Inches(0.4),
         "Training Configuration", font_size=18, color=DARK, bold=True)

txBox = slide.shapes.add_textbox(Inches(1.2), Inches(5.2), Inches(4), Inches(1.6))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_text(tf, [
    "\u2022  Optimizer: AdamW (lr=1e-3, wd=1e-4)",
    "\u2022  Scheduler: ReduceLROnPlateau (patience=5)",
    "\u2022  Loss: L1 (MAE)",
    "\u2022  Batch size: 4096",
], font_size=13, color=DARK, spacing=Pt(6))

txBox2 = slide.shapes.add_textbox(Inches(5.5), Inches(5.2), Inches(4), Inches(1.6))
tf2 = txBox2.text_frame
tf2.word_wrap = True
add_bullet_slide_text(tf2, [
    "\u2022  Early stopping: patience=20 on val MAE",
    "\u2022  Gradient clipping: max_norm=1.0",
    "\u2022  GPU: NVIDIA RTX 5090 (32GB)",
    "\u2022  Training time: ~50 seconds",
], font_size=13, color=DARK, spacing=Pt(6))


# ══════════════════════════════════════════════════════════════
# SLIDE 8: Results Comparison Chart
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Results: Model Comparison", font_size=36, color=DARK, bold=True)

# Chart
chart_data = CategoryChartData()
chart_data.categories = [
    "XGBoost\nBaseline", "XGBoost\nTuned", "MLP\nSimple",
    "MLP\nWider", "ResNet\nMLP", "+Deviation\n+Data+SiLU",
]
chart_data.add_series("MAE", (2.246, 2.018, 1.931, 1.900, 1.848, 1.825))

chart = slide.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED,
    Inches(0.8), Inches(1.4), Inches(7.5), Inches(5.5),
    chart_data,
).chart

chart.has_legend = False
chart.style = 2

plot = chart.plots[0]
plot.gap_width = 80
series = plot.series[0]
series.format.fill.solid()
series.format.fill.fore_color.rgb = BLUE

# Data labels
series.has_data_labels = True
data_labels = series.data_labels
data_labels.font.size = Pt(12)
data_labels.font.bold = True
data_labels.font.color.rgb = DARK
data_labels.number_format = "0.000"

# Axis formatting
value_axis = chart.value_axis
value_axis.minimum_scale = 1.5
value_axis.maximum_scale = 2.4
value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE5, 0xE7, 0xEB)
value_axis.has_title = True
value_axis.axis_title.text_frame.paragraphs[0].text = "Test MAE (lower is better)"
value_axis.axis_title.text_frame.paragraphs[0].font.size = Pt(12)

# Results table on right
table_data = [
    ["Model", "MAE", "RMSE", "R\u00b2"],
    ["XGBoost Baseline", "2.246", "3.418", "0.891"],
    ["XGBoost Tuned", "2.018", "3.220", "0.904"],
    ["Simple MLP", "1.931", "3.120", "0.910"],
    ["Wider MLP", "1.900", "3.107", "0.911"],
    ["ResNet MLP", "1.848", "3.091", "0.911"],
    ["Final (best)", "1.825", "3.055", "0.917"],
]

table = slide.shapes.add_table(len(table_data), 4, Inches(8.8), Inches(1.8), Inches(4.0), Inches(3.5)).table

for col_idx in range(4):
    table.columns[col_idx].width = Inches(1.0)

for row_idx, row_data in enumerate(table_data):
    for col_idx, text in enumerate(row_data):
        cell = table.cell(row_idx, col_idx)
        cell.text = text
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(12)
        p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

        if row_idx == 0:
            p.font.bold = True
            p.font.color.rgb = WHITE
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE
        elif row_idx == len(table_data) - 1:
            p.font.bold = True
            p.font.color.rgb = ACCENT
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xF5, 0xF3, 0xFF)
        else:
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if row_idx % 2 == 1 else LIGHT_GRAY

# Improvement callout
add_text(slide, Inches(8.8), Inches(5.5), Inches(4.0), Inches(1.2),
         "Overall: 18.7% MAE reduction\nR\u00b2 improved from 0.891 to 0.917\n47 experiments, 28 kept",
         font_size=14, color=GRAY, alignment=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════
# SLIDE 9: Autoresearch Progress (embed image)
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Autoresearch: Experiment Progress", font_size=36, color=DARK, bold=True)

add_text(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(0.4),
         "Inspired by karpathy/autoresearch \u2014 autonomous AI-driven experiment loop with keep/discard decisions",
         font_size=16, color=GRAY)

progress_img = os.path.join("results", "autoresearch_progress.png")
if os.path.exists(progress_img):
    slide.shapes.add_picture(progress_img, Inches(0.5), Inches(1.8), Inches(12.3), Inches(5.2))

# ══════════════════════════════════════════════════════════════
# SLIDE 10: What Worked / Didn't
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Key Findings", font_size=36, color=DARK, bold=True)

# Left - What worked
card_l = add_shape(slide, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.5), RGBColor(0xEC, 0xFD, 0xF5), 0.03)
add_shape(slide, Inches(0.8), Inches(1.4), Inches(5.6), Inches(0.06), GREEN)
add_text(slide, Inches(1.2), Inches(1.7), Inches(4.8), Inches(0.4),
         "What Worked", font_size=22, color=GREEN, bold=True)

txBox = slide.shapes.add_textbox(Inches(1.2), Inches(2.3), Inches(4.8), Inches(4.3))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_text(tf, [
    "\u2713  MAE loss objective (aligning loss with metric)",
    "\u2713  Filtering low-quality road segments (< 72h data)",
    "\u2713  Residual connections in MLP",
    "\u2713  Speed deviation target (centering near 0)",
    "\u2713  More data (adding 2025 speed observations)",
    "\u2713  SiLU activation over ReLU",
    "\u2713  Lag features at multiple time horizons",
    "\u2713  Data quality > data quantity",
], font_size=15, color=DARK, spacing=Pt(12))

# Right - What didn't
card_r = add_shape(slide, Inches(6.9), Inches(1.4), Inches(5.6), Inches(5.5), RGBColor(0xFE, 0xF2, 0xF2), 0.03)
add_shape(slide, Inches(6.9), Inches(1.4), Inches(5.6), Inches(0.06), RED)
add_text(slide, Inches(7.3), Inches(1.7), Inches(4.8), Inches(0.4),
         "What Didn't Work", font_size=22, color=RED, bold=True)

txBox = slide.shapes.add_textbox(Inches(7.3), Inches(2.3), Inches(4.8), Inches(4.3))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_text(tf, [
    "\u2717  Entity embeddings (too few samples per segment)",
    "\u2717  CatBoost (couldn't match XGBoost on this data)",
    "\u2717  OneCycleLR (ReduceLROnPlateau was better)",
    "\u2717  GELU / SmoothL1 loss",
    "\u2717  LayerNorm (BatchNorm was superior)",
    "\u2717  XGB+MLP ensemble (MLP already dominant)",
    "\u2717  Very low learning rate (0.005) for XGBoost",
    "\u2717  Seasonal features (month sin/cos, week_of_year)",
], font_size=15, color=DARK, spacing=Pt(12))


# ══════════════════════════════════════════════════════════════
# SLIDE 11: Conclusion
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), BLUE)

add_text(slide, Inches(1.5), Inches(0.8), Inches(10), Inches(0.7),
         "Conclusion", font_size=36, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

# Three result cards
results_cards = [
    ("18.7%", "MAE Reduction", "2.246 \u2192 1.825 km/h"),
    ("0.917", "R\u00b2 Score", "up from 0.891 baseline"),
    ("47", "Experiments Run", "28 kept, 19 discarded"),
]

for i, (big_num, label, detail) in enumerate(results_cards):
    left = Inches(1.0 + i * 4.0)
    card = add_shape(slide, left, Inches(1.8), Inches(3.4), Inches(2.5), RGBColor(0x2D, 0x3A, 0x50), 0.03)

    add_text(slide, left + Inches(0.2), Inches(2.0), Inches(3.0), Inches(0.8),
             big_num, font_size=44, color=BLUE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide, left + Inches(0.2), Inches(2.9), Inches(3.0), Inches(0.4),
             label, font_size=18, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide, left + Inches(0.2), Inches(3.4), Inches(3.0), Inches(0.4),
             detail, font_size=14, color=GRAY, alignment=PP_ALIGN.CENTER)

# Key takeaways
txBox = slide.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(11.3), Inches(2.2))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_text(tf, [
    "\u2022  Progressive model improvement from tree-based to neural network methods yields consistent gains",
    "\u2022  Data quality (filtering noisy segments) and loss alignment (MAE objective) had the biggest single impacts",
    "\u2022  ResNet-style MLP with SiLU activation outperforms gradient boosting on this traffic prediction task",
    "\u2022  Autoresearch methodology enables systematic, reproducible experimentation at scale",
], font_size=16, color=RGBColor(0xC0, 0xC8, 0xD4), spacing=Pt(14))


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
output_path = os.path.join("results", "ECE1513_Traffic_Prediction.pptx")
prs.save(output_path)
print(f"Saved: {output_path}")
