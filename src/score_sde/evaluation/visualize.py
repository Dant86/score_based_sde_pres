"""Plotly figure generators for score-SDE evaluation outputs."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import Tensor

# CIFAR-10 class names (index → human label)
CIFAR10_CLASSES: dict[int, str] = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
}

# Fine-grained CIFAR-100 class names (index → human label)
CIFAR100_CLASSES: dict[int, str] = {
    0: "apple", 1: "aquarium fish", 2: "baby", 3: "bear", 4: "beaver",
    5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
    10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
    15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
    20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
    25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
    30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
    35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
    40: "lamp", 41: "lawn mower", 42: "leopard", 43: "lion", 44: "lizard",
    45: "lobster", 46: "man", 47: "maple tree", 48: "motorcycle", 49: "mountain",
    50: "mouse", 51: "mushroom", 52: "oak tree", 53: "orange", 54: "orchid",
    55: "otter", 56: "palm tree", 57: "pear", 58: "pickup truck", 59: "pine tree",
    60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
    65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
    70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
    75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
    80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet pepper",
    84: "table", 85: "tank", 86: "telephone", 87: "television", 88: "tiger",
    89: "tractor", 90: "train", 91: "trout", 92: "tulip", 93: "turtle",
    94: "wardrobe", 95: "whale", 96: "willow tree", 97: "wolf", 98: "woman",
    99: "worm",
}

# Metropolis theme + Beaver colour scheme
_MAROON = "#800000"          # \usecolortheme{beaver} primary
_BG = "#FAFAFA"              # Metropolis slide background
_FONT_FAMILY = "Fira Sans, sans-serif"

_LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_MAROON, family=_FONT_FAMILY),
    title_font=dict(color=_MAROON, family=_FONT_FAMILY, size=16),
)


def _to_uint8(t: Tensor) -> np.ndarray:
    """Convert a single image tensor (C, H, W) in [−1, 1] to (H, W, C) uint8."""
    arr = t.permute(1, 2, 0).cpu().float().numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def plot_sample_grid(
    samples: dict[str, Tensor],
    title: str = "",
    n_cols: int = 8,
    img_size_px: int = 120,
) -> go.Figure:
    """Render a grid of generated images grouped by row label.

    Args:
        samples (dict[str, Tensor]): mapping from row label to image batch of
            shape (N, C, H, W) in [−1, 1]. At least ``n_cols`` images required
            per entry.
        title (str): figure title displayed at the top.
        n_cols (int): number of image columns per row.
        img_size_px (int): pixel height and width to render each image cell.

    Returns:
        go.Figure: interactive Plotly figure with one subplot per image.
    """
    row_labels = list(samples.keys())
    n_rows = len(row_labels)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        row_titles=row_labels,
        vertical_spacing=0.04,
        horizontal_spacing=0.005,
    )

    for row_idx, (label, batch) in enumerate(samples.items(), start=1):
        for col_idx in range(n_cols):
            img = _to_uint8(batch[col_idx])
            fig.add_trace(go.Image(z=img, hoverinfo="skip"), row=row_idx, col=col_idx + 1)

    _LABEL_WIDTH = 160  # px reserved for right-side row-title annotations
    t_margin = 40 if title else 12  # skip top padding when slide already has a title
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title),
        width=n_cols * img_size_px + _LABEL_WIDTH,
        height=n_rows * img_size_px + t_margin + 12,
        margin=dict(l=20, r=_LABEL_WIDTH, t=t_margin, b=12),
        showlegend=False,
    )
    for annotation in fig.layout.annotations:
        annotation.update(
            font=dict(color=_MAROON, family=_FONT_FAMILY, size=13),
            textangle=0,
        )

    return fig


def plot_fid_bars(
    fid_scores: dict[str, float],
    title: str = "FID Comparison (↓ better)",
) -> go.Figure:
    """Render a bar chart comparing FID scores across models or samplers.

    Args:
        fid_scores (dict[str, float]): mapping from model/sampler label to FID
            score. Lower is better.
        title (str): figure title.

    Returns:
        go.Figure: Plotly bar chart with value annotations on each bar.
    """
    labels = list(fid_scores.keys())
    values = list(fid_scores.values())

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            marker_color=_MAROON,
        )
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        showlegend=False,
        yaxis=dict(title="FID ↓", gridcolor="#E0D0D0"),
        xaxis=dict(linecolor=_MAROON),
        width=620,
        height=400,
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


def plot_sde_with_fid(
    samples: dict[str, "Tensor"],
    fid_scores: dict[str, float],
    title: str = "",
    n_cols: int = 8,
    img_size_px: int = 120,
    bar_width_px: int = 300,
    label_width_px: int = 130,
) -> go.Figure:
    """Combined slide figure: image grid on the left, FID bar chart on the right.

    Renders one row per SDE (VP-SDE, VE-SDE, Sub-VP SDE) with ``n_cols``
    sample images, then a single FID bar chart that spans all rows on the right.
    Row labels appear as left-side annotations.

    Args:
        samples (dict[str, Tensor]): mapping from SDE label to image batch of
            shape (N, C, H, W) in [−1, 1]. Keys must match ``fid_scores``.
            At least ``n_cols`` images required per entry.
        fid_scores (dict[str, float]): mapping from SDE label to FID score.
        title (str): optional figure title at the top.
        n_cols (int): number of image columns per SDE row.
        img_size_px (int): pixel height/width for each image cell.
        bar_width_px (int): pixel width reserved for the FID bar chart column.
        label_width_px (int): pixel width reserved for the left row-label margin.

    Returns:
        go.Figure: interactive Plotly figure suitable for embedding in a slide.
    """
    row_labels = list(samples.keys())
    n_rows = len(row_labels)

    # specs: n_cols image cells + 1 bar cell (rowspan=n_rows) per row
    specs: list[list[dict | None]] = []
    for r in range(n_rows):
        row: list[dict | None] = [{"type": "image"}] * n_cols
        if r == 0:
            row.append({"type": "xy", "rowspan": n_rows})
        else:
            row.append(None)
        specs.append(row)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols + 1,
        specs=specs,
        vertical_spacing=0.03,
        horizontal_spacing=0.005,
    )

    # --- image traces ---
    for row_idx, (label, batch) in enumerate(samples.items(), start=1):
        for col_idx in range(n_cols):
            img = _to_uint8(batch[col_idx])
            fig.add_trace(go.Image(z=img, hoverinfo="skip"), row=row_idx, col=col_idx + 1)

    # --- FID bar chart in the last column (row 1 due to rowspan) ---
    bar_labels = list(fid_scores.keys())
    bar_values = list(fid_scores.values())
    fig.add_trace(
        go.Bar(
            x=bar_labels,
            y=bar_values,
            text=[f"{v:.1f}" for v in bar_values],
            textposition="outside",
            marker_color=_MAROON,
            name="FID",
        ),
        row=1,
        col=n_cols + 1,
    )

    # --- axes cleanup ---
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # Re-enable tick labels on the bar chart axes (last subplot column)
    bar_axis_idx = n_cols + 1
    fig.update_xaxes(
        showticklabels=True,
        tickangle=-30,
        tickfont=dict(color=_MAROON, size=11),
        col=bar_axis_idx,
    )
    fig.update_yaxes(
        showticklabels=True,
        title_text="FID ↓",
        title_font=dict(color=_MAROON, size=12),
        gridcolor="#E0D0D0",
        col=bar_axis_idx,
    )

    t_margin = 40 if title else 12
    total_width = label_width_px + n_cols * img_size_px + bar_width_px
    total_height = n_rows * img_size_px + t_margin + 12

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title),
        width=total_width,
        height=total_height,
        margin=dict(l=label_width_px, r=20, t=t_margin, b=12),
        showlegend=False,
    )

    # --- left-side row label annotations ---
    for row_idx, label in enumerate(row_labels):
        # y position: centre of each row, in paper coordinates
        row_frac_height = 1.0 / n_rows
        y_centre = 1.0 - (row_idx + 0.5) * row_frac_height
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=-label_width_px / total_width,
            y=y_centre,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color=_MAROON, family=_FONT_FAMILY, size=13),
            xanchor="left",
            yanchor="middle",
        )

    return fig


def save_figure(fig: go.Figure, path: str) -> None:
    """Write a Plotly figure to an interactive HTML file.

    Args:
        fig (go.Figure): figure to serialise.
        path (str): destination file path (should end in ``.html``).
    """
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  Saved: {path}")
