"""Small plotting adapter: matplotlib on Kaggle, Pillow fallback for smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def line_plot(path: Path, series: list[tuple[Iterable[float], Iterable[float], str]], *, title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for x, y, label in series:
            ax.plot(list(x), list(y), marker="o", markersize=2, label=label)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if any(label for _, _, label in series):
            ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        return
    from PIL import Image, ImageDraw
    width, height, margin = 900, 620, 80
    image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    all_x = [float(v) for x, _, _ in series for v in x]; all_y = [float(v) for _, y, _ in series for v in y]
    x_min, x_max = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
    y_min, y_max = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)
    if x_min == x_max: x_max = x_min + 1.0
    if y_min == y_max: y_max = y_min + 1.0
    for index, (x_values, y_values, label) in enumerate(series):
        points = []
        for x, y in zip(x_values, y_values):
            px = margin + (float(x) - x_min) / (x_max - x_min) * (width - 2 * margin)
            py = height - margin - (float(y) - y_min) / (y_max - y_min) * (height - 2 * margin)
            points.append((px, py))
        if len(points) >= 2: draw.line(points, fill=colors[index % len(colors)], width=4)
        draw.text((margin + 180 * index, 25), label, fill=colors[index % len(colors)])
    draw.text((margin, 5), title, fill="black"); draw.text((width // 2, height - 35), xlabel, fill="black")
    draw.text((5, margin), ylabel, fill="black"); image.save(path)


def histogram(path: Path, negative: Iterable[float], positive: Iterable[float], *, tau_low: float | None, tau_high: float | None, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    negative, positive = list(negative), list(positive)
    plt = _matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(negative, bins=30, alpha=.65, label="non-vulnerable")
        ax.hist(positive, bins=30, alpha=.65, label="vulnerable")
        if tau_low is not None: ax.axvline(tau_low, linestyle="--", color="tab:blue", label="tau_low")
        if tau_high is not None: ax.axvline(tau_high, linestyle="--", color="tab:red", label="tau_high")
        ax.set(title=title, xlabel="Calibrated probability", ylabel="Count"); ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        return
    bins = 30
    neg_counts = [0] * bins; pos_counts = [0] * bins
    for value in negative: neg_counts[min(int(float(value) * bins), bins - 1)] += 1
    for value in positive: pos_counts[min(int(float(value) * bins), bins - 1)] += 1
    from PIL import Image, ImageDraw
    width, height, margin = 900, 620, 70; image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
    maximum = max(neg_counts + pos_counts + [1]); bin_width = (width - 2 * margin) / bins
    for index, (neg, pos) in enumerate(zip(neg_counts, pos_counts)):
        x0 = margin + index * bin_width
        draw.rectangle((x0, height - margin - neg / maximum * 450, x0 + bin_width / 2, height - margin), fill="#1f77b4")
        draw.rectangle((x0 + bin_width / 2, height - margin - pos / maximum * 450, x0 + bin_width, height - margin), fill="#d62728")
    for value, color in ((tau_low, "#1f77b4"), (tau_high, "#d62728")):
        if value is not None:
            x = margin + float(value) * (width - 2 * margin); draw.line((x, margin, x, height - margin), fill=color, width=3)
    draw.text((margin, 15), title, fill="black"); image.save(path)


def bar_plot(path: Path, labels: list[str], values: list[float], *, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 4.8)); ax.bar(labels, values); ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); return
    from PIL import Image, ImageDraw
    width, height, margin = 1000, 650, 80; image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
    maximum = max(values + [1.0]); bar_width = (width - 2 * margin) / max(len(values), 1)
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = margin + index * bar_width + 8; x1 = margin + (index + 1) * bar_width - 8
        y0 = height - margin - float(value) / maximum * 450
        draw.rectangle((x0, y0, x1, height - margin), fill="#1f77b4"); draw.text((x0, height - margin + 8), label[:14], fill="black")
    draw.text((margin, 15), title, fill="black"); draw.text((5, margin), ylabel, fill="black"); image.save(path)
