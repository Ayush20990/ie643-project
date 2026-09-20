"""Topographic-contour artwork used as the deck's visual motif."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def field(seed, n=400):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 0.5625, int(n * 0.5625)))
    z = np.zeros_like(x)
    for _ in range(9):  # sum of gaussian "hills"
        cx, cy, s, a = rng.uniform(0, 1), rng.uniform(0, .56), rng.uniform(.08, .25), rng.uniform(.4, 1.2)
        z += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    z += 0.08 * np.sin(9 * x + 3 * y)
    return x, y, z

def contour_bg(name, bg, line, seed, alpha, w=13.333, h=7.5, levels=26, mask_left=None):
    x, y, z = field(seed)
    fig = plt.figure(figsize=(w, h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    fig.patch.set_facecolor(bg)
    if mask_left is not None:  # fade contours out on the left where text sits
        z = np.where(x < mask_left, np.nan, z)
    ax.contour(x, y, z, levels=levels, colors=line, linewidths=0.9, alpha=alpha)
    ax.set_xlim(0, 1); ax.set_ylim(0, .5625)
    fig.savefig(f"assets/{name}.png", facecolor=bg); plt.close(fig)

contour_bg("bg_title", "#16232A", "#6FB3A6", 3, 0.55)
contour_bg("bg_section", "#16232A", "#6FB3A6", 11, 0.35, mask_left=0.42)
contour_bg("bg_light_corner", "#FFFFFF", "#2F7F73", 5, 0.10)
