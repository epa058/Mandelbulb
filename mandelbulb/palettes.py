"""Cosine colour palettes:  colour(t) = a + b * cos(2*pi*(c*t + d))   (rows a, b, c, d)."""

import numpy as np

PALETTES = {
    "Ember": [[0.55, 0.40, 0.30], [0.45, 0.35, 0.30], [1.00, 1.00, 1.00], [0.00, 0.10, 0.20]],
    "Skytopia": [[0.60, 0.50, 0.45], [0.40, 0.40, 0.40], [1.00, 0.90, 0.80], [0.05, 0.25, 0.45]],
    "Ocean": [[0.30, 0.50, 0.60], [0.30, 0.35, 0.35], [1.00, 1.00, 1.00], [0.55, 0.45, 0.35]],
    "Rainbow": [[0.50, 0.50, 0.50], [0.50, 0.50, 0.50], [1.00, 1.00, 1.00], [0.00, 0.33, 0.67]],
    "Jade": [[0.40, 0.55, 0.45], [0.30, 0.35, 0.30], [1.00, 1.00, 1.00], [0.30, 0.20, 0.25]],
    "Amethyst": [[0.55, 0.45, 0.60], [0.40, 0.30, 0.40], [1.00, 1.00, 0.50], [0.80, 0.90, 0.30]],
    "Bone": [[0.70, 0.66, 0.60], [0.15, 0.15, 0.15], [1.00, 1.00, 1.00], [0.00, 0.05, 0.10]],
}

DEFAULT_PALETTE = "Skytopia"


def get_palette(name=None):
    if isinstance(name, np.ndarray):
        return name.astype(np.float64)
    return np.array(PALETTES.get(name or DEFAULT_PALETTE, PALETTES[DEFAULT_PALETTE]),
                    dtype=np.float64)
