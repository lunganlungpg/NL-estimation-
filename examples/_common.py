"""Shared helpers for the example scripts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    path = IMAGES / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"saved {path.relative_to(IMAGES.parents[1])}")
