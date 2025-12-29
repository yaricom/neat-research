from typing import List

from PIL import Image
from matplotlib import pyplot as plt


def build_combined(image_paths, out_path: str, nrows: int, ncols: int, titles: List[str], figsize=(10, 8)):
    """Combine already saved per-N images into a nrowsxncols mosaic."""
    imgs = [Image.open(p) for p in image_paths]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)

    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()