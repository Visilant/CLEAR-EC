"""Peak extraction: local maxima of the lightly smoothed heatmap above a threshold, NMS by min distance."""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def extract_peaks(heatmap, threshold, min_distance, smooth_sigma=1.0, region=None):
    """Returns (N,2) array of (x, y) peak coordinates.

    A peak is a pixel that equals the maximum over a square window of half-size min_distance
    (greedy NMS at min_distance px) and exceeds threshold. `region` (bool mask) optionally
    restricts the search to the mask's bounding box, which keeps the cost independent of the
    unlabelled background.
    """
    hm = gaussian_filter(heatmap, smooth_sigma) if smooth_sigma > 0 else heatmap
    ox = oy = 0
    if region is not None:
        ys, xs = np.where(region)
        oy, ox = ys.min(), xs.min()
        hm = hm[oy:ys.max() + 1, ox:xs.max() + 1]
    r = int(round(min_distance))
    mx = maximum_filter(hm, size=2 * r + 1, mode="nearest")
    ys, xs = np.where((hm >= mx) & (hm > threshold))
    if len(xs) == 0:
        return np.zeros((0, 2), float)
    return np.stack([xs + ox, ys + oy], axis=1).astype(float)
