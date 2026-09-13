"""Reference Voronoi center-method readout (reproduced GT: CD MAPE 3.1%, HEX 6%, CV corr 0.92 at ~2x scale)."""
import numpy as np
from scipy.spatial import Voronoi, ConvexHull, Delaunay

UM = 0.7716049

def poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def voronoi_metrics(pts, um_per_px=UM):
    pts = np.asarray(pts, float)
    if len(pts) < 8:
        return dict(CD=np.nan, CV=np.nan, HEX=np.nan, n_interior=0)
    vor = Voronoi(pts)
    hull = Delaunay(pts[ConvexHull(pts).vertices])
    areas, nsides = [], []
    for ri in vor.point_region:
        reg = vor.regions[ri]
        if -1 in reg or len(reg) == 0:
            continue
        poly = vor.vertices[reg]
        if hull.find_simplex(poly).min() < 0:
            continue
        areas.append(poly_area(poly)); nsides.append(len(reg))
    if not areas:
        return dict(CD=np.nan, CV=np.nan, HEX=np.nan, n_interior=0)
    a = np.array(areas) * um_per_px ** 2
    return dict(CD=len(a) / a.sum() * 1e6, CV=a.std() / a.mean(), HEX=float(np.mean(np.array(nsides) == 6)), n_interior=len(a))
