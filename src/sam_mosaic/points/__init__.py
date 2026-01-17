"""Point generation for SAM prompts."""

from sam_mosaic.points.uniform import make_uniform_grid
from sam_mosaic.points.kmeans import make_kmeans_points
from sam_mosaic.points.dense_grid import make_dense_grid_points
from sam_mosaic.points.filter import filter_points_by_mask

__all__ = [
    "make_uniform_grid",
    "make_kmeans_points",
    "make_dense_grid_points",
    "filter_points_by_mask",
]
