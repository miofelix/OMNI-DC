from evaluation.datasets.base import DatasetCollection
from evaluation.datasets.clearpose import load_clearpose
from evaluation.datasets.dreds import load_dreds
from evaluation.datasets.hammer import load_hammer
from evaluation.datasets.ibims import load_ibims
from evaluation.datasets.kitti import load_kitti
from evaluation.datasets.transpose import load_transpose

__all__ = [
    "DatasetCollection",
    "load_clearpose",
    "load_dreds",
    "load_hammer",
    "load_ibims",
    "load_kitti",
    "load_transpose",
]
