"""Tests for geometric metrics (pure numpy/scipy)."""
import numpy as np

from codemaya.evaluation.geometry import chamfer_distance, compare_meshes, hausdorff_distance
from codemaya.rendering.render import build_mesh


def test_chamfer_zero_for_identical_clouds():
    pts = np.random.default_rng(0).random((100, 3))
    assert chamfer_distance(pts, pts) == 0.0


def test_chamfer_positive_and_symmetric():
    a = np.random.default_rng(1).random((50, 3))
    b = np.random.default_rng(2).random((50, 3))
    assert chamfer_distance(a, b) > 0
    assert np.isclose(chamfer_distance(a, b), chamfer_distance(b, a))


def test_hausdorff_zero_for_identical():
    pts = np.random.default_rng(3).random((40, 3))
    assert hausdorff_distance(pts, pts) == 0.0


def test_compare_identical_meshes():
    cube = build_mesh({"kind": "cube", "radius": 1, "width": 2.0, "height": 1})
    m = compare_meshes(cube, cube, n_points=512)
    # small but nonzero: the two point clouds are independent random surface samples
    assert m["chamfer"] < 1e-2
    assert np.isclose(m["volume_ratio"], 1.0)
    assert np.isclose(m["area_ratio"], 1.0)


def test_volume_ratio_tracks_scale():
    big = build_mesh({"kind": "cube", "radius": 1, "width": 2.2, "height": 1})
    small = build_mesh({"kind": "cube", "radius": 1, "width": 2.0, "height": 1})
    m = compare_meshes(big, small, n_points=512)
    assert np.isclose(m["volume_ratio"], (2.2 / 2.0) ** 3, rtol=1e-3)  # ~1.331
