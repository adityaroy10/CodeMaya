"""Evaluation — geometric fidelity of the generated 3D mesh vs ground truth.

Metrics (paper 4.4-4.7), computed on the exported OBJ meshes:
  - **Chamfer distance**: mean squared nearest-neighbour distance between the two
    point clouds, symmetrised  (1/|P|)Σ min_q||p-q||^2 + (1/|Q|)Σ min_p||q-p||^2.
  - **Hausdorff distance**: worst-case point-wise deviation (max of the two
    directed Hausdorff distances).
  - **Volume ratio** and **surface-area ratio** vs ground truth (1.0 = perfect).

The point-cloud metrics are pure numpy/scipy so they are unit-tested on known
primitives; mesh loading/sampling uses trimesh.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from codemaya.utils.io import ensure_dir, read_jsonl, write_json
from codemaya.utils.logging import get_logger

log = get_logger("eval.geom")


def chamfer_distance(a, b) -> float:
    """Symmetric Chamfer distance between point clouds a[N,3], b[M,3]."""
    import numpy as np
    from scipy.spatial import cKDTree

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d_ab, _ = cKDTree(b).query(a)
    d_ba, _ = cKDTree(a).query(b)
    return float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))


def hausdorff_distance(a, b) -> float:
    """Symmetric Hausdorff distance between point clouds."""
    import numpy as np
    from scipy.spatial.distance import directed_hausdorff

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


def _sample(mesh, n: int):
    pts, _ = mesh.sample(n, return_index=True) if hasattr(mesh, "sample") else (mesh.vertices, None)
    return pts


def compare_meshes(gen_mesh, gt_mesh, n_points: int = 4096) -> dict:
    import numpy as np

    gp = _sample(gen_mesh, n_points)
    tp = _sample(gt_mesh, n_points)
    # normalise scale/translation before shape comparison (center + unit bbox diag)
    def norm(p):
        p = np.asarray(p, float)
        p = p - p.mean(0)
        diag = np.linalg.norm(p.max(0) - p.min(0)) or 1.0
        return p / diag
    gpn, tpn = norm(gp), norm(tp)

    gt_vol = abs(gt_mesh.volume) or 1e-9
    gt_area = gt_mesh.area or 1e-9
    return {
        "chamfer": round(chamfer_distance(gpn, tpn), 6),
        "hausdorff": round(hausdorff_distance(gpn, tpn), 6),
        "volume_ratio": round(float(abs(gen_mesh.volume) / gt_vol), 4),
        "area_ratio": round(float(gen_mesh.area / gt_area), 4),
    }


def _load(obj_path):
    import trimesh
    return trimesh.load(obj_path, force="mesh")


def _aggregate(per_pair: list[dict]) -> dict:
    import numpy as np
    keys = ["chamfer", "hausdorff", "volume_ratio", "area_ratio"]
    return {k: round(float(np.mean([p[k] for p in per_pair])), 6) for k in keys}


def run(cfg, args) -> None:
    n = int(cfg.eval.chamfer_sample_points)

    if args.smoke:
        from codemaya.rendering.render import build_mesh
        gen = build_mesh({"kind": "cube", "radius": 1, "width": 2.2, "height": 1})
        gt = build_mesh({"kind": "cube", "radius": 1, "width": 2.0, "height": 1})
        m = compare_meshes(gen, gt, n_points=1024)
        write_json(Path(ensure_dir(cfg.paths.results)) / "geometry.json", {"finetuned": m})
        log.info("smoke: cube-vs-cube geometry %s", m)
        return

    gen_path = Path(cfg.paths.results) / "generations.jsonl"
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(gen_path):
        if not r.get("obj") or not r.get("gt_obj"):
            continue
        by_model[r["model"]].append(compare_meshes(_load(r["obj"]), _load(r["gt_obj"]), n))

    results = {m: _aggregate(v) for m, v in by_model.items() if v}
    write_json(Path(ensure_dir(cfg.paths.results)) / "geometry.json", results)
    for m, s in results.items():
        log.info("%-10s | chamfer=%.4f hausdorff=%.4f vol=%.3f area=%.3f",
                 m, s["chamfer"], s["hausdorff"], s["volume_ratio"], s["area_ratio"])
