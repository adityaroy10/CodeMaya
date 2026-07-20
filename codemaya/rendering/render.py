"""Rendering — turn a MEL script into a rendered image + exported .obj mesh.

Two backends:

- **maya** (`render.backend: maya`): shells out to `mayapy` running Maya in
  standalone mode to execute the MEL, export the scene to OBJ, and render an
  image. This is the faithful path and needs an Autodesk Maya install.

- **fallback** (`render.backend: fallback`, default): no Maya required. Parses
  the MEL for a primitive (cube/sphere/cylinder/cone/torus) and its size flags,
  builds the mesh with trimesh, exports OBJ, and renders views with matplotlib
  (headless, no GL). Covers the paper's geometric-primitive evaluation set
  (cubes/spheres) so the full eval pipeline runs without Maya.

`make_renderer(cfg)` returns a `callable(mel) -> PIL.Image` used by the stage-2
contrastive loop; `render_mel(...)` returns both the image and the OBJ path.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from codemaya.utils.io import ensure_dir
from codemaya.utils.logging import get_logger

log = get_logger("render")

# MEL primitive command -> canonical shape kind
_PRIM_MAP = {
    "polycube": "cube", "cube": "cube",
    "polysphere": "sphere", "sphere": "sphere",
    "polycylinder": "cylinder", "cylinder": "cylinder",
    "polycone": "cone", "cone": "cone",
    "polytorus": "torus", "torus": "torus",
}


def parse_primitive(mel: str) -> dict:
    """Extract the first primitive and its size flags from a MEL script."""
    text = mel.lower()
    kind = "cube"
    for token, canon in _PRIM_MAP.items():
        if re.search(rf"\b{token}\b", text):
            kind = canon
            break

    def flag(*names, default):
        for n in names:
            m = re.search(rf"-{n}\s+(-?\d+\.?\d*)", text) or re.search(rf"\b{n}\s+(-?\d+\.?\d*)", text)
            if m:
                return float(m.group(1))
        return default

    return {
        "kind": kind,
        "radius": flag("r", "radius", default=1.0),
        "width": flag("w", "width", default=1.0),
        "height": flag("h", "height", default=1.0),
    }


def build_mesh(spec: dict):
    """Build a trimesh mesh from a parsed primitive spec."""
    import trimesh

    kind = spec["kind"]
    r, w, h = spec["radius"], spec["width"], max(spec["height"], spec["width"])
    if kind == "cube":
        return trimesh.creation.box(extents=(w, w, w))
    if kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    if kind == "cylinder":
        return trimesh.creation.cylinder(radius=r, height=h)
    if kind == "cone":
        return trimesh.creation.cone(radius=r, height=h)
    if kind == "torus":
        try:
            return trimesh.creation.torus(major_radius=max(r, 1.0), minor_radius=r * 0.35)
        except Exception:  # noqa: BLE001 - older trimesh without torus
            return trimesh.creation.annulus(r_min=r * 0.6, r_max=r, height=r * 0.4)
    return trimesh.creation.box(extents=(w, w, w))


def render_views(mesh, image_size: int = 512, n_views: int = 1) -> list:
    """Render the mesh from `n_views` azimuths with matplotlib (headless)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    v, f = mesh.vertices, mesh.faces
    images = []
    for i in range(max(1, n_views)):
        azim = 360.0 * i / max(1, n_views)
        fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                        color=(0.6, 0.7, 0.9), edgecolor="none", antialiased=True)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=azim)
        ax.set_axis_off()
        fig.canvas.draw()
        img = Image.frombytes("RGBA", fig.canvas.get_width_height(),
                              bytes(fig.canvas.buffer_rgba())).convert("RGB")
        images.append(img)
        plt.close(fig)
    return images


# --------------------------------------------------------------- maya backend -
_MAYA_DRIVER = r'''
# executed by mayapy: run MEL, export OBJ, render a frame
import sys, maya.standalone
maya.standalone.initialize(name="python")
import maya.mel as mel
import maya.cmds as cmds
mel_path, obj_path, img_path, size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(mel_path) as fh:
    mel.eval(fh.read())
cmds.loadPlugin("objExport", quiet=True)
cmds.select(all=True)
cmds.file(obj_path, force=True, exportSelected=True, type="OBJexport",
          options="groups=1;materials=0;smoothing=1;normals=1")
cmds.setAttr("defaultResolution.width", size)
cmds.setAttr("defaultResolution.height", size)
try:
    cmds.render(cmds.ls(type="camera")[0], x=size, y=size)
except Exception as e:
    sys.stderr.write("render failed: %s\n" % e)
maya.standalone.uninitialize()
'''


def _render_maya(mel: str, cfg, out_stem: Path) -> dict:
    obj_path, img_path = out_stem.with_suffix(".obj"), out_stem.with_suffix(".png")
    with tempfile.NamedTemporaryFile("w", suffix=".mel", delete=False) as mf:
        mf.write(mel)
        mel_file = mf.name
    driver = out_stem.parent / "_maya_driver.py"
    driver.write_text(_MAYA_DRIVER)
    cmd = [cfg.render.mayapy, str(driver), mel_file, str(obj_path), str(img_path),
           str(cfg.render.image_size)]
    subprocess.run(cmd, check=True, timeout=int(cfg.contrastive.render_timeout_sec))
    return {"obj": str(obj_path), "image": str(img_path)}


# ------------------------------------------------------------------- public --
def render_mel(mel: str, cfg, out_stem: str | Path) -> dict:
    """Render one MEL script. Returns {obj, image, image_obj?}."""
    out_stem = Path(out_stem)
    ensure_dir(out_stem.parent)
    if cfg.render.backend == "maya":
        return _render_maya(mel, cfg, out_stem)

    spec = parse_primitive(mel)
    mesh = build_mesh(spec)
    obj_path = out_stem.with_suffix(".obj")
    mesh.export(str(obj_path))
    imgs = render_views(mesh, int(cfg.render.image_size), int(cfg.render.viewpoints))
    img_path = out_stem.with_suffix(".png")
    imgs[0].save(img_path)
    return {"obj": str(obj_path), "image": str(img_path), "spec": spec}


def make_renderer(cfg):
    """Return callable(mel) -> PIL.Image for the contrastive loop (in-memory)."""
    if cfg.render.backend == "maya":
        def _r(mel: str):
            from PIL import Image
            with tempfile.TemporaryDirectory() as d:
                res = _render_maya(mel, cfg, Path(d) / "r")
                return Image.open(res["image"]).convert("RGB")
        return _r

    def _r(mel: str):
        return render_views(build_mesh(parse_primitive(mel)), int(cfg.render.image_size), 1)[0]
    return _r


def run(cfg, args) -> None:
    out = ensure_dir(cfg.paths.renders)
    if args.smoke:
        for i, mel in enumerate(["polyCube -w 2;", "sphere -r 1.5;"]):
            res = render_mel(mel, cfg, Path(out) / f"smoke_{i}")
            log.info("smoke: rendered %r -> obj=%s image=%s (kind=%s)",
                     mel, Path(res["obj"]).name, Path(res["image"]).name,
                     res.get("spec", {}).get("kind"))
        return

    # render every MEL in a directory of *.mel files (or a single --mel file)
    rest = getattr(args, "rest", []) or []
    if "--mel" in rest:
        mel_files = [Path(rest[rest.index("--mel") + 1])]
    else:
        mel_files = sorted(Path(cfg.paths.renders).glob("*.mel"))
    for mf in mel_files:
        res = render_mel(mf.read_text(), cfg, Path(out) / mf.stem)
        log.info("rendered %s -> %s", mf.name, res["image"])
