"""Tests for the fallback renderer's pure logic (no GL)."""
import math

from codemaya.rendering.render import build_mesh, parse_primitive


def test_parse_primitive_kinds_and_flags():
    assert parse_primitive("polyCube -w 2;")["kind"] == "cube"
    assert parse_primitive("polyCube -w 2;")["width"] == 2.0

    s = parse_primitive("sphere -r 1.5 -name foo;")
    assert s["kind"] == "sphere" and s["radius"] == 1.5

    assert parse_primitive("polyCone -r 2 -h 4;")["kind"] == "cone"
    assert parse_primitive("polyTorus -r 3;")["kind"] == "torus"
    # unknown / no primitive defaults to cube
    assert parse_primitive("// nothing here")["kind"] == "cube"


def test_build_mesh_cube_volume():
    mesh = build_mesh({"kind": "cube", "radius": 1.0, "width": 2.0, "height": 1.0})
    # a 2x2x2 box has volume 8
    assert math.isclose(mesh.volume, 8.0, rel_tol=1e-6)
    assert mesh.is_watertight


def test_build_mesh_sphere_is_watertight():
    mesh = build_mesh({"kind": "sphere", "radius": 1.0, "width": 1.0, "height": 1.0})
    assert mesh.is_watertight
    # icosphere radius 1 -> volume slightly under the ideal 4/3*pi
    assert 3.5 < mesh.volume < 4.2
