import numpy.testing as npt
import pytest

from fury.actor import Mesh
from fury.data import fetch_gltf, read_viz_gltf
from fury.gltf import have_gltflib, load_gltf, load_gltf_mesh
from fury.lib import gfx
from fury.optpkg import TripWireError

pytestmark = pytest.mark.skipif(not have_gltflib, reason="Requires gltflib")


def test_load_gltf():
    fetch_gltf(name="Duck")
    filename = read_viz_gltf("Duck")
    gltf_obj = load_gltf(filename)

    npt.assert_equal(len(gltf_obj.scenes), 1)
    npt.assert_equal(len(gltf_obj.cameras), 1)
    npt.assert_equal(isinstance(gltf_obj.scene, gfx.Group), True)


def test_load_gltf_animations():
    fetch_gltf(name="InterpolationTest")
    filename = read_viz_gltf("InterpolationTest")
    gltf_obj = load_gltf(filename)

    npt.assert_equal(len(gltf_obj.animations), 9)
    npt.assert_equal("Step Scale" in [clip.name for clip in gltf_obj.animations], True)


def test_load_gltf_mesh():
    fetch_gltf(name="BoxTextured")
    filename = read_viz_gltf("BoxTextured")
    meshes = load_gltf_mesh(filename)

    npt.assert_equal(len(meshes), 1)
    npt.assert_equal(isinstance(meshes[0], Mesh), True)

    geometry = meshes[0].geometry
    npt.assert_equal(geometry.positions.data.shape, (24, 3))
    npt.assert_equal(geometry.indices.data.shape, (12, 3))
    npt.assert_equal(geometry.normals.data.shape, (24, 3))
    npt.assert_equal(geometry.texcoords.data.shape, (24, 2))


def test_load_gltf_mesh_without_materials():
    fetch_gltf(name="BoxTextured")
    filename = read_viz_gltf("BoxTextured")

    textured = load_gltf_mesh(filename)[0]
    npt.assert_equal(textured.material.map is not None, True)

    plain = load_gltf_mesh(filename, materials=False)[0]
    npt.assert_equal(plain.material.map, None)


def test_load_gltf_mesh_is_actor():
    fetch_gltf(name="BoxTextured")
    filename = read_viz_gltf("BoxTextured")
    mesh = load_gltf_mesh(filename)[0]

    mesh.translate((1, 2, 3))
    npt.assert_array_almost_equal(mesh.local.position, (1, 2, 3))


def test_load_gltf_remote_not_allowed():
    url = "https://example.org/Duck.gltf"

    npt.assert_raises(ValueError, load_gltf, url)
    npt.assert_raises(ValueError, load_gltf_mesh, url)


def test_load_gltf_without_gltflib(monkeypatch):
    monkeypatch.setattr("fury.gltf.have_gltflib", False)

    npt.assert_raises(TripWireError, load_gltf, "Duck.gltf")
    npt.assert_raises(TripWireError, load_gltf_mesh, "Duck.gltf")
