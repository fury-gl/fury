import numpy as np
import pygfx as gfx

from fury import material, window
from fury.geometry import buffer_to_geometry, create_mesh
from fury.material import _create_mesh_material
from fury.primitive import prim_sphere


def test_create_mesh_material():
    color = (1, 0, 0)
    mat = material._create_mesh_material(
        material="phong", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, material.MeshPhongMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0, 0.5)
    mat = material._create_mesh_material(
        material="phong", color=color, opacity=0.5, mode="auto", flat_shading=False
    )
    assert isinstance(mat, material.MeshPhongMaterial)
    assert mat.color == (1, 0, 0, 0.25)
    assert mat.color_mode == "auto"
    assert mat.flat_shading is False

    color = (1, 0, 0)
    mat = material._create_mesh_material(
        material="phong", color=color, opacity=0.5, mode="vertex"
    )
    assert isinstance(mat, material.MeshPhongMaterial)
    assert mat.color == (1, 1, 1)
    assert mat.color_mode == "vertex"

    color = (1, 0, 0)
    mat = material._create_mesh_material(
        material="basic",
        color=color,
        mode="vertex",
        enable_picking=False,
        flat_shading=True,
    )
    assert isinstance(mat, material.MeshBasicMaterial)
    assert mat.color == (1, 1, 1)
    assert mat.color_mode == "vertex"
    assert mat.flat_shading is True

    verts, faces = prim_sphere()

    geo = buffer_to_geometry(
        indices=faces.astype("int32"),
        positions=verts.astype("float32"),
        texcoords=verts.astype("float32"),
        colors=np.ones_like(verts).astype("float32"),
    )

    mat = _create_mesh_material(
        material="phong", enable_picking=False, flat_shading=False
    )

    obj = create_mesh(geometry=geo, material=mat)

    scene = window.Scene()

    scene.add(obj)

    # window.snapshot(scene=scene, fname="mat_test_1.png")
    #
    # img = Image.open("mat_test_1.png")
    # img_array = np.array(img)
    #
    # mean_r, mean_g, mean_b, _ = np.mean(
    #     img_array.reshape(-1, img_array.shape[2]), axis=0
    # )
    #
    # assert 0 <= mean_r <= 255 and 0 <= mean_g <= 255 and 0 <= mean_b <= 255
    #
    # assert sum([mean_r, mean_g, mean_b]) > 0


def test_create_point_material():
    color = (1, 0, 0)
    mat = material._create_points_material(
        material="basic", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, gfx.PointsMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_points_material(
        material="gaussian", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, gfx.PointsGaussianBlobMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0, 0.5)
    mat = material._create_points_material(
        material="basic", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, gfx.PointsMaterial)
    assert mat.color == (1, 0, 0, 0.25)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_points_material(
        material="basic", color=color, opacity=0.5, mode="vertex"
    )
    assert isinstance(mat, gfx.PointsMaterial)
    assert mat.color == (1, 1, 1)
    assert mat.color_mode == "vertex"


def test_create_marker_material():
    color = (1, 0, 0)
    mat = material._create_marker_material(color=color, opacity=0.5, mode="auto")
    assert isinstance(mat, gfx.PointsMarkerMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0, 0.5)
    mat = material._create_marker_material(color=color, opacity=0.5, mode="auto")
    assert isinstance(mat, gfx.PointsMarkerMaterial)
    assert mat.color == (1, 0, 0, 0.25)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_marker_material(color=color, opacity=0.5, mode="vertex")
    assert isinstance(mat, gfx.PointsMarkerMaterial)
    assert mat.color == (1, 1, 1)
    assert mat.color_mode == "vertex"
