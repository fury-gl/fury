import numpy as np
import pytest

from fury.actor.tests._helpers import (
    assert_rejects_pbr_params_on_phong,
    assert_supports_pbr,
    validate_actors,
)


def test_box():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="box")


def test_frustum():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="frustum")


def test_tetrahedron():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="tetrahedron")


def test_icosahedron():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="icosahedron")


def test_rhombicuboctahedron():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="rhombicuboctahedron")


def test_triangularprism():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="triangularprism")


def test_pentagonalprism():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="pentagonalprism")


def test_octagonalprism():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="octagonalprism")


def test_superquadric():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="superquadric")


POLYHEDRON_PBR_ACTORS = [
    "box",
    "frustum",
    "tetrahedron",
    "icosahedron",
    "rhombicuboctahedron",
    "triangularprism",
    "pentagonalprism",
    "octagonalprism",
    "superquadric",
]


@pytest.mark.parametrize("actor_name", POLYHEDRON_PBR_ACTORS)
@pytest.mark.parametrize("mesh_material", ["standard", "physical"])
def test_polyhedron_actors_support_pbr(actor_name, mesh_material):
    assert_supports_pbr(actor_name, mesh_material)


@pytest.mark.parametrize("actor_name", POLYHEDRON_PBR_ACTORS)
def test_polyhedron_actors_reject_pbr_params_on_phong(actor_name):
    assert_rejects_pbr_params_on_phong(actor_name)
