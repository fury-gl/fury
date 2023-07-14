import os
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest

from fury import actor, material, window
from fury.io import load_image
from fury.optpkg import optional_package

dipy, have_dipy, _ = optional_package('dipy')


def test_manifest_pbr_vtk():
    # Test non-supported property
    test_actor = actor.text_3d('Test')
    npt.assert_warns(UserWarning, material.manifest_pbr, test_actor)

    # Test non-supported PBR interpolation
    test_actor = actor.scalar_bar()
    npt.assert_warns(UserWarning, material.manifest_pbr, test_actor)

    # Create tmp dir to save and query images
    # with TemporaryDirectory() as out_dir:
    # tmp_fname = os.path.join(out_dir, 'tmp_img.png')  # Tmp image to test

    scene = window.Scene()  # Setup scene

    test_actor = actor.square(
        np.array([[0, 0, 0]]), directions=(0, 0, 0), colors=(0, 0, 1)
    )

    scene.add(test_actor)

    # Test basic actor
    # window.record(scene, out_path=tmp_fname, size=(200, 200),
    #              reset_camera=True)
    ss = window.snapshot(scene, size=(200, 200))
    # npt.assert_equal(os.path.exists(tmp_fname), True)
    # ss = load_image(tmp_fname)
    actual = ss[100, 100, :] / 1000
    desired = np.array([0, 0, 255]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[40, 40, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test default parameters
    material.manifest_pbr(test_actor)

    ss = window.snapshot(scene, size=(200, 200))
    # window.record(scene, out_path=tmp_fname, size=(200, 200),
    #                 reset_camera=True)
    # npt.assert_equal(os.path.exists(tmp_fname), True)
    # ss = load_image(tmp_fname)
    actual = ss[100, 100, :] / 1000
    desired = np.array([66, 66, 165]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[40, 40, :] / 1000
    desired = np.array([40, 40, 157]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test roughness
    material.manifest_pbr(test_actor, roughness=0)

    ss = window.snapshot(scene, size=(200, 200))
    # window.record(scene, out_path=tmp_fname, size=(200, 200),
    #                 reset_camera=True)
    # npt.assert_equal(os.path.exists(tmp_fname), True)
    # ss = load_image(tmp_fname)
    actual = ss[100, 100, :] / 1000
    desired = np.array([0, 0, 155]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[40, 40, :] / 1000
    desired = np.array([0, 0, 153]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test metallicity
    material.manifest_pbr(test_actor, metallic=1)
    ss = window.snapshot(scene, size=(200, 200))
    # window.record(scene, out_path=tmp_fname, size=(200, 200),
    #                 reset_camera=True)
    # npt.assert_equal(os.path.exists(tmp_fname), True)
    # ss = load_image(tmp_fname)
    actual = ss[100, 100, :] / 1000
    desired = np.array([0, 0, 255]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[40, 40, :] / 1000
    desired = np.array([0, 0, 175]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)


def test_manifest_principled():
    # Test non-supported property
    test_actor = actor.text_3d('Test')
    npt.assert_warns(UserWarning, material.manifest_principled, test_actor)

    center = np.array([[0, 0, 0]])

    # Test expected parameters
    expected_principled_params = {
        'subsurface': 0,
        'metallic': 0,
        'specular': 0,
        'specular_tint': 0,
        'roughness': 0,
        'anisotropic': 0,
        'anisotropic_direction': [0, 1, 0.5],
        'sheen': 0,
        'sheen_tint': 0,
        'clearcoat': 0,
        'clearcoat_gloss': 0,
    }
    test_actor = actor.square(center, directions=(1, 1, 1), colors=(0, 0, 1))
    actual_principled_params = material.manifest_principled(test_actor)
    npt.assert_equal(actual_principled_params, expected_principled_params)


def test_manifest_standard():
    # Test non-supported property
    test_actor = actor.text_3d('Test')
    npt.assert_warns(UserWarning, material.manifest_standard, test_actor)

    center = np.array([[0, 0, 0]])

    # Test non-supported interpolation method
    test_actor = actor.square(center, directions=(1, 1, 1), colors=(0, 0, 1))
    npt.assert_warns(
        UserWarning, material.manifest_standard, test_actor, interpolation='test'
    )

    scene = window.Scene()  # Setup scene

    test_actor = actor.box(center, directions=(1, 1, 1), colors=(0, 0, 1), scales=1)
    scene.add(test_actor)

    # scene.reset_camera()
    # window.show(scene)
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[125, 100, :] / 1000
    desired = np.array([0, 0, 201]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 75, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 75, :] / 1000
    desired = np.array([0, 0, 85]) / 1000
    # TODO: check if camera affects this assert
    # npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test ambient level
    material.manifest_standard(test_actor, ambient_level=1)
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[125, 100, :] / 1000
    desired = np.array([0, 0, 255]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 125, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 75, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test ambient color
    material.manifest_standard(test_actor, ambient_level=0.5, ambient_color=(1, 0, 0))
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[125, 100, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 125, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 75, :] / 1000
    desired = np.array([0, 0, 212]) / 1000
    # TODO: check what affects this
    # npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test diffuse level
    material.manifest_standard(test_actor, diffuse_level=0.75)
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[125, 100, :] / 1000
    desired = np.array([0, 0, 151]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 125, :] / 1000
    desired = np.array([0, 0, 110]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 75, :] / 1000
    desired = np.array([0, 0, 151]) / 1000
    # TODO: check what affects this
    # npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test diffuse color
    material.manifest_standard(test_actor, diffuse_level=0.5, diffuse_color=(1, 0, 0))
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[125, 100, :] / 1000
    desired = np.array([0, 0, 101]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 75, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 125, :] / 1000
    desired = np.array([0, 0, 74]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test specular level
    material.manifest_standard(test_actor, specular_level=1)
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[125, 100, :] / 1000
    desired = np.array([201, 201, 255]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 75, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 125, :] / 1000
    desired = np.array([147, 147, 255]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test specular power
    material.manifest_standard(test_actor, specular_level=1, specular_power=5)
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[125, 100, :] / 1000
    desired = np.array([78, 78, 255]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 75, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 125, :] / 1000
    desired = np.array([16, 16, 163]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    # Test specular color
    material.manifest_standard(
        test_actor, specular_level=1, specular_color=(1, 0, 0), specular_power=5
    )
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[125, 100, :] / 1000
    desired = np.array([78, 0, 201]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 75, :] / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[75, 125, :] / 1000
    desired = np.array([16, 0, 147]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    scene = window.Scene()

    # Special case: Contour from roi
    data = np.zeros((50, 50, 50))
    data[20:30, 25, 25] = 1.0
    data[25, 20:30, 25] = 1.0
    test_actor = actor.contour_from_roi(data, color=np.array([1, 0, 1]))
    scene.add(test_actor)

    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[100, 106, :] / 1000
    desired = np.array([253, 0, 253]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[100, 150, :] / 1000
    desired = np.array([180, 0, 180]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    material.manifest_standard(test_actor)
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[100, 106, :] / 1000
    desired = np.array([253, 253, 253]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[100, 150, :] / 1000
    desired = np.array([180, 180, 180]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)

    material.manifest_standard(test_actor, diffuse_color=(1, 0, 1))
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[100, 106, :] / 1000
    desired = np.array([253, 0, 253]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
    actual = ss[100, 150, :] / 1000
    desired = np.array([180, 0, 180]) / 1000
    npt.assert_array_almost_equal(actual, desired, decimal=2)
