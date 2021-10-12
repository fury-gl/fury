from tempfile import TemporaryDirectory

from fury import actor, material, window
from fury.io import load_image
from fury.optpkg import optional_package
from fury.lib import VTK_9_PLUS


import numpy as np
import numpy.testing as npt
import os
import pytest


dipy, have_dipy, _ = optional_package('dipy')


@pytest.mark.skipif(VTK_9_PLUS, reason="Requires VTK < 9.0.0")
def test_manifest_pbr_vtk_less_than_9():
    center = np.array([[0, 0, 0]])

    # Test non-supported material
    test_actor = actor.square(center, directions=(1, 1, 1), colors=(0, 0, 1))
    npt.assert_warns(UserWarning, material.manifest_pbr, test_actor)


@pytest.mark.skipif(not VTK_9_PLUS, reason="Requires VTK >= 9.0.0")
def test_manifest_pbr_vtk_great_than_9():
    # Test non-supported property
    test_actor = actor.text_3d('Test')
    npt.assert_warns(UserWarning, material.manifest_pbr, test_actor)

    # Test non-supported PBR interpolation
    test_actor = actor.scalar_bar()
    npt.assert_warns(UserWarning, material.manifest_pbr, test_actor)

    # Create tmp dir to save and query images
    with TemporaryDirectory() as out_dir:
        tmp_fname = os.path.join(out_dir, 'tmp_img.png')  # Tmp image to test

        scene = window.Scene()  # Setup scene

        test_actor = actor.square(np.array([[0, 0, 0]]), directions=(0, 0, 0),
                                  colors=(0, 0, 1))

        scene.add(test_actor)

        # Test basic actor
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[100, 100, :] / 1000
        desired = np.array([0, 0, 255]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[40, 40, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test default parameters
        material.manifest_pbr(test_actor)
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[100, 100, :] / 1000
        desired = np.array([66, 66, 165]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[40, 40, :] / 1000
        desired = np.array([40, 40, 157]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test roughness
        material.manifest_pbr(test_actor, roughness=0)
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[100, 100, :] / 1000
        desired = np.array([0, 0, 155]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[40, 40, :] / 1000
        desired = np.array([0, 0, 153]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test metallicity
        material.manifest_pbr(test_actor, metallicity=1)
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[100, 100, :] / 1000
        desired = np.array([0, 0, 255]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[40, 40, :] / 1000
        desired = np.array([0, 0, 175]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)


def test_manifest_standard():
    # Test non-supported property
    test_actor = actor.text_3d('Test')
    npt.assert_warns(UserWarning, material.manifest_standard, test_actor)

    center = np.array([[0, 0, 0]])

    # Test non-supported interpolation method
    test_actor = actor.square(center, directions=(1, 1, 1), colors=(0, 0, 1))
    npt.assert_warns(UserWarning, material.manifest_standard, test_actor,
                     interpolation='test')

    # Create tmp dir to save and query images
    with TemporaryDirectory() as out_dir:
        tmp_fname = os.path.join(out_dir, 'tmp_img.png')  # Tmp image to test

        scene = window.Scene()  # Setup scene

        test_actor = actor.box(center, directions=(1, 1, 1), colors=(0, 0, 1),
                               scales=1)
        scene.add(test_actor)

        # Test basic actor
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[75, 100, :] / 1000
        desired = np.array([0, 0, 170]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 125, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 75, :] / 1000
        desired = np.array([0, 0, 85]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test ambient level
        material.manifest_standard(test_actor, ambient_level=1)
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[75, 100, :] / 1000
        desired = np.array([0, 0, 255]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 125, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 75, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test ambient color
        material.manifest_standard(test_actor, ambient_level=.5,
                                   ambient_color=(1, 0, 0))
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[75, 100, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 125, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 75, :] / 1000
        desired = np.array([0, 0, 212]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test diffuse level
        material.manifest_standard(test_actor, diffuse_level=.75)
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[75, 100, :] / 1000
        desired = np.array([0, 0, 127]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 125, :] / 1000
        desired = np.array([0, 0, 128]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 75, :] / 1000
        desired = np.array([0, 0, 64]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test diffuse color
        material.manifest_standard(test_actor, diffuse_level=.5,
                                   diffuse_color=(1, 0, 0))
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[75, 100, :] / 1000
        desired = np.array([0, 0, 85]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 125, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 75, :] / 1000
        desired = np.array([0, 0, 42]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test specular level
        material.manifest_standard(test_actor, specular_level=1)
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[75, 100, :] / 1000
        desired = np.array([170, 170, 255]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 125, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 75, :] / 1000
        desired = np.array([85, 85, 170]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test specular power
        material.manifest_standard(test_actor, specular_level=1,
                                   specular_power=5)
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[75, 100, :] / 1000
        desired = np.array([34, 34, 204]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 125, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 75, :] / 1000
        desired = np.array([1, 1, 86]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        # Test specular color
        material.manifest_standard(test_actor, specular_level=1,
                                   specular_color=(1, 0, 0), specular_power=5)
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[75, 100, :] / 1000
        desired = np.array([34, 0, 170]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 125, :] / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[125, 75, :] / 1000
        desired = np.array([1, 0, 85]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        scene.clear()  # Reset scene

        # Special case: Contour from roi
        data = np.zeros((50, 50, 50))
        data[20:30, 25, 25] = 1.
        data[25, 20:30, 25] = 1.
        test_actor = actor.contour_from_roi(data, color=np.array([1, 0, 1]))
        scene.add(test_actor)

        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[90, 110, :] / 1000
        desired = np.array([253, 0, 253]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[90, 60, :] / 1000
        desired = np.array([180, 0, 180]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        material.manifest_standard(test_actor)
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[90, 110, :] / 1000
        desired = np.array([253, 253, 253]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[90, 60, :] / 1000
        desired = np.array([180, 180, 180]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)

        material.manifest_standard(test_actor, diffuse_color=(1, 0, 1))
        window.record(scene, out_path=tmp_fname, size=(200, 200),
                      reset_camera=True)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        actual = ss[90, 110, :] / 1000
        desired = np.array([253, 0, 253]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
        actual = ss[90, 60, :] / 1000
        desired = np.array([180, 0, 180]) / 1000
        npt.assert_array_almost_equal(actual, desired, decimal=2)
