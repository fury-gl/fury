import numpy as np
import os
import numpy.testing as npt
from fury.gltf import glTF
from fury import window, utils
from fury.data import fetch_gltf, read_viz_gltf
from fury.testing import assert_greater


def test_load_gltf():
    fetch_gltf('Duck')
    filename = read_viz_gltf('Duck', 'glTF')
    importer = glTF(filename)
    polydatas = importer.polydatas
    vertices = utils.get_polydata_vertices(polydatas[0])
    triangles = utils.get_polydata_triangles(polydatas[0])

    npt.assert_equal(vertices.shape, (2399, 3))
    npt.assert_equal(triangles.shape, (4212, 3))
    os.remove(filename)

    scene = window.Scene()
    scene.add(utils.get_actor_from_polydata(polydatas[0]))
    display = window.snapshot(scene)
    res = window.analyze_snapshot(display)
    npt.assert_equal(res.objects, 1)
    scene.clear()


def test_load_texture():
    fetch_gltf('Duck')
    filename = read_viz_gltf('Duck', 'glTF')
    importer = glTF(filename)
    actor = importer.actors()[0]

    scene = window.Scene()
    scene.add(actor)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(display, bg_color=(0, 0, 0),
                                  colors=[(255, 216, 0)],
                                  find_objects=False)
    npt.assert_equal(res.colors_found, [True])
    scene.clear()


def test_vertex_colors():
    fetch_gltf('BoxVertexColors')
    file = read_viz_gltf('BoxVertexColors', 'glTF')
    importer = glTF(file)
    actor = importer.actors()[0]
    scene = window.Scene()
    scene.add(actor)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(display, bg_color=(0, 0, 0),
                                  colors=[(250, 11, 11), (136, 245, 6),
                                          (31, 41, 232)],
                                  find_objects=False)
    npt.assert_equal(res.colors_found, [True, True, True])
    scene.clear()


def test_orientation():
    fetch_gltf('BoxTextured', 'glTF-Embedded')
    file = read_viz_gltf('BoxTextured', 'glTF-Embedded')
    importer = glTF(file)
    actor = importer.actors()[0]

    scene = window.Scene()
    scene.add(actor)
    # if oriented correctly avg of blues on top half will be greater
    # than the bottom half (window.snapshot captures oppsite of it)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(display, bg_color=(0, 0, 0),
                                  colors=[(108, 173, 223), (92, 135, 39)],
                                  find_objects=False)
    npt.assert_equal(res.colors_found, [True, True])
    blue = display[:, :, -1:].reshape((300, 300))
    upper, lower = np.split(blue, 2)
    upper = np.mean(upper)
    lower = np.mean(lower)

    assert_greater(lower, upper)
    scene.clear()
