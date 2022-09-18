import os
import numpy as np
import numpy.testing as npt
from fury.gltf import glTF, export_scene
from fury import window, utils, actor
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


def test_colors():
    # vertex colors
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

    # material colors
    fetch_gltf('BoxAnimated')
    file = read_viz_gltf('BoxAnimated', 'glTF')
    importer = glTF(file)
    actors = importer.actors()
    scene.add(*actors)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(display, bg_color=(0, 0, 0),
                                  colors=[(77, 136, 204)],
                                  find_objects=True)

    npt.assert_equal(res.colors_found, [True])
    npt.assert_equal(res.objects, 1)
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


def test_export_gltf():
    scene = window.Scene()

    centers = np.zeros((3, 3))
    colors = np.array([1, 1, 1])

    cube = actor.cube(np.add(centers, np.array([2, 0, 0])), colors=colors)
    scene.add(cube)
    export_scene(scene, 'test.gltf')
    gltf_obj = glTF('test.gltf')
    actors = gltf_obj.actors()
    npt.assert_equal(len(actors), 1)

    sphere = actor.sphere(centers, np.array([1, 0, 0]), use_primitive=False)
    scene.add(sphere)
    export_scene(scene, 'test.gltf')
    gltf_obj = glTF('test.gltf')
    actors = gltf_obj.actors()

    scene.clear()
    scene.add(*actors)
    npt.assert_equal(len(actors), 2)

    scene.set_camera(position=(150.0, 10.0, 10.0), focal_point=(0.0, 0.0, 0.0),
                     view_up=(0.0, 0.0, 1.0))
    export_scene(scene, 'test.gltf')
    gltf_obj = glTF('test.gltf')
    actors = gltf_obj.actors()

    scene.clear()
    scene.add(*actors)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(display)
    npt.assert_equal(res.objects, 1)

    scene.reset_camera_tight()
    scene.clear()

    fetch_gltf('BoxTextured', 'glTF')
    filename = read_viz_gltf('BoxTextured')
    gltf_obj = glTF(filename)
    box_actor = gltf_obj.actors()
    scene.add(* box_actor)
    export_scene(scene, 'test.gltf')
    scene.clear()

    gltf_obj = glTF('test.gltf')
    actors = gltf_obj.actors()
    scene.add(* actors)

    display = window.snapshot(scene)
    res = window.analyze_snapshot(display, bg_color=(0, 0, 0),
                                  colors=[(108, 173, 223), (92, 135, 39)],
                                  find_objects=False)
    npt.assert_equal(res.colors_found, [True, True])
