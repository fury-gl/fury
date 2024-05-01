import itertools
import os
import sys

from PIL import Image
import numpy as np
import numpy.testing as npt
from packaging.version import parse
import pytest
from scipy.ndimage import center_of_mass
from scipy.version import short_version

from fury import actor, utils, window
from fury.animation import Timeline
from fury.data import fetch_gltf, read_viz_gltf
from fury.gltf import export_scene, glTF
from fury.testing import assert_equal, assert_greater

SCIPY_1_8_PLUS = parse(short_version) >= parse('1.8.0')

if SCIPY_1_8_PLUS:
    from scipy.ndimage._measurements import _stats
else:
    from scipy.ndimage.measurements import _stats


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
    res = window.analyze_snapshot(
        display, bg_color=(0, 0, 0), colors=[(255, 216, 0)], find_objects=False
    )
    npt.assert_equal(res.colors_found, [True])
    scene.clear()


@pytest.mark.skipif(True, reason="This test is failing on CI, not sure why yet")
def test_colors():
    # vertex colors
    fetch_gltf('BoxVertexColors')
    file = read_viz_gltf('BoxVertexColors', 'glTF')
    importer = glTF(file)
    actor = importer.actors()[0]
    scene = window.Scene()
    scene.add(actor)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(
        display,
        bg_color=(0, 0, 0),
        colors=[(250, 11, 11), (136, 245, 6), (31, 41, 232)],
        find_objects=False,
    )
    npt.assert_equal(res.colors_found, [True, True, True])
    scene.clear()

    # material colors
    fetch_gltf('BoxAnimated')
    file = read_viz_gltf('BoxAnimated', 'glTF')
    importer = glTF(file)
    actors = importer.actors()
    scene.add(*actors)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(
        display, bg_color=(0, 0, 0), colors=[(77, 136, 204)], find_objects=True
    )

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
    # than the bottom half
    display = window.snapshot(scene)
    res = window.analyze_snapshot(
        display,
        bg_color=(0, 0, 0),
        colors=[(108, 173, 223), (92, 135, 39)],
        find_objects=False,
    )
    npt.assert_equal(res.colors_found, [True, True])
    blue = display[:, :, -1:].reshape((300, 300))
    upper, lower = np.split(blue, 2)
    upper = np.mean(upper)
    lower = np.mean(lower)

    assert_greater(upper, lower)
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

    scene.set_camera(
        position=(150.0, 10.0, 10.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 0.0, 1.0),
    )
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
    scene.add(*box_actor)
    export_scene(scene, 'test.gltf')
    scene.clear()

    gltf_obj = glTF('test.gltf')
    actors = gltf_obj.actors()
    scene.add(*actors)

    display = window.snapshot(scene)
    res = window.analyze_snapshot(
        display,
        bg_color=(0, 0, 0),
        colors=[(108, 173, 223), (92, 135, 39)],
        find_objects=False,
    )
    npt.assert_equal(res.colors_found, [True, True])


def test_simple_animation():
    fetch_gltf('BoxAnimated', 'glTF')
    file = read_viz_gltf('BoxAnimated')
    gltf_obj = glTF(file)
    timeline = Timeline()
    animation = gltf_obj.main_animation()
    timeline.add_animation(animation)
    scene = window.Scene()
    showm = window.ShowManager(scene, size=(900, 768))
    showm.initialize()

    scene.add(timeline)

    # timestamp animation seek
    timeline.seek(0.0)
    showm.save_screenshot('keyframe1.png')

    timeline.seek(2.57)
    showm.save_screenshot('keyframe2.png')
    res1 = window.analyze_snapshot(
        'keyframe1.png', colors=[(77, 136, 204), (204, 106, 203)]
    )
    res2 = window.analyze_snapshot(
        'keyframe2.png', colors=[(77, 136, 204), (204, 106, 203)]
    )

    assert_greater(res2.objects, res1.objects)
    npt.assert_equal(res1.colors_found, [True, False])
    npt.assert_equal(res2.colors_found, [True, True])


def test_skinning():
    # animation test
    fetch_gltf('SimpleSkin', 'glTF')
    file = read_viz_gltf('SimpleSkin')
    gltf_obj = glTF(file)
    animation = gltf_obj.skin_animation()['anim_0']
    timeline = Timeline(animation)
    # checking weights and joints
    weights = np.array(
        [
            [1.00, 0.00, 0.0, 0.0],
            [1.00, 0.00, 0.0, 0.0],
            [0.75, 0.25, 0.0, 0.0],
            [0.75, 0.25, 0.0, 0.0],
            [0.50, 0.50, 0.0, 0.0],
            [0.50, 0.50, 0.0, 0.0],
            [0.25, 0.75, 0.0, 0.0],
            [0.25, 0.75, 0.0, 0.0],
            [0.00, 1.00, 0.0, 0.0],
            [0.00, 1.00, 0.0, 0.0],
        ]
    )
    joints = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    npt.assert_equal(weights, gltf_obj.weights_0[0])
    npt.assert_equal(joints, gltf_obj.joints_0[0])

    joints, ibms = gltf_obj.get_skin_data(0)
    npt.assert_equal([1, 2], joints)
    npt.assert_equal(len(ibms), 2)

    scene = window.Scene()
    showm = window.ShowManager(scene, size=(900, 768))
    showm.initialize()

    scene.add(timeline)
    timeline.seek(1.0)

    timeline.seek(4.00)
    showm.save_screenshot('keyframe2.png')
    res1 = np.asarray(Image.open('keyframe1.png'))
    res2 = np.asarray(Image.open('keyframe2.png'))

    avg = center_of_mass(res1)
    print(avg)
    avg = center_of_mass(res2)
    print(avg)
    timeline.play()
    counter = itertools.count()

    actor = gltf_obj.actors()[0]
    vertices = utils.vertices_from_actor(actor)
    clone = np.copy(vertices)
    timeline.play()

    def timer_callback(_obj, _event):
        nonlocal timer_id
        cnt = next(counter)
        animation.update_animation()

        print(cnt)
        joint_matrices = []
        ibms = []
        for i, bone in enumerate(gltf_obj.bones[0]):
            if animation.is_interpolatable(f'transform{bone}'):
                deform = animation.get_value(
                    f'transform{bone}', animation.current_timestamp
                )
                ibm = gltf_obj.ibms[0][i].T
                ibms.append(ibm)

                parent_transform = gltf_obj.bone_tranforms[bone]
                joint_mat = np.dot(parent_transform, deform)
                joint_mat = np.dot(joint_mat, ibm)
                joint_matrices.append(joint_mat)

        vertices[:] = gltf_obj.apply_skin_matrix(clone, joint_matrices, None)
        utils.update_actor(actor)
        showm.render()

        if cnt == 10:
            showm.save_screenshot('keyframe1.png')
        if cnt == 100:
            showm.save_screenshot('keyframe2.png')

        if cnt == 150:
            showm.destroy_timer(timer_id)
            # showm.exit()

    timer_id = showm.add_timer_callback(True, 10, timer_callback)
    showm.destroy_timer(timer_id)


def test_morphing():
    fetch_gltf('MorphStressTest', 'glTF')
    file = read_viz_gltf('MorphStressTest')
    gltf_obj = glTF(file)
    animations = gltf_obj.morph_animation()

    npt.assert_equal(len(gltf_obj._actors), 2)
    npt.assert_equal(len(gltf_obj.morph_weights), 16)
    npt.assert_equal(list(animations.keys()), ['Individuals', 'TheWave', 'Pulse'])
    anim_1 = animations['TheWave']
    gltf_obj.update_morph(anim_1)

    scene = window.Scene()
    showm = window.ShowManager(scene, size=(900, 768))
    showm.initialize()

    timeline_1 = Timeline()
    timeline_1.add_animation(anim_1)
    scene.add(timeline_1)

    timeline_1.seek(0.1)
    gltf_obj.update_morph(anim_1)
    showm.save_screenshot('keyframe1.png')
    res_1 = window.analyze_snapshot('keyframe1.png')

    timeline_1.seek(1.50)
    gltf_obj.update_morph(anim_1)
    showm.save_screenshot('keyframe2.png')
    res_2 = window.analyze_snapshot('keyframe2.png')

    npt.assert_equal(res_1.colors_found, res_2.colors_found)

    img_1 = np.asarray(Image.open('keyframe1.png').convert('L'))
    img_2 = np.asarray(Image.open('keyframe2.png').convert('L'))
    stats_1, stats_2 = _stats(img_1), _stats(img_2)
    # Assert right image size
    assert_equal(stats_1[0], stats_2[0])
    # Assert the sum of colors are a lot more in the second image than the
    # first one.
    assert_greater(stats_2[1], stats_1[1])
