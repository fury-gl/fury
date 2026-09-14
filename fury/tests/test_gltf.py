import numpy as np
import numpy.testing as npt
import pytest

from fury import window
from fury.actor import Mesh, SkinnedMesh
from fury.data import fetch_gltf, read_viz_gltf
from fury.gltf import GLTFAnimation, glTF, have_gltflib, load_gltf, load_gltf_mesh
from fury.lib import gfx
from fury.motion import Timeline
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


def test_gltf_scene():
    fetch_gltf(name="Duck")
    filename = read_viz_gltf("Duck")
    gltf_obj = glTF(filename)

    npt.assert_equal(isinstance(gltf_obj.scene, gfx.Group), True)
    npt.assert_equal(len(gltf_obj.scenes), 1)
    npt.assert_equal(len(gltf_obj.cameras), 1)
    npt.assert_equal(gltf_obj.lights, [])


def test_gltf_actors():
    fetch_gltf(name="Duck")
    filename = read_viz_gltf("Duck")
    gltf_obj = glTF(filename)
    actors = gltf_obj.actors()

    npt.assert_equal(len(actors), 1)
    npt.assert_equal(isinstance(actors[0], Mesh), True)
    npt.assert_equal(actors[0].geometry.positions.data.shape, (2399, 3))


def test_gltf_actors_are_scene_nodes():
    fetch_gltf(name="Duck")
    filename = read_viz_gltf("Duck")
    gltf_obj = glTF(filename)
    actor = gltf_obj.actors()[0]

    npt.assert_equal(actor in list(gltf_obj.scene.iter()), True)
    npt.assert_equal(gltf_obj.actors() is gltf_obj.actors(), True)


def test_gltf_skinned_actors():
    fetch_gltf(name="RiggedFigure")
    filename = read_viz_gltf("RiggedFigure")
    gltf_obj = glTF(filename)
    actors = gltf_obj.actors()

    npt.assert_equal(len(actors), 1)
    npt.assert_equal(isinstance(actors[0], SkinnedMesh), True)
    npt.assert_equal(isinstance(actors[0], gfx.SkinnedMesh), True)


def test_gltf_renders():
    fetch_gltf(name="Duck")
    filename = read_viz_gltf("Duck")
    gltf_obj = glTF(filename)

    scene = window.Scene()
    scene.add(*gltf_obj.actors())
    image = window.snapshot(scene=scene, fname=None, return_array=True)

    npt.assert_equal(image.shape[-1], 4)
    npt.assert_equal(image[..., :3].any(), True)


def test_gltf_animations():
    fetch_gltf(name="Fox")
    filename = read_viz_gltf("Fox")
    gltf_obj = glTF(filename)

    npt.assert_equal(list(gltf_obj.animations), ["Survey", "Walk", "Run"])
    npt.assert_equal(isinstance(gltf_obj.animations["Walk"], GLTFAnimation), True)
    npt.assert_equal(gltf_obj.animations is gltf_obj.animations, True)


def test_gltf_animations_unnamed():
    fetch_gltf(name="BoxAnimated")
    filename = read_viz_gltf("BoxAnimated")
    gltf_obj = glTF(filename)

    npt.assert_equal(list(gltf_obj.animations), ["anim_0"])


def test_gltf_animations_absent():
    fetch_gltf(name="Duck")
    filename = read_viz_gltf("Duck")
    gltf_obj = glTF(filename)

    npt.assert_equal(gltf_obj.animations, {})
    npt.assert_equal(gltf_obj.main_animation().duration, 0.0)


def test_gltf_animation_duration():
    fetch_gltf(name="BoxAnimated")
    filename = read_viz_gltf("BoxAnimated")
    animation = glTF(filename).animations["anim_0"]

    npt.assert_almost_equal(animation.duration, 3.7083333, decimal=5)
    npt.assert_equal(animation.name, None)


def test_gltf_main_animation():
    fetch_gltf(name="BoxAnimated")
    filename = read_viz_gltf("BoxAnimated")
    gltf_obj = glTF(filename)
    timeline = gltf_obj.main_animation()

    npt.assert_equal(isinstance(timeline, Timeline), True)
    npt.assert_equal(timeline.animations, list(gltf_obj.animations.values()))
    npt.assert_almost_equal(timeline.duration, 3.7083333, decimal=5)
    npt.assert_equal(timeline.has_playback_panel, False)

    panelled = gltf_obj.main_animation(playback_panel=True)
    npt.assert_equal(panelled.has_playback_panel, True)


def test_gltf_animation_moves_nodes():
    fetch_gltf(name="BoxAnimated")
    filename = read_viz_gltf("BoxAnimated")
    gltf_obj = glTF(filename)
    timeline = gltf_obj.main_animation()
    actor = gltf_obj.actors()[-1]

    timeline.seek(0.0)
    timeline.update(force=True)
    start = np.array(actor.world.position, copy=True)

    timeline.seek(timeline.duration / 2)
    timeline.update(force=True)
    middle = np.array(actor.world.position, copy=True)

    npt.assert_equal(np.allclose(start, middle), False)


def test_gltf_animation_drives_skeleton():
    fetch_gltf(name="RiggedFigure")
    filename = read_viz_gltf("RiggedFigure")
    gltf_obj = glTF(filename)
    timeline = gltf_obj.main_animation()
    bones = [obj for obj in gltf_obj.scene.iter() if isinstance(obj, gfx.Bone)]

    timeline.seek(0.0)
    timeline.update(force=True)
    start = [np.array(bone.local.matrix, copy=True) for bone in bones]

    timeline.seek(timeline.duration / 2)
    timeline.update(force=True)
    moved = [
        not np.allclose(before, bone.local.matrix)
        for before, bone in zip(start, bones, strict=True)
    ]

    npt.assert_equal(len(bones), 19)
    npt.assert_equal(any(moved), True)


def test_gltf_animation_drives_morph_weights():
    fetch_gltf(name="AnimatedMorphCube")
    filename = read_viz_gltf("AnimatedMorphCube")
    gltf_obj = glTF(filename)
    timeline = gltf_obj.main_animation()
    mesh = gltf_obj.actors()[0]

    timeline.seek(0.0)
    timeline.update(force=True)
    start = np.array(mesh.morph_target_influences, copy=True)

    timeline.seek(timeline.duration / 4)
    timeline.update(force=True)
    later = np.array(mesh.morph_target_influences, copy=True)

    npt.assert_equal(np.allclose(start, later), False)


def test_gltf_animation_adds_scene_to_window():
    fetch_gltf(name="BoxAnimated")
    filename = read_viz_gltf("BoxAnimated")
    gltf_obj = glTF(filename)
    timeline = gltf_obj.main_animation()

    scene = window.Scene()
    timeline.add_to_scene(scene)

    npt.assert_equal(gltf_obj.scene in scene.main_scene.children, True)
