from unittest.mock import patch

import numpy as np
import pytest

from fury.actor import sphere
from fury.io import load_image
from fury.lib import (
    AmbientLight,
    DirectionalLight,
    OffscreenCanvas,
    OrbitController,
    PerspectiveCamera,
    Renderer,
    Scene as GfxScene,
    ScreenCoordsCamera,
    Texture,
)
from fury.ui import Rectangle2D
from fury.window import (
    Scene,
    ShowManager,
    calculate_screen_sizes,
    create_screen,
    show,
    update_camera,
    update_viewports,
)


@pytest.fixture
def sample_actor():
    "Fixture to provide a simple actor."
    actor = sphere(np.zeros((1, 3)))
    return actor


@pytest.fixture
def sample_ui_actor():
    "Fixture to provide a simple ui actor."
    actor = Rectangle2D(size=(5, 5))
    return actor


def test_scene_initialization_default():
    """Test Scene initialization with default parameters."""
    scene = Scene()
    assert scene.background == (0, 0, 0, 1)
    assert len(scene.lights) == 1
    assert isinstance(scene.lights[0], AmbientLight)
    assert isinstance(scene.main_scene, GfxScene)
    assert isinstance(scene.ui_scene, GfxScene)
    assert isinstance(scene.ui_camera, ScreenCoordsCamera)


def test_scene_initialization_custom_background():
    """Test Scene initialization with custom background."""
    scene = Scene(background=(1, 1, 1, 1))
    assert scene.background == (1, 1, 1, 1)


def test_scene_initialization_with_lights():
    """Test Scene initialization with custom lights."""
    lights = [AmbientLight(), DirectionalLight()]
    scene = Scene(lights=lights)
    assert len(scene.lights) == 2
    assert isinstance(scene.lights[0], AmbientLight)
    assert isinstance(scene.lights[1], DirectionalLight)


def test_scene_initialization_with_skybox():
    """Test Scene initialization with a skybox."""
    skybox = Texture(
        np.zeros((100, 100, 3), dtype=np.uint8), dim=2, size=(100, 100)
    )  # Mock skybox
    scene = Scene(skybox=skybox)
    assert scene._bg_actor is not None


def test_scene_set_skybox():
    """Test setting a skybox."""
    scene = Scene()
    skybox = Texture(
        np.zeros((100, 100, 3), dtype=np.uint8), dim=2, size=(100, 100)
    )  # Mock skybox
    scene.set_skybox(skybox)
    assert scene._bg_actor is not None


def test_scene_clear(sample_actor, sample_ui_actor):
    """Test clearing the scene. Should only remove the actors."""
    scene = Scene()
    scene.add(sample_actor)
    scene.add(sample_ui_actor)
    assert len(scene.main_scene.children) == 3  # Background + actor + AmbientLight
    assert len(scene.ui_scene.children) == 2  #  ui camera + ui actor
    scene.clear()
    assert len(scene.main_scene.children) == 2  # Background + AmbientLight
    assert len(scene.ui_scene.children) == 1


def test_screen_initialization_default():
    """Test Screen initialization with default parameters."""
    renderer = Renderer(OffscreenCanvas())
    screen = create_screen(renderer)
    assert screen.size == (640, 480)  # Default size of pygfx
    assert screen.position == (0, 0)  # Default position of pygfx
    assert isinstance(screen.camera, PerspectiveCamera)
    assert isinstance(screen.controller, OrbitController)

    assert (
        len(screen.scene.main_scene.children) == 3
    )  # Background + AmbientLight + Camera
    assert len(screen.scene.ui_scene.children) == 1  # Camera
    # Directional Light
    assert len(screen.camera.children) == 1


def test_screen_initialization_custom():
    """Test Screen initialization with custom parameters."""
    renderer = Renderer(OffscreenCanvas())
    scene = Scene()
    camera = PerspectiveCamera(75)
    controller = OrbitController(camera)
    screen = create_screen(renderer, scene=scene, camera=camera, controller=controller)
    assert screen.scene == scene
    assert screen.camera == camera
    assert screen.controller == controller

    assert len(screen.scene.main_scene.children) == 2  # Background + AmbientLight
    assert len(screen.scene.ui_scene.children) == 1  # Camera


def test_screen_bounding_box():
    """Test setting and getting the bounding box."""
    renderer = Renderer(OffscreenCanvas())
    screen = create_screen(renderer)
    screen.bounding_box = (100, 100, 600, 600)
    assert screen.bounding_box == (100, 100, 600, 600)


def test_show_manager_initialization_default():
    """Test ShowManager initialization with default parameters."""
    show_m = ShowManager(window_type="offscreen")
    assert show_m.title == "FURY 2.0"
    assert show_m.size == (800, 800)
    assert show_m.pixel_ratio == 1
    assert show_m.enable_events is True


def test_show_manager_initialization_custom():
    """Test ShowManager initialization with custom parameters."""
    show_m = ShowManager(
        title="Custom Title",
        size=(1024, 768),
        pixel_ratio=2,
        enable_events=False,
        screen_config=[2],  # Two vertical sections
        window_type="offscreen",
    )
    assert show_m.title == "Custom Title"
    assert show_m.size == (1024, 768)
    assert show_m.pixel_ratio == 2
    assert show_m.enable_events is False
    assert show_m._total_screens == 2  # Two screens
    assert len(show_m.screens) == 2


def test_show_manager_initialization_multiple_screens():
    """Test ShowManager initialization with multiple screens."""
    show_m = ShowManager(
        screen_config=[2, 3], window_type="offscreen"
    )  # 2 vertical sections, 3 horizontal sections
    assert show_m._total_screens == 5  # 2 + 3 screens
    assert len(show_m.screens) == 5


def test_show_manager_initialization_custom_scene_camera():
    """Test ShowManager initialization with custom scene and camera."""
    scene = Scene()
    camera = PerspectiveCamera(75)
    show_m = ShowManager(scene=scene, camera=camera, window_type="offscreen")
    assert show_m.screens[0].scene == scene
    assert show_m.screens[0].camera == camera


def test_show_manager_initialization_default_window():
    """Test ShowManager initialization with a Qt window."""
    show_m = ShowManager()
    assert show_m._is_qt is False
    assert show_m.app is None


# @pytest.mark.skipif(
#     not (have_py_side6 or have_py_qt6 or have_py_qt5), reason="Needs Qt"
# )
# def test_show_manager_initialization_qt_window():
#     """Test ShowManager initialization with a Qt window."""
#     show_m = ShowManager(window_type="qt")
#     assert show_m._is_qt is True
#     assert show_m.app is not None


def test_show_manager_screen_setup():
    """Test screen setup with custom scenes and cameras."""
    scene1 = Scene()
    scene2 = Scene()
    camera1 = PerspectiveCamera(50)
    camera2 = PerspectiveCamera(75)
    show_m = ShowManager(
        scene=[scene1, scene2],
        camera=[camera1, camera2],
        screen_config=[1, 1],
        window_type="offscreen",
    )
    assert len(show_m.screens) == 2
    assert show_m.screens[0].scene == scene1
    assert show_m.screens[1].scene == scene2
    assert show_m.screens[0].camera == camera1
    assert show_m.screens[1].camera == camera2


def test_show_manager_update_viewports():
    """Test updating screen viewports."""
    show_m = ShowManager(screen_config=[2], window_type="offscreen")  # Two screens
    new_bbs = [(0, 0, 400, 800), (400, 0, 400, 800)]  # Split window vertically
    update_viewports(show_m.screens, new_bbs)
    for screen, bb in zip(show_m.screens, new_bbs, strict=False):
        assert screen.bounding_box == bb


def test_show_manager_calculate_screen_sizes():
    """Test calculating screen sizes based on configuration."""
    screen_config = [2, 3]  # 2 vertical sections, 3 horizontal sections
    window_size = (800, 800)
    screen_bbs = calculate_screen_sizes(screen_config, window_size)
    np.testing.assert_array_almost_equal(
        screen_bbs[2], (400, 0, 400, 266.66), decimal=2
    )
    assert len(screen_bbs) == 5  # 2 + 3 screens
    assert screen_bbs[0] == (0, 0, 400, 400)  # First screen
    assert screen_bbs[1] == (0, 400, 400, 400)  # Second screen
    np.testing.assert_array_almost_equal(
        screen_bbs[2], (400, 0, 400, 266.66), decimal=2
    )  # Third screen
    np.testing.assert_array_almost_equal(
        screen_bbs[3], (400, 266.66, 400, 266.66), decimal=2
    )  # Fourth screen
    np.testing.assert_array_almost_equal(
        screen_bbs[4], (400, 266.66 * 2, 400, 266.66), decimal=2
    )  # Fifth screen


def test_show_manager_set_enable_events():
    """Test enabling and disabling events."""
    show_m = ShowManager(window_type="offscreen")
    show_m.set_enable_events(False)
    assert show_m.enable_events is False
    for screen in show_m.screens:
        assert screen.controller.enabled is False

    show_m.set_enable_events(True)
    assert show_m.enable_events is True
    for screen in show_m.screens:
        assert screen.controller.enabled is True


def test_show_manager_update_camera(sample_actor):
    """Test updating the camera to face the target and show the size if empty scene."""
    scene = Scene()
    show_m = ShowManager(scene=scene, window_type="offscreen")
    update_camera(show_m.screens[0].camera, show_m.size, scene)
    assert show_m.screens[0].camera.width == 800
    assert show_m.screens[0].camera.height == 800

    scene.add(sample_actor)
    update_camera(show_m.screens[0].camera, show_m.size, scene)
    assert show_m.screens[0].camera.width != 800
    assert show_m.screens[0].camera.height != 800


def test_show_manager_snapshot(tmpdir):
    """Test taking a snapshot of the scene."""
    fname = tmpdir.join("snapshot.png")

    show_m = ShowManager(window_type="offscreen")
    show_m.render()
    show_m.window.draw_frame()
    arr = show_m.snapshot(str(fname))

    saved_arr = load_image(str(fname))

    assert isinstance(arr, np.ndarray)
    assert arr.shape[2] == 4  # RGBA image
    np.testing.assert_equal(arr, saved_arr)


def test_show_manager_snapshot_multiple_screens(tmpdir):
    """Test taking a snapshot with multiple screens."""
    show_m = ShowManager(screen_config=[2], window_type="offscreen")  # Two screens
    fname = tmpdir.join("snapshot_multiple.png")
    show_m.render()
    show_m.window.draw_frame()
    arr = show_m.snapshot(str(fname))
    saved_arr = load_image(str(fname))
    assert isinstance(arr, np.ndarray)
    assert arr.shape[2] == 4  # RGBA image
    np.testing.assert_equal(arr, saved_arr)


def test_show_manager_invalid_window_type():
    """Test initialization with an invalid window type."""
    with pytest.raises(ValueError):
        ShowManager(window_type="invalid")


def test_show_manager_empty_scene():
    """Test initialization with an empty scene."""
    show_m = ShowManager(scene=Scene(), window_type="offscreen")
    assert (
        len(show_m.screens[0].scene.main_scene.children) == 3
    )  # Background + AmbientLight + Camera
    assert len(show_m.screens[0].scene.ui_scene.children) == 1  # UI Camera


def test_show_manager_with_empty_config():
    """Test initialization with empty screen config."""
    show_m = ShowManager(screen_config=[], window_type="offscreen")
    assert show_m._total_screens == 1
    assert len(show_m.screens) == 1


def test_display_default(sample_actor):
    """Test the display function with default parameters."""

    with patch("fury.window.ShowManager") as mock_show_manager:
        show([sample_actor])
        mock_show_manager.assert_called_once()


def test_add_remove_ui_to_from_scene(sample_actor):
    """Test add/remove ui hierarchy to/from scene."""

    parent = Rectangle2D()
    child_1, child_2, child_3 = (
        Rectangle2D(),
        Rectangle2D(),
        Rectangle2D(),
    )
    subchild_21 = Rectangle2D()
    subchild_31, subchild_32 = Rectangle2D(), Rectangle2D()

    parent.children.extend([child_1, child_2, child_3])
    child_2.children.append(subchild_21)
    child_3.children.extend([subchild_31, subchild_32])

    all_ui_objects = [
        parent,
        child_1,
        child_2,
        child_3,
        subchild_21,
        subchild_31,
        subchild_32,
    ]
    all_ui_actors = []
    for ui_obj in all_ui_objects:
        all_ui_actors.extend(ui_obj.actors)

    scene = Scene()

    # Test Add and Remove Parent
    scene.add(parent)

    assert len(scene.ui_scene.children) == 1 + len(all_ui_actors)
    assert parent in scene.ui_elements
    assert all(actor in scene.ui_scene.children for actor in all_ui_actors)

    scene.remove(parent)

    assert len(scene.ui_elements) == 0
    assert len(scene.ui_scene.children) == 1  # UI Camera
    assert all(actor not in scene.ui_scene.children for actor in all_ui_actors)

    # Test Add and Clear Parent
    scene.add(parent)

    assert len(scene.ui_scene.children) == 1 + len(all_ui_actors)
    assert len(scene.ui_elements) == 1

    scene.clear()

    assert len(scene.ui_scene.children) == 1  # UI camera
    assert len(scene.ui_elements) == 0

    # Add Parent, Remove Child
    scene.add(parent)

    to_be_removed_ui = [child_3, subchild_31, subchild_32]
    to_be_removed_actors = []
    for ui_obj in to_be_removed_ui:
        to_be_removed_actors.extend(ui_obj.actors)

    should_remain_ui = [parent, child_1, child_2, subchild_21]
    should_remain_actors = []
    for ui_obj in should_remain_ui:
        should_remain_actors.extend(ui_obj.actors)

    assert len(scene.ui_elements) == 1
    assert len(scene.ui_scene.children) == 1 + len(
        all_ui_actors
    )  # UI Camera + UI actors

    scene.remove(child_3)

    assert parent in scene.ui_elements
    for ui_obj in to_be_removed_ui:
        assert all(actor not in scene.ui_scene.children for actor in ui_obj.actors)

    for ui_obj in should_remain_ui:
        assert all(actor in scene.ui_scene.children for actor in ui_obj.actors)

    assert len(scene.ui_elements) == 1
    assert len(scene.ui_scene.children) == 1 + len(should_remain_actors)

    # Remove Non Existent Child
    scene.remove(subchild_31)

    assert parent in scene.ui_elements
    for ui_obj in to_be_removed_ui:
        assert all(actor not in scene.ui_scene.children for actor in ui_obj.actors)

    for ui_obj in should_remain_ui:
        assert all(actor in scene.ui_scene.children for actor in ui_obj.actors)

    assert len(scene.ui_elements) == 1
    assert len(scene.ui_scene.children) == 1 + len(should_remain_actors)


def test_add_to_scene(sample_actor, sample_ui_actor):
    """Test add/remove ui hierarchy to/from scene."""

    scene = Scene()

    sample_gfx_scene = GfxScene()
    sample_camera = PerspectiveCamera()
    scene.add(sample_actor, sample_ui_actor, sample_gfx_scene, sample_camera)

    assert sample_actor in scene.main_scene.children
    assert sample_camera in scene.main_scene.children
    assert sample_ui_actor in scene.ui_elements
    assert sample_ui_actor.actors[0] in scene.ui_scene.children
    assert sample_gfx_scene in scene.children

    scene.remove(sample_actor, sample_ui_actor, sample_gfx_scene, sample_camera)

    assert sample_actor not in scene.main_scene.children
    assert sample_camera not in scene.main_scene.children
    assert sample_ui_actor not in scene.ui_elements
    assert sample_ui_actor.actors[0] not in scene.ui_scene.children
    assert sample_gfx_scene not in scene.children
