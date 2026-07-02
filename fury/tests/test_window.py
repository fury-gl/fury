import logging
import os
from unittest.mock import patch

import numpy as np
import pytest

from fury import actor, window
from fury.actor import sphere
from fury.io import load_image
from fury.lib import (
    AmbientLight,
    DirectionalLight,
    EventType,
    OffscreenCanvas,
    OrbitController,
    PerspectiveCamera,
    PointerEvent,
    QtWidgets,
    Renderer,
    Scene as GfxScene,
    ScreenCoordsCamera,
    Texture,
    TrackballController,
    have_imgui_bundle,
    have_py_side6,
)
from fury.motion import Animation, CameraAnimation, Timeline
from fury.ui import Rectangle2D, UIContext
from fury.window import (
    Scene,
    ShowManager,
    _get_scene_center,
    _reference_up_for_axis,
    analyze_snapshot,
    calculate_screen_sizes,
    create_screen,
    set_camera_from_axis,
    show,
    update_camera,
    update_viewports,
)


@pytest.fixture
def sample_actor():
    "Fixture to provide a simple actor."
    actor = sphere(np.zeros((1, 3)), material="basic", impostor=False)
    return actor


@pytest.mark.skipif(not window.have_cv2, reason="OpenCV is required for mp4 export")
def test_show_manager_record_animation(tmp_path):
    scene = Scene()
    show_m = ShowManager(scene=scene, size=(64, 64), window_type="offscreen")
    timeline = Timeline(length=0.4, loop=False)
    cube = actor.box(
        np.array([[0, 0, 0]]),
        colors=np.array([[1, 0, 0]]),
        scales=np.array([[1, 1, 1]]),
    )
    animation = Animation(actors=cube)
    original_camera = PerspectiveCamera()
    camera_animation = CameraAnimation(camera=original_camera, loop=False)
    camera_animation.set_position(0, np.array([3, 3, 3]))
    camera_animation.set_position(0.2, np.array([5, 3, 3]))
    camera_animation.set_focal(0, np.array([0, 0, 0]))
    camera_animation.set_focal(0.2, np.array([0, 0, 0]))
    timeline.add_animation([animation, camera_animation])
    show_m.add_animation(timeline)

    try:
        fname = tmp_path / "animation.mp4"
        frames = timeline.record(fname, fps=5, return_frames=True)

        assert fname.exists()
        assert fname.stat().st_size > 0
        assert len(frames) == 2
        assert frames[0].shape == (64, 64, 4)
        assert not np.array_equal(frames[0], frames[1])
        assert camera_animation.camera is original_camera
        assert timeline.playing
    finally:
        show_m.window.close()


def test_show_manager_record_animation_validates_params():
    show_m = ShowManager(size=(2, 2), window_type="offscreen")
    anim = Animation()

    try:
        with pytest.raises(ValueError, match="fps"):
            show_m.record_animation(anim, "animation.mp4", fps=0)

        with pytest.raises(ValueError, match="speed"):
            show_m.record_animation(anim, "animation.mp4", speed=0)
    finally:
        show_m.window.close()


def test_show_manager_record_callback_propagates_to_child_animations():
    show_m = ShowManager(size=(2, 2), window_type="offscreen")
    timeline = Timeline()
    parent_animation = Animation()
    child_animation = Animation()
    late_child_animation = Animation()
    parent_animation.add_child_animation(child_animation)
    timeline.add_animation(parent_animation)

    try:
        show_m.add_animation(timeline)

        assert timeline._record_callback == show_m.record_animation
        assert parent_animation._record_callback == show_m.record_animation
        assert child_animation._record_callback == show_m.record_animation

        parent_animation.add_child_animation(late_child_animation)
        assert late_child_animation._record_callback == show_m.record_animation

        with pytest.raises(ValueError, match="fps"):
            child_animation.record("child.mp4", fps=0)

        show_m.remove_animation(timeline)
        assert timeline._record_callback is None
        assert parent_animation._record_callback is None
        assert child_animation._record_callback is None
        assert late_child_animation._record_callback is None
    finally:
        show_m.window.close()


@pytest.fixture
def sample_ui_actor():
    "Fixture to provide a simple ui actor."
    actor = Rectangle2D(size=(5, 5))
    return actor


@pytest.fixture
def timeline():
    return Timeline(length=1)


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


def test_get_scene_center_uses_bounding_box():
    """Ensure bounding box midpoint is used when available."""
    bbox = np.array([0, 0, 800.0, 800.0], dtype=np.float32)
    renderer = Renderer(OffscreenCanvas())
    screen = create_screen(renderer, rect=bbox)
    center = _get_scene_center(screen.camera, screen.scene)
    # Empty scene use cam.forward as center.
    np.testing.assert_array_equal(center, [0, 0, -1])


def test_get_scene_center_defaults_to_camera_direction():
    """
    Fallback should use camera position and forward vector when bbox is
    missing.
    """
    renderer = Renderer(OffscreenCanvas())
    screen = create_screen(renderer)
    expected = screen.camera.world.position + screen.camera.world.forward
    center = _get_scene_center(screen.camera, screen.scene)
    np.testing.assert_array_equal(center, expected)


@pytest.mark.parametrize(
    "axis, expected",
    [
        (
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        ),
        (
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
        (
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ),
    ],
)
def test_reference_up_for_axis(axis, expected):
    """
    Reference up vector should switch near the poles to avoid gimbal
    lock.
    """
    np.testing.assert_array_equal(_reference_up_for_axis(axis), expected)


def test_set_camera_from_axis_updates_camera_and_controller_target():
    """Camera snap via axes helper should reposition camera and controller."""
    bbox = np.array([0, 0, 800.0, 800.0], dtype=np.float32)
    screen = create_screen(Renderer(OffscreenCanvas()), rect=bbox)
    set_camera_from_axis(screen, (1.0, 0.0, 0.0))

    np.testing.assert_array_equal(
        screen.camera.world.reference_up, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )


def test_show_axes_gizmo_click_callback_invoked_with_axis_direction():
    """
    Custom click callbacks should receive the clicked axis direction
    vector.
    """
    bbox = [[0, 0, 800.0, 800.0]]
    show_m = ShowManager(window_type="offscreen", screen_config=bbox)
    captured_axes = []

    show_m.show_axes_gizmo(
        click_callback=lambda axis: captured_axes.append(axis.copy())
    )
    show_m.renderer.dispatch_event(
        PointerEvent(
            x=-1, y=-1, type="pointer_down", target=show_m._axes_helper.children[1]
        )
    )

    assert len(captured_axes) == 1


def test_show_axes_gizmo_defaults_to_noop_click_callback_when_invalid():
    """Non-callable click callbacks should fall back to a no-op function."""
    bbox = [[0, 0, 800.0, 800.0]]
    show_m = ShowManager(window_type="offscreen", screen_config=bbox)
    invalid_callback = "not a function"

    show_m.show_axes_gizmo(click_callback=invalid_callback)

    assert callable(show_m._axes_helper_click_callback)
    assert show_m._axes_helper_click_callback != invalid_callback
    assert (
        show_m._axes_helper_click_callback(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        is None
    )


def test_scene_clear(sample_actor, sample_ui_actor):
    """
    Test clearing the scene.

    Should only remove the actors.
    """
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
    assert isinstance(screen.controller, TrackballController)

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
    assert show_m.pixel_ratio != 1.25
    assert float(show_m.pixel_ratio).is_integer()
    assert show_m.enable_events is True
    assert show_m._show_fps is False
    assert show_m._max_fps == 60


def test_show_manager_initialization_custom():
    """Test ShowManager initialization with custom parameters."""
    show_m = ShowManager(
        title="Custom Title",
        size=(1024, 768),
        pixel_ratio=2,
        enable_events=False,
        screen_config=[2],  # Two vertical sections
        window_type="offscreen",
        show_fps=True,
        max_fps=120,
    )
    assert show_m.title == "Custom Title"
    assert show_m.size == (1024, 768)
    assert show_m.pixel_ratio == 2
    assert show_m.enable_events is False
    assert show_m._total_screens == 2  # Two screens
    assert len(show_m.screens) == 2
    assert show_m._show_fps is True
    assert show_m._max_fps == 120


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


@pytest.mark.skipif(not (have_py_side6), reason="Needs Qt")
def test_show_manager_initialization_qt_window():
    """Test ShowManager initialization with a Qt window."""
    show_m = ShowManager(window_type="qt")
    assert show_m._is_qt is True
    assert isinstance(show_m.window, QtWidgets.QWidget)


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


def test_show_manager_resize_callback():
    """Test triggering and canceling the resize callback."""
    show_m = ShowManager(window_type="offscreen")
    resize_calls = []

    def on_resize(size):
        resize_calls.append(size)

    show_m.resize_callback(on_resize)
    new_size = (640, 480)
    show_m._resize(new_size)

    assert resize_calls == [new_size]

    show_m.cancel_resize_callback()
    show_m._resize((320, 240))
    assert resize_calls == [new_size]


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


def test_show_manager_calculate_screen_sizes_explicit_bounding_boxes():
    """Test calculate_screen_sizes with explicit bounding boxes."""
    screen_config = [
        (0, 0, 400, 400),
        (400, 0, 400, 400),
        (0, 400, 400, 400),
    ]
    window_size = (800, 800)
    screen_bbs = calculate_screen_sizes(screen_config, window_size)

    assert screen_bbs == screen_config


def test_show_manager_calculate_screen_sizes_invalid_bounding_boxes(caplog):
    """Test calculate_screen_sizes with invalid bounding box format."""
    screen_config = [(0, 0, 400), (400, 0, 400, 400)]
    window_size = (800, 800)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            calculate_screen_sizes(screen_config, window_size)

    assert excinfo.value.code == 1
    assert "Invalid screen bounding box format" in caplog.text


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
    """
    Test updating the camera to face the target and show the size if empty
    scene.
    """
    scene = Scene()
    show_m = ShowManager(scene=scene, window_type="offscreen")
    update_camera(show_m.screens[0].camera, None, None)

    scene = Scene()
    show_m = ShowManager(scene=scene, window_type="offscreen")
    update_camera(show_m.screens[0].camera, show_m.size, scene)
    assert show_m.screens[0].camera.width == 800
    assert show_m.screens[0].camera.height == 800

    scene.add(sample_actor)
    update_camera(show_m.screens[0].camera, show_m.size, scene)
    assert show_m.screens[0].camera.width != 800
    assert show_m.screens[0].camera.height != 800


def test_show_manager_snapshot(tmpdir, sample_actor):
    """Test taking a snapshot of the scene."""
    fname = tmpdir.join("snapshot.png")

    scene = Scene()
    scene.add(sample_actor)

    show_m = ShowManager(scene=scene, window_type="offscreen")
    show_m.render()
    show_m.window.draw()
    arr = show_m.snapshot(fname=str(fname))

    saved_arr = load_image(str(fname))

    assert isinstance(arr, np.ndarray)
    assert arr.shape[2] == 4  # RGBA image
    np.testing.assert_equal(arr, saved_arr)

    report = analyze_snapshot(str(fname), colors=[(255, 0, 0)])
    assert report.colors_found == [True]
    assert report.objects == 1


def test_show_manager_snapshot_multiple_screens(tmpdir, sample_actor):
    """Test taking a snapshot with multiple screens."""
    fname = tmpdir.join("snapshot_multiple.png")

    scene = Scene()
    scene.add(sample_actor)

    show_m = ShowManager(
        scene=scene, screen_config=[2], window_type="offscreen"
    )  # Two screens
    show_m.render()
    show_m.window.draw()
    arr = show_m.snapshot(fname=str(fname))
    saved_arr = load_image(str(fname))

    assert isinstance(arr, np.ndarray)
    assert arr.shape[2] == 4  # RGBA image
    np.testing.assert_equal(arr, saved_arr)

    report = analyze_snapshot(str(fname), colors=[(255, 0, 0)])
    assert report.colors_found == [True]
    assert report.objects == 2


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


def test_show_manager_with_empty_title():
    """Test initialization with empty screen config."""
    show_m = ShowManager(window_type="offscreen", title=None)
    assert show_m._title == "FURY 2.0"
    show_m = ShowManager(window_type="offscreen", title="")
    assert show_m._title == "FURY 2.0"


def test_display_default(sample_actor):
    """Test the display function with default parameters."""
    with patch("fury.window.ShowManager") as mock_show_manager:
        show(sample_actor)
        mock_show_manager.assert_called_once()
        kwargs = mock_show_manager.call_args.kwargs
        assert kwargs["window_type"] == "default"
        assert kwargs["title"] == "FURY 2.0"
        assert sample_actor in kwargs["scene"].main_scene.children
        mock_show_manager.return_value.start.assert_called_once_with()


def test_display_accepts_iterable_actors(sample_actor):
    """Test the display function with a non-list iterable of actors."""
    second_actor = sphere(np.array([[1, 0, 0]]), material="basic", impostor=False)
    actors = (item for item in (sample_actor, second_actor))

    with patch("fury.window.ShowManager") as mock_show_manager:
        show(actors, window_type="offscreen", title="Iterable actors")

        kwargs = mock_show_manager.call_args.kwargs
        assert kwargs["window_type"] == "offscreen"
        assert kwargs["title"] == "Iterable actors"
        assert sample_actor in kwargs["scene"].main_scene.children
        assert second_actor in kwargs["scene"].main_scene.children
        mock_show_manager.return_value.start.assert_called_once_with()


def test_add_remove_ui_to_from_scene(sample_actor):
    """Test add/remove UI hierarchy to/from scene."""
    parent = Rectangle2D()
    child_1, child_2, child_3 = (
        Rectangle2D(),
        Rectangle2D(),
        Rectangle2D(),
    )
    subchild_21 = Rectangle2D()
    subchild_31, subchild_32 = Rectangle2D(), Rectangle2D()

    parent._children.extend([child_1, child_2, child_3])
    child_2._children.append(subchild_21)
    child_3._children.extend([subchild_31, subchild_32])

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
    """Test add/remove elements to/from scene."""
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


def test_show_manager_fps_display():
    """Test FPS display using Stats overlay."""
    show_m = ShowManager(show_fps=True)

    assert show_m._stats is None
    assert show_m._stats_initialized is False
    assert show_m.get_fps() is None

    show_m_no_fps = ShowManager(show_fps=False)
    assert show_m_no_fps._stats is None
    assert show_m_no_fps._stats_initialized is False
    assert show_m_no_fps.get_fps() is None


@pytest.mark.skipif(not have_imgui_bundle, reason="Needs Imgui Bundle")
def test_show_manager_enable_imgui_creates_renderer():
    """ImGui should be disabled by default and created on enable_imgui()."""
    show_m = ShowManager(window_type="offscreen")

    assert show_m._imgui is None

    show_m.enable_imgui()
    assert show_m._imgui is not None


@pytest.mark.skipif(not have_imgui_bundle, reason="Needs Imgui Bundle")
def test_show_manager_disable_imgui_clears_renderer():
    """disable_imgui() should clear the _imgui reference."""
    show_m = ShowManager(window_type="offscreen")

    show_m.enable_imgui()
    assert show_m._imgui is not None

    show_m.disable_imgui()
    assert show_m._imgui is None


@pytest.mark.skipif(not have_imgui_bundle, reason="Needs Imgui Bundle")
def test_show_manager_enable_imgui_idempotent():
    """Calling enable_imgui() twice should keep the same renderer instance."""
    show_m = ShowManager(window_type="offscreen")

    show_m.enable_imgui()
    first_imgui = show_m._imgui

    show_m.enable_imgui()
    second_imgui = show_m._imgui

    assert first_imgui is second_imgui


@pytest.mark.skipif(not have_imgui_bundle, reason="Needs Imgui Bundle")
def test_show_manager_set_imgui_render_callback_only_when_enabled():
    """
    set_imgui_render_callback should only wire callback once ImGui is
    enabled.
    """
    show_m = ShowManager(window_type="offscreen")

    def dummy_callback():
        pass

    show_m.set_imgui_render_callback(dummy_callback)
    assert show_m._imgui is None

    show_m.enable_imgui()
    show_m.set_imgui_render_callback(dummy_callback)
    assert show_m._imgui is not None
    assert show_m._imgui._update_gui_function == dummy_callback


def test_analyze_snapshot(tmpdir):
    """Test analyze_snapshot function."""
    # Create a test image with a black background and two colored squares
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[2:6, 2:6] = [255, 0, 0]  # Red square
    img[10:15, 10:15] = [0, 255, 0]  # Green square

    # Test 1: Find objects in the image array
    report = analyze_snapshot(img, find_objects=True)
    assert report.objects == 2
    assert report.labels.shape == (20, 20)
    assert not report.colors_found

    # Test 2: Find specific colors
    report = analyze_snapshot(
        img, colors=[(255, 0, 0), (0, 0, 255)], find_objects=False
    )
    assert report.objects is None
    assert report.labels is None
    assert report.colors_found == [True, False]

    # Test 3: Find a single color
    report = analyze_snapshot(img, colors=(0, 255, 0), find_objects=False)
    assert report.colors_found == [True]

    # Test 4: Analyze from a saved file
    fname = tmpdir.join("test_snapshot.png")
    from fury.io import save_image

    save_image(img, str(fname))
    report_from_file = analyze_snapshot(str(fname))
    assert report_from_file.objects == 2

    # Test 5: With RGBA image
    img_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    img_rgba[2:6, 2:6] = [255, 0, 0, 255]
    img_rgba[10:15, 10:15] = [0, 255, 0, 255]
    report_rgba = analyze_snapshot(img_rgba)
    assert report_rgba.objects == 2

    # Test 6: Custom structuring element
    # A larger strel might merge close objects
    strel = np.ones((3, 3))
    report_strel = analyze_snapshot(img, strel=strel)
    # Depending on the image and strel, this could be 1 or 2.
    # For this specific setup, the squares are far apart.
    assert report_strel.objects == 2


def test_show_manager_toggle_screen_controllers():
    """Test toggling the screen controllers."""
    show_m = ShowManager(window_type="offscreen")
    show_m._toggle_screen_controllers(disable=True)
    for screen in show_m.screens:
        assert screen.controller.enabled is False
    show_m._toggle_screen_controllers(disable=False)
    for screen in show_m.screens:
        assert screen.controller.enabled is True


def test_show_manager_register_drag():
    """Test pointer event handling for drag interactions."""
    show_m = ShowManager(window_type="offscreen")

    event_down = PointerEvent(x=0, y=0, type=EventType.POINTER_DOWN, target="target1")

    original_hot_ui = UIContext.hot_ui
    UIContext.hot_ui = Rectangle2D(size=(5, 5))
    try:
        show_m._register_drag(event_down)
        assert show_m._is_dragging is True
        assert show_m._drag_target == "target1"
        for screen in show_m.screens:
            assert screen.controller.enabled is False
    finally:
        UIContext.hot_ui = original_hot_ui

    drag_called = []
    show_m._handle_drag = lambda event: drag_called.append(event)

    event_move = PointerEvent(x=10, y=10, type=EventType.POINTER_MOVE, target="target1")
    show_m._register_drag(event_move)
    assert len(drag_called) == 1
    assert drag_called[0] == event_move

    event_up = PointerEvent(x=10, y=10, type=EventType.POINTER_UP, target="target1")
    show_m._register_drag(event_up)
    assert show_m._is_dragging is False
    assert show_m._drag_target is None
    for screen in show_m.screens:
        assert screen.controller.enabled is True


def test_offscreen_animation_recording(tmp_path):
    scene = window.Scene()
    scene.add(actor.axes(scale=(1, 1, 1)))

    show_m = window.ShowManager(scene=scene, size=(200, 200), title="test_anim")

    def noop_callback():
        pass

    show_m.register_callback(noop_callback, time=1.0, repeat=True, name="noop")

    original_cwd = os.getcwd()
    original_offscreen = os.environ.get("FURY_OFFSCREEN")
    original_record = os.environ.get("FURY_RECORD_ANIMATION")
    os.chdir(tmp_path)
    os.environ["FURY_OFFSCREEN"] = "1"
    os.environ["FURY_RECORD_ANIMATION"] = "0"

    try:
        show_m.start()
    finally:
        os.chdir(original_cwd)
        if original_offscreen is None:
            os.environ.pop("FURY_OFFSCREEN", None)
        else:
            os.environ["FURY_OFFSCREEN"] = original_offscreen
        if original_record is None:
            os.environ.pop("FURY_RECORD_ANIMATION", None)
        else:
            os.environ["FURY_RECORD_ANIMATION"] = original_record

    assert (tmp_path / "test_anim.png").exists()
    assert not (tmp_path / "test_anim.gif").exists()
    os.remove(tmp_path / "test_anim.png")

    original_cwd = os.getcwd()
    original_offscreen = os.environ.get("FURY_OFFSCREEN")
    original_record = os.environ.get("FURY_RECORD_ANIMATION")
    original_max_frames = os.environ.get("FURY_OFFSCREEN_MAX_FRAMES")
    os.chdir(tmp_path)
    os.environ["FURY_OFFSCREEN"] = "1"
    os.environ["FURY_RECORD_ANIMATION"] = "1"
    os.environ["FURY_OFFSCREEN_MAX_FRAMES"] = "2"

    try:
        show_m2 = window.ShowManager(scene=scene, size=(200, 200), title="test_anim")
        show_m2.register_callback(noop_callback, time=1.0, repeat=True, name="noop")
        show_m2.start()
    finally:
        os.chdir(original_cwd)
        if original_offscreen is None:
            os.environ.pop("FURY_OFFSCREEN", None)
        else:
            os.environ["FURY_OFFSCREEN"] = original_offscreen
        if original_record is None:
            os.environ.pop("FURY_RECORD_ANIMATION", None)
        else:
            os.environ["FURY_RECORD_ANIMATION"] = original_record
        if original_max_frames is None:
            os.environ.pop("FURY_OFFSCREEN_MAX_FRAMES", None)
        else:
            os.environ["FURY_OFFSCREEN_MAX_FRAMES"] = original_max_frames

    assert (tmp_path / "test_anim.gif").exists()
    assert not (tmp_path / "test_anim.png").exists()
    os.remove(tmp_path / "test_anim.gif")


@pytest.mark.skipif(
    not (have_py_side6),
    reason="A Qt binding is required for Qt offscreen capture",
)
def test_qt_offscreen_capture_saves_parent_widget(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.resize(240, 120)

    scene = window.Scene()
    scene.add(actor.axes(scale=(1, 1, 1)))

    show_m = window.ShowManager(
        scene=scene,
        size=(100, 100),
        title="qt_capture",
        window_type="qt",
        qt_app=app,
        qt_parent=parent,
    )

    layout = QtWidgets.QHBoxLayout()
    parent.setLayout(layout)
    layout.addWidget(QtWidgets.QPushButton("Capture", parent))
    layout.addWidget(show_m.window)

    original_cwd = os.getcwd()
    original_offscreen = os.environ.get("FURY_OFFSCREEN")
    original_record = os.environ.get("FURY_RECORD_ANIMATION")
    os.chdir(tmp_path)
    os.environ["FURY_OFFSCREEN"] = "1"
    os.environ["FURY_RECORD_ANIMATION"] = "0"

    try:
        show_m.start()
    finally:
        os.chdir(original_cwd)
        if original_offscreen is None:
            os.environ.pop("FURY_OFFSCREEN", None)
        else:
            os.environ["FURY_OFFSCREEN"] = original_offscreen
        if original_record is None:
            os.environ.pop("FURY_RECORD_ANIMATION", None)
        else:
            os.environ["FURY_RECORD_ANIMATION"] = original_record

    fname = tmp_path / "qt_capture.png"
    assert fname.exists()
    image = load_image(str(fname))
    assert image.shape[0] == parent.height()
    assert image.shape[1] == parent.width()


def test_show_manager_add_animation_registers_update_callback(timeline):
    show_m = ShowManager(window_type="offscreen")

    show_m.add_animation(timeline, update_rate=0.25)

    assert show_m._animations == [timeline]
    assert timeline._scene is show_m.screens[0].scene
    assert timeline.playing is True
    assert f"animation_{id(timeline)}" in show_m._callbacks

    show_m.add_animation(timeline, update_rate=0.5)

    assert show_m._animations == [timeline]
    assert len(show_m._callbacks) == 1


def test_show_manager_add_animation_rejects_invalid_object():
    show_m = ShowManager(window_type="offscreen")

    with pytest.raises(TypeError, match="Expected an Animation or Timeline object"):
        show_m.add_animation(object())


def test_show_manager_remove_animation_unregisters_and_removes_from_scene():
    show_m = ShowManager(window_type="offscreen")
    animation = Animation()

    show_m.add_animation(animation)
    show_m.remove_animation(animation)

    assert show_m._animations == []
    assert f"animation_{id(animation)}" not in show_m._callbacks
    assert animation._added_to_scene is False

    show_m.remove_animation(animation)
    assert show_m._animations == []


def test_show_manager_update_animation_updates_and_renders(timeline):
    animation = Animation()
    updates = []
    show_m = ShowManager(window_type="offscreen")
    animation.add_update_callback(lambda time: updates.append(time))

    timeline.add_animation(animation)
    timeline.play()

    show_m._update_animation(timeline)
    show_m._update_animation(animation)

    assert len(updates) == 2


def test_show_manager_setup_camera_animations_recurses():
    camera = object()
    parent = Animation()
    child = Animation()
    camera_animation = CameraAnimation()
    nested_camera_animation = CameraAnimation()
    child.add_child_animation(nested_camera_animation)
    parent.add_child_animation([camera_animation, child])
    show_m = ShowManager(window_type="offscreen")

    show_m._setup_camera_animations(parent, camera)

    assert camera_animation.camera is camera
    assert nested_camera_animation.camera is camera
