"""Test containers module."""

from os.path import join as pjoin
import itertools

import numpy as np
import numpy.testing as npt

from fury import window, actor, ui
from fury.colormap import distinguishable_colormap
from fury.data import DATA_DIR, read_viz_icons, fetch_viz_icons
from fury.utils import shallow_copy
from fury.testing import EventCounter


def setup_module():
    fetch_viz_icons()


def test_wrong_interactor_style():
    panel = ui.Panel2D(size=(300, 150))
    dummy_scene = window.Scene()
    _ = window.ShowManager(dummy_scene, interactor_style='trackball')
    npt.assert_raises(TypeError, panel.add_to_scene, dummy_scene)


def test_ui_tab_ui(interactive=False):
    filename = "test_ui_tab_ui"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    tab_ui = ui.TabUI(position=(50, 50), size=(300, 300), nb_tabs=3,
                      draggable=True)

    tab_ui.tabs[0].title = "Tab 1"
    tab_ui.tabs[1].title = "Tab 2"
    tab_ui.tabs[2].title = "Tab 3"

    tab_ui.add_element(0, ui.Checkbox(["Option 1", "Option 2"]), (0.5, 0.5))
    tab_ui.add_element(1, ui.LineSlider2D(), (0.0, 0.5))
    tab_ui.add_element(2, ui.TextBlock2D(), (0.5, 0.5))

    with npt.assert_raises(IndexError):
        tab_ui.add_element(3, ui.TextBlock2D(), (0.5, 0.5, 0.5))

    with npt.assert_raises(IndexError):
        tab_ui.remove_element(3, ui.TextBlock2D())

    with npt.assert_raises(IndexError):
        tab_ui.update_element(3, ui.TextBlock2D(), (0.5, 0.5, 0.5))

    npt.assert_equal("Tab 1", tab_ui.tabs[0].title)
    npt.assert_equal("Tab 2", tab_ui.tabs[1].title)
    npt.assert_equal("Tab 3", tab_ui.tabs[2].title)

    npt.assert_equal(3, tab_ui.nb_tabs)

    collapses = itertools.count()
    changes = itertools.count()

    def collapse(tab_ui):
        if tab_ui.collapsed:
            next(collapses)

    def tab_change(tab_ui):
        next(changes)

    tab_ui.on_change = tab_change
    tab_ui.on_collapse = collapse

    event_counter = EventCounter()
    event_counter.monitor(tab_ui)

    current_size = (800, 800)
    show_manager = window.ShowManager(
        size=current_size, title="Tab UI Test")
    show_manager.scene.add(tab_ui)

    if interactive:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)
    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)

    npt.assert_equal(0, tab_ui.active_tab_idx)
    npt.assert_equal(11, next(changes))
    npt.assert_equal(5, next(collapses))


def test_grid_ui(interactive=False):
    vol1 = np.zeros((100, 100, 100))
    vol1[25:75, 25:75, 25:75] = 100

    colors = distinguishable_colormap(nb_colors=3)
    contour_actor1 = actor.contour_from_roi(vol1, np.eye(4),
                                            colors[0], 1.)

    vol2 = np.zeros((100, 100, 100))
    vol2[25:75, 25:75, 25:75] = 100

    contour_actor2 = actor.contour_from_roi(vol2, np.eye(4),
                                            colors[1], 1.)

    vol3 = np.zeros((100, 100, 100))
    vol3[25:75, 25:75, 25:75] = 100

    contour_actor3 = actor.contour_from_roi(vol3, np.eye(4),
                                            colors[2], 1.)

    scene = window.Scene()
    actors = []
    texts = []

    actors.append(contour_actor1)
    text_actor1 = actor.text_3d('cube 1', justification='center')
    texts.append(text_actor1)

    actors.append(contour_actor2)
    text_actor2 = actor.text_3d('cube 2', justification='center')
    texts.append(text_actor2)

    actors.append(contour_actor3)
    text_actor3 = actor.text_3d('cube 3', justification='center')
    texts.append(text_actor3)

    actors.append(shallow_copy(contour_actor1))
    text_actor1 = actor.text_3d('cube 4', justification='center')
    texts.append(text_actor1)

    actors.append(shallow_copy(contour_actor2))
    text_actor2 = actor.text_3d('cube 5', justification='center')
    texts.append(text_actor2)

    actors.append(shallow_copy(contour_actor3))
    text_actor3 = actor.text_3d('cube 6', justification='center')
    texts.append(text_actor3)

    actors.append(shallow_copy(contour_actor1))
    text_actor1 = actor.text_3d('cube 7', justification='center')
    texts.append(text_actor1)

    actors.append(shallow_copy(contour_actor2))
    text_actor2 = actor.text_3d('cube 8', justification='center')
    texts.append(text_actor2)

    actors.append(shallow_copy(contour_actor3))
    text_actor3 = actor.text_3d('cube 9', justification='center')
    texts.append(text_actor3)

    counter = itertools.count()
    show_m = window.ShowManager(scene)
    show_m.initialize()

    def timer_callback(_obj, _event):
        cnt = next(counter)
        show_m.scene.zoom(1)
        show_m.render()
        if cnt == 10:
            show_m.exit()
            show_m.destroy_timers()

    # show the grid with the captions
    grid_ui = ui.GridUI(actors=actors, captions=texts,
                        caption_offset=(0, -50, 0),
                        cell_padding=(60, 60), dim=(3, 3),
                        rotation_axis=(1, 0, 0))

    scene.add(grid_ui)

    show_m.add_timer_callback(True, 200, timer_callback)
    show_m.start()

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects > 9, True)

    # this needs to happen automatically when start() ends.
    for act in actors:
        act.RemoveAllObservers()

    filename = "test_grid_ui"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    current_size = (900, 600)
    scene = window.Scene()
    show_manager = window.ShowManager(scene,
                                      size=current_size,
                                      title="FURY GridUI")
    show_manager.initialize()

    grid_ui2 = ui.GridUI(actors=actors, captions=texts,
                         caption_offset=(0, -50, 0),
                         cell_padding=(60, 60), dim=(3, 3),
                         rotation_axis=None)

    scene.add(grid_ui2)

    event_counter = EventCounter()
    event_counter.monitor(grid_ui2)

    if interactive:
        show_manager.start()

    recording = False

    if recording:
        # Record the following events
        # 1. Left click on top left box (will rotate the box)
        show_manager.record_events_to_file(recording_filename)
        # print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_image_container_2d(interactive=False):
    image_test = ui.ImageContainer2D(
        img_path=read_viz_icons(fname='home3.png'))

    image_test.center = (300, 300)
    npt.assert_equal(image_test.size, (100, 100))

    image_test.scale((2, 2))
    npt.assert_equal(image_test.size, (200, 200))

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title="FURY Button")
    show_manager.scene.add(image_test)
    if interactive:
        show_manager.start()
