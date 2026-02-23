"""Tests for RingSlider2D."""

import numpy as np
import numpy.testing as npt

from fury import ui
from fury.ui.helpers import TWO_PI


def test_ring_slider_2d_initialization_default():
    rs = ui.RingSlider2D()
    npt.assert_equal(rs.value, 180)
    npt.assert_equal(rs.min_value, 0)
    npt.assert_equal(rs.max_value, 360)
    npt.assert_almost_equal(rs.ratio, 0.5)
    npt.assert_almost_equal(rs.angle, np.pi)


def test_ring_slider_2d_initialization_custom():
    rs = ui.RingSlider2D(
        center=(100, 100),
        initial_value=90,
        min_value=0,
        max_value=360,
        slider_inner_radius=50,
        slider_outer_radius=60,
    )
    npt.assert_equal(rs.value, 90)
    npt.assert_almost_equal(rs.ratio, 0.25)
    npt.assert_almost_equal(rs.angle, np.pi / 2)
    npt.assert_equal(rs.track.inner_radius, 50)
    npt.assert_equal(rs.track.outer_radius, 60)


def test_ring_slider_2d_value_property():
    rs = ui.RingSlider2D(min_value=0, max_value=100, initial_value=50)
    rs.value = 75
    npt.assert_almost_equal(rs.ratio, 0.75)
    npt.assert_almost_equal(rs.angle, 0.75 * TWO_PI)


def test_ring_slider_2d_ratio_property():
    rs = ui.RingSlider2D(min_value=0, max_value=100, initial_value=50)
    rs.ratio = 0.25
    npt.assert_equal(rs.value, 25)
    npt.assert_almost_equal(rs.angle, 0.25 * TWO_PI)


def test_ring_slider_2d_angle_property():
    rs = ui.RingSlider2D(min_value=0, max_value=100, initial_value=50)
    rs.angle = np.pi  # 180 degrees
    npt.assert_almost_equal(rs.ratio, 0.5)
    npt.assert_almost_equal(rs.value, 50)


def test_ring_slider_2d_move_handle():
    rs = ui.RingSlider2D(center=(0, 0), min_value=0, max_value=360)
    # Click at (10, 0) - should be 0 degrees
    rs.move_handle((10, 0))
    npt.assert_almost_equal(rs.angle, 0)

    # Click at (0, 10) - should be 90 degrees (pi/2)
    rs.move_handle((0, 10))
    npt.assert_almost_equal(rs.angle, np.pi / 2)


def test_ring_slider_2d_hooks():
    rs = ui.RingSlider2D()
    res = {"changed": 0, "value_changed": 0, "moving": 0}

    rs.on_change = lambda slider: res.update({"changed": res["changed"] + 1})
    rs.on_value_changed = lambda slider: res.update(
        {"value_changed": res["value_changed"] + 1}
    )
    rs.on_moving_slider = lambda slider: res.update({"moving": res["moving"] + 1})

    # Setting value should trigger on_change and on_value_changed
    rs.value = 100
    npt.assert_equal(res["value_changed"], 1)
    npt.assert_equal(res["changed"], 1)
    npt.assert_equal(res["moving"], 0)


def test_ring_slider_2d_min_max_setters():
    rs = ui.RingSlider2D(min_value=0, max_value=100, initial_value=50)
    rs.min_value = 25
    npt.assert_equal(rs.min_value, 25)
    # Value should remain 50, but ratio should update to (50-25)/(100-25) = 25/75 = 1/3
    npt.assert_almost_equal(rs.ratio, 1 / 3)

    rs.max_value = 200
    npt.assert_equal(rs.max_value, 200)
    # Value 50, min 25, max 200 -> ratio = (50-25)/(200-25) = 25/175 = 1/7
    npt.assert_almost_equal(rs.ratio, 1 / 7)


def test_ring_slider_2d_text_formatting():
    # Test string template
    rs = ui.RingSlider2D(text_template="Val: {value:.1f}")
    rs.value = 45.67
    npt.assert_equal(rs.text.message, "Val: 45.7")

    # Test callable template
    def my_template(slider):
        return f"Custom {slider.value:.0f}"

    rs = ui.RingSlider2D(text_template=my_template)
    rs.value = 10
    npt.assert_equal(rs.text.message, "Custom 10")


def test_ring_slider_2d_visibility():
    rs = ui.RingSlider2D()
    rs.set_visibility(False)
    for actor in rs.actors:
        npt.assert_equal(actor.visible, False)

    rs.set_visibility(True)
    for actor in rs.actors:
        npt.assert_equal(actor.visible, True)


def test_ring_slider_2d_mid_track_radius():
    rs = ui.RingSlider2D(slider_inner_radius=40, slider_outer_radius=60)
    npt.assert_equal(rs.mid_track_radius, 50)


def test_ring_slider_2d_interaction_callbacks():
    rs = ui.RingSlider2D()
    res = {"moving": 0}
    rs.on_moving_slider = lambda slider: res.update({"moving": res["moving"] + 1})

    # Mock an event
    class MockEvent:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def stop_propagation(self):
            pass

    # Simulate track click
    rs.track_click_callback(MockEvent(300, 300))
    npt.assert_equal(res["moving"], 1)

    # Simulate handle move
    rs.handle_move_callback(MockEvent(310, 310))
    npt.assert_equal(res["moving"], 2)
    npt.assert_equal(rs.handle.color, rs.active_color)

    # Simulate handle release
    rs.handle_release_callback(MockEvent(310, 310))
    npt.assert_equal(rs.handle.color, rs.default_color)
