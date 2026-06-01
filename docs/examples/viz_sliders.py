"""
==========
UI Sliders
==========
"""

from fury.ui import LineSlider2D, RingSlider2D, LineDoubleSlider2D, RangeSlider
from fury.window import Scene, ShowManager

scene = Scene()

# Horizontal Line Slider
line_slider_h = LineSlider2D(
    position=(100, 350),
    initial_value=70,
    min_value=0,
    max_value=100,
    length=400,
    orientation="horizontal",
    shape="disk",
    text_template="Horizontal: {value:.1f}",
)

# Vertical Line Slider
line_slider_v = LineSlider2D(
    position=(650, 200),
    initial_value=30,
    min_value=0,
    max_value=100,
    length=400,
    orientation="vertical",
    shape="square",
    text_template="Vertical: {value:.1f}",
)

# Ring Slider
ring_slider = RingSlider2D(
    center=(300, 550),
    initial_value=0,
    min_value=0,
    max_value=360,
    slider_inner_radius=50,
    slider_outer_radius=55,
    handle_outer_radius=12,
    text_template="Angle: {angle:.0f}°",
)

scene.add(line_slider_h)
scene.add(line_slider_v)
scene.add(ring_slider)

# Line Double Slider
line_double_slider = LineDoubleSlider2D(
    position=(100, 250),
    initial_values=(20, 80),
    min_value=0,
    max_value=100,
    length=400,
    orientation="horizontal",
    shape="square",
    text_template="Double: {value:.1f}",
)

# Range Slider
range_slider = RangeSlider(
    range_slider_center=(100, 150),
    value_slider_center=(100, 80),
    length=400,
    min_value=0,
    max_value=100,
    orientation="horizontal",
    shape="disk",
)

scene.add(line_double_slider)
scene.add(range_slider)

current_size = (800, 700)
show_manager = ShowManager(
    scene=scene,
    size=current_size,
    title="FURY Sliders Example",
)

show_manager.start()
