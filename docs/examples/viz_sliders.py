"""
==========
UI Sliders
==========
"""

from fury.ui import LineSlider2D, RingSlider2D
from fury.window import Scene, ShowManager

scene = Scene()

# Horizontal Line Slider
line_slider_h = LineSlider2D(
    position=(50, 200),
    initial_value=70,
    min_value=0,
    max_value=100,
    length=300,
    orientation="horizontal",
    shape="disk",
    text_template="Horizontal: {value:.1f}",
)

# Vertical Line Slider
line_slider_v = LineSlider2D(
    position=(500, 100),
    initial_value=30,
    min_value=0,
    max_value=100,
    length=300,
    orientation="vertical",
    shape="square",
    text_template="Vertical: {value:.1f}",
)

# Ring Slider
ring_slider = RingSlider2D(
    center=(200, 400),
    initial_value=180,
    min_value=0,
    max_value=360,
    slider_inner_radius=50,
    slider_outer_radius=60,
    handle_outer_radius=12,
    text_template="Angle: {angle:.0f}°",
)

scene.add(line_slider_h)
scene.add(line_slider_v)
scene.add(ring_slider)

current_size = (800, 700)
show_manager = ShowManager(
    scene=scene,
    size=current_size,
    title="FURY Sliders Example",
)

show_manager.start()
