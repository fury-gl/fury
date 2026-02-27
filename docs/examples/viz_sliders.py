"""
==========
UI Sliders
==========
"""

from fury.ui import LineSlider2D
from fury.window import Scene, ShowManager

scene = Scene()

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

scene.add(line_slider_h)
scene.add(line_slider_v)

if __name__ == "__main__":
    current_size = (800, 700)
    show_manager = ShowManager(
        scene=scene,
        size=current_size,
        title="FURY 2.0: Sliders Example",
    )

    show_manager.start()
