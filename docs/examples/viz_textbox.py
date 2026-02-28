"""
============
UI Text Box
============
"""

from fury.ui import TextBox2D
from fury.window import Scene, ShowManager

scene = Scene()

textbox_single = TextBox2D(
    width=20,
    height=1,
    text="Single line",
    position=(50, 400),
    color=(1, 0, 0),
    font_size=20,
)

textbox_multi = TextBox2D(
    width=20,
    height=3,
    text="Multi-line\nText box",
    position=(50, 300),
    color=(0, 0, 1),
    font_size=25,
)

textbox_styled = TextBox2D(
    width=20,
    height=2,
    text="Bold & Italic\nCenter justification",
    position=(50, 150),
    color=(0, 0.5, 0),
    font_size=18,
    justification="center",
    bold=True,
    italic=True,
)

scene.add(textbox_single)
scene.add(textbox_multi)
scene.add(textbox_styled)

if __name__ == "__main__":
    current_size = (800, 700)
    show_manager = ShowManager(
        scene=scene,
        size=current_size,
        title="FURY 2.0: Text Box Example",
    )

    show_manager.start()
