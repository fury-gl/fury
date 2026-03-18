"""
=========
TextBox2D
=========

This example shows how to use TextBox2D.
"""

from fury.ui import TextBox2D
from fury.window import Scene, ShowManager

scene = Scene()

textbox = TextBox2D(
    width=20,
    height=5,
    text="1",
    position=(100, 100),
    color=(0, 0, 0),
    font_size=50,
)


def on_off_focus(tb):
    print("TextBox2D:", tb.message)


textbox.off_focus = on_off_focus
scene.add(textbox)

if __name__ == "__main__":
    show_manager = ShowManager(scene=scene, size=(800, 600), title="FURY TextBox2D Example")
    show_manager.start()

