"""
============
Modal Loader
============

Show an animated loader while preparing a display, then restore interaction.
The loader dims the entire canvas and blocks pointer and keyboard input until
``hide_loader()`` is called. Click **Load again** to repeat the demonstration.

Scheduled callbacks simulate loading stages without blocking the GUI event loop.
Showing a loader does not make synchronous work asynchronous: do not put slow I/O,
long computations, or ``time.sleep()`` in these callbacks. Real application work
must yield to the event loop or deliver its results back to the GUI thread.
"""

import numpy as np

from fury.actor import sphere
from fury.ui import TextBlock2D, TextButton2D
from fury.window import Scene, ShowManager

###############################################################################
# Use generated geometry, so the example needs no downloaded assets.

scene = Scene(background=(0.12, 0.16, 0.22))
spheres = sphere(
    np.array([[-2, 0, 0], [0, 0, 0], [2, 0, 0]]),
    radii=0.65,
    colors=np.array([[0.2, 0.7, 1.0], [1.0, 0.4, 0.3], [1.0, 0.8, 0.2]]),
    material="basic",
    impostor=False,
)
scene.add(spheres)

status = TextBlock2D(
    text="Ready",
    position=(24, 24),
    size=(592, 30),
    font_size=18,
)
load_button = TextButton2D(
    label="Load again", position=(24, 72), size=(140, 44), font_size=18
)
scene.add(status, load_button)
show_m = ShowManager(scene=scene, size=(800, 800), title="FURY Loader Example")

###############################################################################
# Repeated show_loader calls change the message without restarting rotation.
# Calling it without a message leaves only the spinner and dimming overlay.


def prepare_display():
    """Update the visible loader's message while animation continues."""
    show_m.show_loader(message="Preparing display…")


def clear_message():
    """Keep the spinner visible without any message text."""
    show_m.show_loader()


def finish_loading():
    """Reveal the generated geometry and restore canvas interaction."""
    spheres.visible = True
    status.message = "Ready — drag to rotate, or click Load again."
    show_m.hide_loader()


def start_loading(_button=None):
    """Show the loader on startup or from the button's click callback."""
    spheres.visible = False
    status.message = "Preparing generated geometry"
    show_m.show_loader(message="Reading data…\nPlease wait")
    show_m.register_callback(
        prepare_display, time=1.5, repeat=False, name="loader_example_prepare"
    )
    show_m.register_callback(
        clear_message, time=2.5, repeat=False, name="loader_example_clear"
    )
    show_m.register_callback(
        finish_loading, time=3.5, repeat=False, name="loader_example_finish"
    )


###############################################################################
# show_loader requests a frame; start runs the event loop that animates it.
# Closing the window also stops the manager-owned loader timer.

load_button.on_clicked = start_loading
try:
    start_loading()
    show_m.start()
finally:
    show_m.close()
