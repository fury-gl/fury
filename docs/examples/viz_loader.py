"""
============
Modal Loader
============

This example shows how to display a loading screen while data is prepared.
:meth:`fury.window.ShowManager.show_loader` dims the whole window, draws a
spinning loader with an optional message, and ignores mouse and keyboard input.
:meth:`fury.window.ShowManager.hide_loader` removes it and gives control back
to the user.
"""

import numpy as np

from fury.actor import sphere
from fury.ui import TextBlock2D, TextButton2D
from fury.window import Scene, ShowManager

###############################################################################
# First, we create a scene with three spheres. They stand in for the data that
# your application would load.

scene = Scene(background=(0.12, 0.16, 0.22))
spheres = sphere(
    np.array([[-2, 0, 0], [0, 0, 0], [2, 0, 0]]),
    radii=0.65,
    colors=np.array([[0.2, 0.7, 1.0], [1.0, 0.4, 0.3], [1.0, 0.8, 0.2]]),
    material="basic",
    impostor=False,
)
scene.add(spheres)

###############################################################################
# Next, we add a text block that shows the current status and a button that
# starts loading again.

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
# Calling ``show_loader`` again while the loader is visible only changes the
# message. Calling it without ``message`` keeps the spinner and removes the
# text.


def update_message():
    """Change the loader message while it keeps spinning."""
    show_m.show_loader(message="Preparing display…")


def clear_message():
    """Keep the spinner and remove the message."""
    show_m.show_loader()


def finish_loading():
    """Show the spheres and hide the loader."""
    spheres.visible = True
    status.message = "Ready. Drag to rotate, or click Load again."
    show_m.hide_loader()


###############################################################################
# ``start_loading`` hides the spheres, shows the loader, and schedules the next
# steps with :meth:`fury.window.ShowManager.register_callback`. The timers only
# simulate loading. The loader spins only while the event loop runs, so real
# work must not block it, for example with ``time.sleep``.


def start_loading(_button=None):
    """Show the loader and schedule the simulated loading steps."""
    spheres.visible = False
    status.message = "Loading..."
    show_m.show_loader(message="Reading data…\nPlease wait")
    show_m.register_callback(
        update_message, time=1.5, repeat=False, name="update_message"
    )
    show_m.register_callback(
        clear_message, time=2.5, repeat=False, name="clear_message"
    )
    show_m.register_callback(
        finish_loading, time=3.5, repeat=False, name="finish_loading"
    )


###############################################################################
# Finally, we connect the button, start loading once, and run the window.

load_button.on_clicked = start_loading
start_loading()
show_m.start()
