"""
=========================
Displaying a 16-bit image
=========================

Grayscale scientific images often store intensities as unsigned 16-bit
integers. This example loads a PNG without reducing those values to 8-bit,
then creates a floating-point copy for display with an image actor.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fury import actor, io, window

###############################################################################
# Create five intensity bands spanning the unsigned 16-bit range. A temporary
# PNG keeps the example self-contained; use an existing image path to load your
# own data instead.

bands = np.array([0, 16384, 32768, 49152, 65535], dtype=np.uint16)
data = np.repeat(np.repeat(bands[None, :], 64, axis=0), 64, axis=1)

with TemporaryDirectory() as directory:
    filename = str(Path(directory) / "image_16bit.png")
    io.save_image(data, filename)
    loaded = io.load_image(filename)

###############################################################################
# Keep the original intensities for analysis. Normalize a separate float32 copy
# to [0, 1] for display, and set the contrast limits to the same range. This
# avoids the image actor's default scaling for values above 1, which divides
# by 255.

display_data = loaded.astype(np.float32) / np.iinfo(np.uint16).max
image_actor = actor.image(image=display_data, clim=(0, 1))

scene = window.Scene(background=(0.1, 0.15, 0.2, 1))
scene.add(image_actor)

###############################################################################
# Display the five grayscale levels using the image shader.

show_m = window.ShowManager(scene=scene, title="FURY 16-bit image", size=(640, 240))
window.update_camera(show_m.screens[0].camera, None, scene)
show_m.start()
