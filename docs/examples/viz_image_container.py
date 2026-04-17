"""
==================
ImageContainer2D
==================
"""

##############################################################################
# Imports

import numpy as np
from PIL import Image
from fury.ui import ImageContainer2D
from fury.window import Scene, ShowManager

##############################################################################
# Use a controlled image instead of random textures

img_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

for i in range(512):
    img_rgb[:, i, 0] = i % 256
    img_rgb[i, :, 1] = (i * 2) % 256

##############################################################################
# Ensure contiguous memory (VTK requirement)

img_rgb = np.ascontiguousarray(img_rgb)

img_rgb = np.flipud(img_rgb)

# # Convert to grayscale manually
img_gray = (
    0.2989 * img_rgb[..., 0]
    + 0.5870 * img_rgb[..., 1]
    + 0.1140 * img_rgb[..., 2]
).astype(np.uint8)

img_gray_rgb = np.stack([img_gray] * 3, axis=-1)

img_gray_rgb = np.ascontiguousarray(img_gray_rgb)
img_gray_rgb = np.flipud(img_gray_rgb)

##############################################################################
# Scene

scene = Scene()

##############################################################################
# Fixed UI size (no distortion)
size = (300, 300)

# UI elements
image_ui_gray = ImageContainer2D(
    img_gray_rgb,
    position=(50, 200),
    size=size,
)

image_ui_rgb = ImageContainer2D(
    img_rgb,
    position=(400, 200),
    size=size,
)

scene.add(image_ui_gray)
scene.add(image_ui_rgb)

##############################################################################
# RUN\

if __name__ == "__main__":
    show_manager = ShowManager(
        scene=scene,
        size=(800, 600),
        title="FURY 2.0: ImageContainer2D Example",
    )

    show_manager.start()
    