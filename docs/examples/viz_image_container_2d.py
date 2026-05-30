"""
==================
ImageContainer2D
==================

"""

import numpy as np
from fury.ui import ImageContainer2D
from fury.window import Scene, ShowManager

from fury.data import fetch_viz_cubemaps, read_viz_cubemap

fetch_viz_cubemaps()
skybox_images = read_viz_cubemap("skybox")
texture_path = skybox_images[0]

img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    img_rgb[:, i, 0] = i
    img_rgb[i, :, 1] = (i * 2) % 256
    img_rgb[:, :, 2] = 128

img_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
for i in range(256):
    img_rgba[:, i, 0] = 255 - i
    img_rgba[i, :, 1] = (i * 2) % 256
    img_rgba[:, :, 2] = 128

    img_rgba[:, i, 3] = 255 - i


img_gray = (
    0.2989 * img_rgb[..., 0].astype(np.float32)
    + 0.5870 * img_rgb[..., 1].astype(np.float32)
    + 0.1140 * img_rgb[..., 2].astype(np.float32)
).astype(np.uint8)


scene = Scene()


gray_container = ImageContainer2D(
    img_path=img_gray,
    position=(50, 450),
    size=(256, 256),
)

rgb_container = ImageContainer2D(
    img_path=img_rgb,
    position=(350, 450),
    size=(256, 256),
)

rgba_container = ImageContainer2D(
    img_path=img_rgba,
    position=(50, 100),
    size=(256, 256),
)

skybox_container = ImageContainer2D(
    img_path=texture_path,
    position=(350, 100),
    size=(256, 256),
)

scene.add(gray_container)
scene.add(rgb_container)
scene.add(rgba_container)
scene.add(skybox_container)


show_manager = ShowManager(
    scene=scene,
    size=(700, 800),
    title="FURY ImageContainer2D Example",
)
show_manager.start()
