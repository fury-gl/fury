"""
==================
ImageContainer2D
==================
"""

##############################################################################
# Imports

from fury.ui import ImageContainer2D
from fury.window import Scene, ShowManager
from fury.data import fetch_viz_icons, read_viz_icons
from fury.io import load_image

##############################################################################
# Fetch icons

fetch_viz_icons()

##############################################################################
# Load image properly

img_path = read_viz_icons(fname="circle-right.png")
img_rgb = load_image(img_path)

# Convert to grayscale manually (for demo comparison)
img_gray = (
    0.2989 * img_rgb[..., 0]
    + 0.5870 * img_rgb[..., 1]
    + 0.1140 * img_rgb[..., 2]
).astype("uint8")

##############################################################################
# Scene

scene = Scene()

##############################################################################
# UI elements

image_ui_gray = ImageContainer2D(
    img_gray,
    position=(50, 250),
    size=(200, 200),
)

image_ui_rgb = ImageContainer2D(
    img_rgb,
    position=(300, 250),
    size=(200, 200),
)

scene.add(image_ui_gray)
scene.add(image_ui_rgb)

##############################################################################
# Run

if __name__ == "__main__":
    show_manager = ShowManager(
        scene=scene,
        size=(700, 700),
        title="FURY 2.0: ImageContainer2D Example",
    )

    show_manager.start()