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

##############################################################################
# Load RGB image from assets
img_rgb = read_viz_icons(fname="circle-right.png")

# Convert to grayscale manually (for demo comparison)
img_gray = img_rgb.mean(axis=2).astype("uint8")

##############################################################################
# Creating a Scene

scene = Scene()

##############################################################################
# Creating ImageContainer2D UI elements

# Grayscale → displayed directly
image_ui_gray = ImageContainer2D(
    img_gray,
    position=(50, 250),
    size=(200, 200),
)

# RGB → converted to grayscale before rendering
image_ui_rgb = ImageContainer2D(
    img_rgb,
    position=(300, 250),
    size=(200, 200),
)

scene.add(image_ui_gray)
scene.add(image_ui_rgb)

##############################################################################
# Starting the ShowManager

if __name__ == "__main__":
    show_manager = ShowManager(
        scene=scene,
        size=(700, 700),
        title="FURY 2.0: ImageContainer2D Example",
    )

    show_manager.start()
