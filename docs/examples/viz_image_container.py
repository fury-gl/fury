"""
==================
ImageContainer2D
==================
"""

##############################################################################
# Imports

import numpy as np
from fury.ui import ImageContainer2D
from fury.window import Scene, ShowManager

##############################################################################
# Creating a Scene

scene = Scene()

##############################################################################
# Creating sample images

# Grayscale image (used as-is)
img_gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

# RGB image (will be converted internally to grayscale)
img_rgb = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

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