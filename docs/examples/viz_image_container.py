"""
==================
ImageContainer2D
==================
"""

##############################################################################
# First, a bunch of imports

import numpy as np

from fury.ui import ImageContainer2D
from fury.window import (
    Scene,
    ShowManager,
)

##############################################################################
# Creating a Scene

scene = Scene()

##############################################################################
# Creating a sample image

img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

##############################################################################
# Creating the ImageContainer2D UI element

image_ui = ImageContainer2D(
    img,
    position=(250, 250),
    size=(200, 200),
)

scene.add(image_ui)

###############################################################################
# Starting the ShowManager

if __name__ == "__main__":
    current_size = (700, 700)

    show_manager = ShowManager(
        scene=scene,
        size=current_size,
        title="FURY 2.0: ImageContainer2D Example",
    )

    show_manager.start()
