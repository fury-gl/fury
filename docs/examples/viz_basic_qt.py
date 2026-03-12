"""
==================
Using FURY with Qt
==================

This example demonstrates how to use FURY with a Qt application to create a simple GUI.
It integrates FURY's rendering capabilities with Qt's event handling and widget system.
"""

import numpy as np

from fury.window import show
from fury.actor import box

###############################################################################
# Let's create sphere actor to add three spheres to display.

centers = np.random.rand(5, 3) * 10
box_actor = box(centers=centers)

show(actors=[box_actor], window_type="qt")
