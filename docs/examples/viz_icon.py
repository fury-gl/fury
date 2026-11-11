"""
=======
Icon2D
=======

This example demonstrates how to use the ``Icon2D`` UI element to display any
icon of the Lucide pack by name. The pack is downloaded the first time an icon
is requested. Browse the available names at https://lucide.dev/icons.

First, let's import the necessary modules.
"""

from fury.ui import Icon2D
from fury.window import Scene, ShowManager

###############################################################################
# Create a Scene. The icons are dark grey by default, so we pick a light
# background.

scene = Scene(background=(1, 1, 1, 1))

###############################################################################
# Create one ``Icon2D`` per icon name and lay them out in a grid. The ``color``
# argument accepts any form understood by ``normalize_colors``: a hex string,
# or RGB values in [0, 1] or [0, 255]. Each row below uses a different form,
# and the first row keeps the default grey. Add ``-filled`` to any name to get
# the solid version of an icon, as in the first row. It suits closed shapes like
# a heart or a star, while icons with inner details lose them when filled.

names = [
    "house-filled",
    "heart-filled",
    "star-filled",
    "bookmark-filled",
    "folder",
    "database",
    "download",
    "save",
    "pencil",
    "trash",
    "play",
    "pause",
    "arrow-up",
    "arrow-down",
    "arrow-left",
    "arrow-right",
]
row_colors = [None, "#E61A1A", (26, 153, 51), (0.1, 0.4, 0.9)]

columns = 4
size = (64, 64)
spacing = 100

icons = []
for i, name in enumerate(names):
    row, column = divmod(i, columns)
    options = {} if row_colors[row] is None else {"color": row_colors[row]}
    icons.append(
        Icon2D(
            name,
            position=(50 + column * spacing, 50 + row * spacing),
            size=size,
            **options,
        )
    )

scene.add(*icons)

###############################################################################
# The displayed icon and its color can be changed later by assigning a new
# ``name`` or ``color``.

icons[0].name = "star-filled"
icons[0].color = (0.8, 0.5, 0.0)

###############################################################################
# Create and start the ShowManager.

show_manager = ShowManager(
    scene=scene,
    size=(500, 500),
    title="FURY Icon2D Example",
)
show_manager.start()
