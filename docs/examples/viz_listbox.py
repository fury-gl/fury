"""
================
ListBox2D UI
================

This example demonstrates how to use the ``ListBox2D`` UI component in FURY.
A ``ListBox2D`` displays a scrollable list of selectable text items. It
supports single and multi-selection, text clipping for long entries, and
interactive scrolling via the mouse wheel or a draggable scroll bar.

**Interactions:**

- **Scroll** the mouse wheel to navigate through the list.
- **Click** an item to select it (background turns red, text becomes bold).
- **Ctrl + Click** to select multiple items when ``multiselection=True``.
- **Drag** the red scroll bar on the right to jump through the list.
"""

from fury import ui, window

###############################################################################
# Create the scene that will hold our UI elements.

scene = window.Scene()

###############################################################################
# Build a list of values for the listbox.

values = [
    "Argentina",
    "Brazil",
    "Canada",
    "Denmark",
    "Egypt",
    "This is a very long item that should be clipped with trailing dots",
    "France",
    "Germany",
    "Hungary",
    "India",
    "Japan",
    "Kenya",
    "Another lengthy entry to verify clip_overflow works with PyGfx",
    "Lebanon",
    "Mexico",
    "Netherlands",
    "Oman",
    "Portugal",
    "Qatar",
    "Russia",
    "Spain",
    "Thailand",
    "United Kingdom",
    "United States of America",
    "Venezuela",
    "Zimbabwe",
]

###############################################################################
# Create the ``ListBox2D`` component.
# ``multiselection=True`` lets the user hold **Ctrl** and click to select
# more than one item at a time.  Selected items are highlighted in red with
# bold text.  The component is sized to show roughly 10 items at once, so
# the remaining entries are reachable via scrolling.

listbox = ui.ListBox2D(
    values=values,
    position=(100, 100),
    size=(300, 500),
    multiselection=True,
)

scene.add(listbox)

###############################################################################
# Set up the ``ShowManager`` and start the interactive render loop.

current_size = (600, 700)
show_manager = window.ShowManager(
    scene=scene,
    size=current_size,
    title="FURY ListBox2D Example",
)

show_manager.start()
