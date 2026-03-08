"""
===============
TabUI Example
===============

This example demonstrates how to create a Tab panel with the new v2 UI architecture.
"""

from fury import ui, window

scene = window.Scene()

# Create a TextBlock2D to show some text in the first tab
text_block = ui.TextBlock2D(
    text="Welcome to Tab 1",
    color=(0, 0, 0),
    bg_color=(0.9, 0.9, 0.9),
    size=(200, 30),
)

# Create a slider to include in the second tab
slider = ui.LineSlider2D(
    initial_value=50,
    min_value=0,
    max_value=100,
    length=200,
    orientation="horizontal",
)

# Create a rectangle in the third tab
rect = ui.Rectangle2D(size=(50, 50), color=(1, 0, 0))

# Initialize TabUI with 3 tabs
tab_ui = ui.TabUI(
    position=(50, 500),
    size=(300, 200),
    nb_tabs=3,
    active_color=(0.8, 0.8, 0.8),
    inactive_color=(0.4, 0.4, 0.4),
    draggable=True,
    tab_bar_pos="top",
)

# Customize the titles of each tab
tab_ui.tabs[0].title = "Intro"
tab_ui.tabs[1].title = "Settings"
tab_ui.tabs[2].title = "Graphics"

# Add elements to their corresponding tabs using normal UI offset coords
tab_ui.add_element(0, text_block, (10, 80), anchor="position")
tab_ui.add_element(1, slider, (50, 80), anchor="position")
tab_ui.add_element(2, rect, (125, 60), anchor="position")

# Add the TabUI element to the scene
scene.add(tab_ui)

showm = window.ShowManager(scene=scene, size=(600, 600))

if __name__ == "__main__":
    showm.start()
