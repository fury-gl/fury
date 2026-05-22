"""
=====
TabUI
=====

This example shows how to use :class:`fury.ui.TabUI` to organize multiple
2D UI panels inside the same window.

Left click on a tab header to show its content. Left click on the active tab
again to hide that tab's content. Right click on a tab header to collapse the
tab UI. The whole widget can be dragged from the tab header or panel
background.
"""

##############################################################################
# First, a bunch of imports.

from fury.ui import Disk2D, LineSlider2D, Rectangle2D, TabUI, TextBlock2D
from fury.window import Scene, ShowManager


##############################################################################
# Creating a Scene.

scene = Scene()


##############################################################################
# Create a TabUI with three tabs. The first tab is visible on startup.

tab_ui = TabUI(
    position=(80, 80),
    size=(420, 320),
    nb_tabs=3,
    startup_tab_id=0,
    draggable=True,
    active_color=(0.95, 0.95, 1.0),
    inactive_color=(0.45, 0.48, 0.56),
)

tab_ui.tabs[0].title = "Shapes"
tab_ui.tabs[1].title = "Slider"
tab_ui.tabs[2].title = "Notes"

for tab in tab_ui.tabs:
    tab.title_font_size = 16


##############################################################################
# Add content to the first tab.

rect = Rectangle2D(size=(90, 70), color=(0.9, 0.25, 0.35))
disk = Disk2D(outer_radius=45, color=(0.2, 0.75, 0.95))
caption = TextBlock2D(
    text="The first tab contains simple 2D shapes.",
    size=(340, 40),
    color=(1, 1, 1),
    font_size=16,
)

tab_ui.add_element(0, rect, (45, 75))
tab_ui.add_element(0, disk, (285, 110), anchor="center")
tab_ui.add_element(0, caption, (40, 205))


##############################################################################
# Add content to the second tab.

slider_label = TextBlock2D(
    text="A LineSlider2D can live inside a tab content panel.",
    size=(350, 40),
    color=(1, 1, 1),
    font_size=16,
)
slider = LineSlider2D(
    initial_value=35,
    min_value=0,
    max_value=100,
    length=300,
    orientation="horizontal",
    text_template="Value: {value:.0f}",
)

tab_ui.add_element(1, slider_label, (35, 65))
tab_ui.add_element(1, slider, (55, 150))


##############################################################################
# Add content to the third tab.

notes = TextBlock2D(
    text=(
        "TabUI groups several UI panels behind tab headers.\n"
        "Use TabUI.add_element(tab_index, element, coords)\n"
        "to place UI components in a specific tab."
    ),
    size=(360, 150),
    color=(1, 1, 1),
    font_size=15,
)

tab_ui.add_element(2, notes, (35, 70))


##############################################################################
# Add a status text outside the TabUI. The callbacks update this text when the
# active tab changes or the TabUI is collapsed.

status = TextBlock2D(
    text="Active tab: Shapes",
    position=(80, 430),
    size=(420, 35),
    color=(1, 1, 1),
    font_size=16,
)


def update_status(ui):
    """Update status text after selecting a tab."""
    if ui.active_tab_idx is None:
        status.message = "TabUI collapsed"
    else:
        status.message = f"Active tab: {ui.tabs[ui.active_tab_idx].title}"


def collapse_status(ui):
    """Update status text after collapsing the TabUI."""
    status.message = "TabUI collapsed"


tab_ui.on_change = update_status
tab_ui.on_collapse = collapse_status


##############################################################################
# TabUI can also use an accordion layout. In this mode each tab title spans the
# widget width. The active tab expands below its title while the other titles
# remain visible below the expanded content.

accordion_tab_ui = TabUI(
    position=(560, 80),
    size=(340, 320),
    nb_tabs=3,
    startup_tab_id=0,
    tab_bar_pos="accordion",
    draggable=True,
    active_color=(0.95, 0.95, 1.0),
    inactive_color=(0.45, 0.48, 0.56),
)

accordion_tab_ui.tabs[0].title = "Info"
accordion_tab_ui.tabs[1].title = "Color"
accordion_tab_ui.tabs[2].title = "Help"

for tab in accordion_tab_ui.tabs:
    tab.title_font_size = 15

accordion_info = TextBlock2D(
    text="The active tab expands below its title.",
    size=(280, 80),
    color=(1, 1, 1),
    font_size=15,
)
accordion_square = Rectangle2D(size=(90, 90), color=(0.2, 0.8, 0.45))
accordion_help = TextBlock2D(
    text="Other tab titles stay visible\nbelow the expanded content.",
    size=(260, 90),
    color=(1, 1, 1),
    font_size=15,
)

accordion_tab_ui.add_element(0, accordion_info, (30, 45))
accordion_tab_ui.add_element(1, accordion_square, (125, 55))
accordion_tab_ui.add_element(2, accordion_help, (40, 45))

accordion_status = TextBlock2D(
    text="Accordion active tab: Info",
    position=(560, 430),
    size=(340, 35),
    color=(1, 1, 1),
    font_size=16,
)


def update_accordion_status(ui):
    """Update status text after selecting an accordion tab."""
    if ui.active_tab_idx is None:
        accordion_status.message = "Accordion TabUI collapsed"
    else:
        title = ui.tabs[ui.active_tab_idx].title
        accordion_status.message = f"Accordion active tab: {title}"


def collapse_accordion_status(ui):
    """Update status text after collapsing the accordion TabUI."""
    accordion_status.message = "Accordion TabUI collapsed"


accordion_tab_ui.on_change = update_accordion_status
accordion_tab_ui.on_collapse = collapse_accordion_status


##############################################################################
# The tab bar can also be placed at the bottom of the widget.

bottom_tab_ui = TabUI(
    position=(80, 500),
    size=(420, 300),
    nb_tabs=2,
    startup_tab_id=0,
    tab_bar_pos="bottom",
    draggable=True,
    active_color=(0.95, 0.95, 1.0),
    inactive_color=(0.45, 0.48, 0.56),
)

bottom_tab_ui.tabs[0].title = "Bottom Tab 1"
bottom_tab_ui.tabs[1].title = "Bottom Tab 2"

for tab in bottom_tab_ui.tabs:
    tab.title_font_size = 15

bottom_text = TextBlock2D(
    text="This TabUI uses tab_bar_pos='bottom'.",
    size=(330, 45),
    color=(1, 1, 1),
    font_size=15,
)
bottom_rect = Rectangle2D(size=(120, 80), color=(0.9, 0.65, 0.2))

bottom_tab_ui.add_element(0, bottom_text, (40, 55))
bottom_tab_ui.add_element(1, bottom_rect, (145, 65))

bottom_status = TextBlock2D(
    text="Bottom active tab: Bottom Tab 1",
    position=(80, 820),
    size=(420, 35),
    color=(1, 1, 1),
    font_size=16,
)


def update_bottom_status(ui):
    """Update status text after selecting a bottom tab."""
    if ui.active_tab_idx is None:
        bottom_status.message = "Bottom TabUI collapsed"
    else:
        title = ui.tabs[ui.active_tab_idx].title
        bottom_status.message = f"Bottom active tab: {title}"


def collapse_bottom_status(ui):
    """Update status text after collapsing the bottom TabUI."""
    bottom_status.message = "Bottom TabUI collapsed"


bottom_tab_ui.on_change = update_bottom_status
bottom_tab_ui.on_collapse = collapse_bottom_status


##############################################################################
# Now that all elements have been initialized, add them to the scene.

scene.add(tab_ui)
scene.add(status)
scene.add(accordion_tab_ui)
scene.add(accordion_status)
scene.add(bottom_tab_ui)
scene.add(bottom_status)


##############################################################################
# Starting the ShowManager.

current_size = (980, 900)
show_manager = ShowManager(
    scene=scene,
    size=current_size,
    title="FURY TabUI Example",
)
show_manager.start()
