"""
========
Button2D
========

This example shows the two button types of FURY and how they react to the
pointer. Both are driven by a ``states`` mapping whose keys are ``"default"``,
``"hover"``, ``"pressed"`` and ``"disabled"``:

* ``TextButton2D`` changes its background color and, optionally, its text.
* ``TexturedButton2D`` swaps the image it displays. Here we use icons of the
  Lucide pack.

Buttons can also behave as toggles (``is_toggle=True``), be disabled through
their ``enabled`` property, and call a function when clicked (``on_clicked``).
"""

##############################################################################
# First, a bunch of imports.

from fury.data import read_viz_icons
from fury.ui import TextBlock2D, TexturedButton2D, TextButton2D
from fury.window import Scene, ShowManager

##############################################################################
# Create a Scene with a light background, since the Lucide icons are dark grey.

scene = Scene(background=(0.85, 0.87, 0.9, 1))

##############################################################################
# A ``TextButton2D`` takes one entry per state. An entry is either an RGB color,
# or a dictionary with the ``"color"`` of the background and the ``"text"`` to
# show in that state.

text_button = TextButton2D(
    label="Hello",
    size=(140, 60),
    position=(60, 100),
    font_size=22,
    states={
        "default": {"text": "Hello", "color": (1, 1, 1)},
        "hover": {"text": "Hover", "color": (0.9, 0.95, 1)},
        "pressed": {"text": "Pressed", "color": (0.6, 0.75, 1)},
        "disabled": {"text": "Disabled", "color": (0.6, 0.6, 0.6)},
    },
)

##############################################################################
# ``on_clicked`` is called with the button every time it is clicked. We use it
# to count the clicks and show the result in a ``TextBlock2D``.

clicks = 0
click_counter = TextBlock2D(
    text="Clicks: 0",
    font_size=20,
    color=(0, 0, 0),
    position=(60, 180),
    dynamic_bbox=True,
)


def count_click(button):
    """
    Show how many times the button has been clicked.

    Parameters
    ----------
    button : TextButton2D
        The clicked button.
    """
    global clicks
    clicks += 1
    click_counter.message = f"Clicks: {clicks}"


text_button.on_clicked = count_click

##############################################################################
# A toggle button stays in its ``"pressed"`` state until it is clicked again.
# The ``toggled`` property tells which state it is in.

toggle_button = TextButton2D(
    label="Toggle",
    size=(140, 60),
    position=(60, 250),
    font_size=22,
    is_toggle=True,
    states={
        "default": {"text": "Off", "color": (1, 1, 1)},
        "hover": {"text": "Off", "color": (0.9, 0.95, 1)},
        "pressed": {"text": "On", "color": (0.5, 0.85, 0.55)},
    },
)

##############################################################################
# Setting ``enabled`` to False ignores the pointer and shows the
# ``"disabled"`` state.

disabled_button = TextButton2D(
    label="Disabled",
    size=(140, 60),
    position=(60, 340),
    font_size=22,
    states={
        "default": (1, 1, 1),
        "disabled": (0.6, 0.6, 0.6),
    },
)
disabled_button.enabled = False

##############################################################################
# A ``TexturedButton2D`` takes one image path per state. The Lucide pack stores
# every icon three times: ``<name>.png``, a lighter ``<name>-pressed.png`` and a
# solid ``<name>-filled.png``. This play/pause toggle shows ``play`` until it is
# clicked, then ``pause``.

play_pause_button = TexturedButton2D(
    states={
        "default": read_viz_icons(fname="play.png"),
        "pressed": read_viz_icons(fname="pause.png"),
    },
    size=(60, 60),
    position=(300, 100),
    is_toggle=True,
)

##############################################################################
# Each interaction can have its own image. This button changes icon when the
# pointer hovers it and uses the lighter variant of the icon while it is held.

arrow_button = TexturedButton2D(
    states={
        "default": read_viz_icons(fname="circle-chevron-right.png"),
        "hover": read_viz_icons(fname="circle-chevron-up.png"),
        "pressed": read_viz_icons(fname="circle-chevron-down-pressed.png"),
    },
    size=(60, 60),
    position=(300, 200),
)

##############################################################################
# A state can also describe a Lucide icon and its color instead of an image
# path: ``{"icon": "heart", "color": "#F43F5E"}``. The color accepts a hex
# string or RGB values in [0, 1] or [0, 255].
#
# When a ``"selected"`` state is present, the button shows the selected state
# of a toggle, combined with the pointer feedback of ``"pressed"``. The icon of
# a selected state is filled automatically, so this like button turns into a
# solid rose heart once it is clicked.

like_button = TexturedButton2D(
    states={
        "default": {"icon": "heart", "color": "#6B7280"},
        "pressed": {"icon": "heart", "color": "#111827"},
        "selected": {"icon": "heart", "color": "#F43F5E"},
        "pressed-selected": {"icon": "heart", "color": "#E11D48"},
    },
    size=(60, 60),
    position=(400, 100),
    is_toggle=True,
)

##############################################################################
# A disabled ``TexturedButton2D`` uses its ``"disabled"`` image, if it has one.

locked_button = TexturedButton2D(
    states={
        "default": read_viz_icons(fname="download.png"),
        "disabled": read_viz_icons(fname="download-pressed.png"),
    },
    size=(60, 60),
    position=(300, 300),
)
locked_button.enabled = False

##############################################################################
# Add everything to the scene.

scene.add(
    text_button,
    click_counter,
    toggle_button,
    disabled_button,
    play_pause_button,
    arrow_button,
    locked_button,
    like_button,
)

##############################################################################
# Starting the ShowManager. Hover and click the buttons to see their states.

show_manager = ShowManager(
    scene=scene,
    size=(500, 450),
    title="FURY Button2D Example",
)
show_manager.start()
