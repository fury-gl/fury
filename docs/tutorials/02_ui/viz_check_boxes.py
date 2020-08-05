"""
============================================================
Figure and Color Control using Check boxes and Radio Buttons
============================================================

This example shows how to use the CheckBox UI API. We will demonstrate how to
create a cube, sphere, cone and arrow and control its color and visibility
using checkboxes.

First, some imports.
"""

from fury import actor, ui, window
import numpy as np

###############################################################################
# We create the corresponding object actors for cube, sphere, cone and arrow.

cube = actor.box(centers=np.array([[15, 0, 0]]),
                 colors=np.array([[0, 0, 255]]),
                 scale=np.array([[20, 20, 20]]),
                 directions=np.array([[0, 0, 1]]))

sphere = actor.sphere(centers=np.array([[50, 0, 0]]),
                      colors=np.array([[0, 0, 1]]),
                      radii=11.0, theta=360, phi=360)

cone = actor.cone(centers=np.array([[-20, -0.5, 0]]),
                  directions=np.array([[0, 1, 0]]),
                  colors=np.array([[0, 0, 1]]),
                  heights=20, resolution=100)

arrow = actor.arrow(centers=np.array([[0, 25, 0]]),
                    colors=np.array([[0, 0, 1]]),
                    directions=np.array([[1, 0, 0]]),
                    heights=40, resolution=100)

###############################################################################
# We perform symmetric difference to determine which objects to be rendered.
# We also define a couple of methods to render visibility and color.

# Get difference between two lists.
def sym_diff(l1, l2):
    return list(set(l1).symmetric_difference(set(l2)))

# Set Visiblity of the figures
def set_figure_visiblity(checkboxes):
    checked = checkboxes.checked_labels
    unchecked = sym_diff(list(figure_dict), checked)

    for visible in checked:
        figure_dict[visible].SetVisibility(True)

    for invisible in unchecked:
        figure_dict[invisible].SetVisibility(False)


# Toggle colors of the figures
def toggle_color(radio):
    color = options[radio.checked_labels[0]]
    for _, figure in figure_dict.items():
        figure.GetProperty().SetColor(*color)

figure_dict = {'cube': cube, 'sphere': sphere, 'cone': cone, 'arrow': arrow}
check_box = ui.Checkbox(list(figure_dict), list(figure_dict),
                        padding=1, font_size=18, font_family='Arial',
                        position=(400, 85))

options = {'Blue': (0, 0, 1), 'Red': (1, 0, 0), 'Green': (0, 1, 0)}
color_toggler = ui.RadioButton(list(options), checked_labels=['Blue'],
                               padding=1, font_size=16,
                               font_family='Arial', position=(600, 120))


check_box.on_change = set_figure_visiblity
color_toggler.on_change = toggle_color


###############################################################################
# Show Manager
# ==================================
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Checkbox Example")

show_manager.scene.add(cube)
show_manager.scene.add(sphere)
show_manager.scene.add(cone)
show_manager.scene.add(arrow)
show_manager.scene.add(check_box)
show_manager.scene.add(color_toggler)

cube.SetVisibility(True)
sphere.SetVisibility(True)
cone.SetVisibility(True)
arrow.SetVisibility(True)
check_box.set_visibility(True)
color_toggler.set_visibility(True)

###############################################################################
# Set camera for better visualization

show_manager.scene.reset_camera()
show_manager.scene.set_camera(position=(0, 0, 150))
show_manager.scene.reset_clipping_range()
show_manager.scene.azimuth(30)
interactive = True

if interactive:
    show_manager.start()

window.record(show_manager.scene,
              size=current_size, out_path="viz_slider.png")
