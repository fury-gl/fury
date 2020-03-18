"""
======================================
Cube Color Control using Radio Buttons
======================================

This example shows how to use the UI API. We will demonstrate how to
create a cube and control its color using radio buttons.

First, some imports.
"""

from fury import ui, window

########################################################################
# Cube and Radio Buttons
# ================
#
# Add a cube to the scene.


def cube_maker(color=(1, 1, 1), size=(0.2, 0.2, 0.2), center=(0, 0, 0)):
    cube = window.vtk.vtkCubeSource()
    cube.SetXLength(size[0])
    cube.SetYLength(size[1])
    cube.SetZLength(size[2])
    
    if center is not None:
        cube.SetCenter(*center)
    
    cube_mapper = window.vtk.vtkPolyDataMapper()
    cube_mapper.SetInputConnection(cube.GetOutputPort())
    cube_actor = window.vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    
    if color is not None:
        cube_actor.GetProperty().SetColor(color)
    
    return cube_actor


# Creating a dict of possible options and mapping it with their values.
options = {'Blue': (0, 0, 1) , 'Red': (1, 0, 0), 'Green': (0, 1, 0)}

color_toggler = ui.RadioButton(list(options), checked_labels=['Blue'],
                               padding=1, font_size=16,
                               font_family='Arial', position=(200, 200))

cube = cube_maker(color=(0, 0, 1), size=(20, 20, 20), center=(15, 0, 0))

# A callback which will set the values for the box
def toggle_color(radio):
    color = options[radio.checked_labels[0]]
    cube.GetProperty().SetColor(*color)

color_toggler.on_change = toggle_color

###############################################################################
# Show Manager
# ============================================================================
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size, title="DIPY Cube Example")

show_manager.scene.add(cube)
show_manager.scene.add(color_toggler)
color_toggler.set_visibility(True)
cube.SetVisibility(True)

###############################################################################
# Set camera for better visualization

show_manager.scene.reset_camera()
show_manager.scene.set_camera(position=(0, 0, 150))
show_manager.scene.reset_clipping_range()
show_manager.scene.azimuth(30)
interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene,
              size=current_size, out_path="radio_button.png")
