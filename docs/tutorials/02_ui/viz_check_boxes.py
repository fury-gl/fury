"""
============================================================
Figure and Color Control using Check boxes and Radio Buttons
============================================================

This example shows how to use the UI API. We will demonstrate how to
create a cube and control its color using radio buttons.

First, some imports.
"""

from fury import actor, ui, window
import numpy as np

########################################################################
# Add a sphere to the scene.
# =========================


def sphere_maker(color=(1, 1, 1), radius=5.0, center=(0, 0, 0),
                 theta_resolution=360, phi_resolution=360):
    sphere = window.vtk.vtkSphereSource()
    sphere.SetCenter(*center)
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(theta_resolution)
    sphere.SetPhiResolution(phi_resolution)

    sphere_mapper = window.vtk.vtkPolyDataMapper()
    if window.vtk.VTK_MAJOR_VERSION <= 5:
        sphere_mapper.SetInput(sphere.GetOutput())
    else:
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

    sphere_actor = window.vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    if color is not None:
        sphere_actor.GetProperty().SetColor(color)

    return sphere_actor


########################################################################
# Add a cone to the scene.
# =======================


def cone_maker(color=(1, 1, 1), radius=5.0, center=(0, 0, 0),
               height=15, resolution=100, direction=(0, 5, 0)):
    cone = window.vtk.vtkConeSource()
    cone.SetResolution(resolution)
    cone.SetCenter(*center)
    cone.SetRadius(radius)
    cone.SetHeight(height)

    cone.SetDirection(*direction)
    cone_mapper = window.vtk.vtkPolyDataMapper()
    if window.vtk.VTK_MAJOR_VERSION <= 5:
        cone_mapper.SetInput(cone.GetOutput())
    else:
        cone_mapper.SetInputConnection(cone.GetOutputPort())

    cone_actor = window.vtk.vtkActor()
    cone_actor.SetMapper(cone_mapper)

    if color is not None:
        cone_actor.GetProperty().SetColor(color)
    return cone_actor


########################################################################
# Add an arrow to the scene.
# =========================


def arrow_maker(color=(1, 1, 1), start_point=(0, 25, 0),
                end_point=(40, 25, 0), shaft_resolution=50,
                tip_resolution=50):

    # Create an arrow.
    arrow = window.vtk.vtkArrowSource()
    arrow.SetShaftResolution(shaft_resolution)
    arrow.SetTipResolution(tip_resolution)

    # Compute a basis
    normalizedX = [0 for i in range(3)]
    normalizedY = [0 for i in range(3)]
    normalizedZ = [0 for i in range(3)]

    # The X axis is a vector from start to end
    math = window.vtk.vtkMath()
    math.Subtract(end_point, start_point, normalizedX)
    length = math.Norm(normalizedX)
    math.Normalize(normalizedX)

    # The Z axis is an arbitrary vector cross X
    arbitrary = [60, 10, 0]
    math.Cross(normalizedX, arbitrary, normalizedZ)
    math.Normalize(normalizedZ)

    # The Y axis is Z cross X
    math.Cross(normalizedZ, normalizedX, normalizedY)
    matrix = window.vtk.vtkMatrix4x4()

    # Create the direction cosine matrix
    matrix.Identity()
    for i in range(3):
        matrix.SetElement(i, 0, normalizedX[i])
        matrix.SetElement(i, 1, normalizedY[i])
        matrix.SetElement(i, 2, normalizedZ[i])

    # Apply the transforms
    transform = window.vtk.vtkTransform()
    transform.Translate(start_point)
    transform.Concatenate(matrix)
    transform.Scale(length, length, length)

    # Transform the polydata
    transformPD = window.vtk.vtkTransformPolyDataFilter()
    transformPD.SetTransform(transform)
    transformPD.SetInputConnection(arrow.GetOutputPort())

    # Create a mapper and actor for the arrow
    arrow_mapper = window.vtk.vtkPolyDataMapper()
    arrow_actor = window.vtk.vtkActor()
    arrow_actor.SetUserMatrix(transform.GetMatrix())

    if window.vtk.VTK_MAJOR_VERSION <= 5:
        arrow_mapper.SetInput(arrow.GetOutput())
    else:
        arrow_mapper.SetInputConnection(arrow.GetOutputPort())

    if color is not None:
        arrow_actor.GetProperty().SetColor(color)

    arrow_actor.SetMapper(arrow_mapper)

    return arrow_actor


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


cube = actor.box(centers=np.array([[15, 0, 0]]),
                 colors=np.array([[0, 0, 255]]),
                 scale=np.array([[20, 20, 20]]),
                 directions=np.array([[0, 0, 1]]))

sphere = sphere_maker(color=(0, 0, 1), radius=11.0, center=(50, 0, 0),
                      theta_resolution=360, phi_resolution=360)
cone = cone_maker(color=(0, 0, 1), radius=10.0, center=(-20, -0.5, 0),
                  height=20)
arrow = arrow_maker(color=(0, 0, 1), start_point=(0, 25, 0),
                    end_point=(40, 25, 0), shaft_resolution=50,
                    tip_resolution=50)


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
