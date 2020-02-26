# -*- coding: utf-8 -*-
"""
=====================
Cube & Slider Control
=====================

This example shows how to use the UI API. We will demonstrate how to
create a cube and control with sliders.

First, some imports.
"""
from fury.data import read_viz_icons, fetch_viz_icons

from fury import ui, window

###############################################################################
# Cube and sliders
# ================
#
# Add a cube to the scene .


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


cube = cube_maker(color=(0, 0, 1), size=(20, 20, 20), center=(15, 0, 0))

###############################################################################
# Now we'll add two sliders: one circular and one linear.

ring_slider = ui.RingSlider2D(center=(630, 400), initial_value=0,
                              text_template="{angle:5.1f}°")

line_slider = ui.LineSlider2D(center=(400, 230), initial_value=0,
                              min_value=-10, max_value=10)

###############################################################################
# We can use a callback to rotate the cube with the ring slider.


def rotate_cube(slider):
    angle = slider.value
    previous_angle = slider.previous_value
    rotation_angle = angle - previous_angle
    cube.RotateX(rotation_angle)


ring_slider.on_change = rotate_cube

###############################################################################
# Similarly, we can translate the cube with the line slider.


def translate_cube(slider):
    value = slider.value
    cube.SetPosition(value, 0, 0)


line_slider.on_change = translate_cube

###############################################################################
# Show Manager
# ==================================
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size, title="DIPY Cube Example")

show_manager.scene.add(cube)
show_manager.scene.add(ring_slider)
show_manager.scene.add(line_slider)

###############################################################################
# Visibility by default is True

cube.SetVisibility(True)
ring_slider.set_visibility(True)
line_slider.set_visibility(True)

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
              size=current_size, out_path="viz_slider.png")
