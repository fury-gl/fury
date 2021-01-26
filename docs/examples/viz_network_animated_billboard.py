"""
=======================================================
Visualize Networks using Billboard (Animated version)
=======================================================

The goal of this demo is to show how to visualize a
complex network using billboards and simple interaction
and use a force directed algorithm to layout the network.
Hovering over nodes will show node index.

"""

###############################################################################
# First, let's import some useful modules and functions

import math
import random
import time
from collections import Counter
from os.path import join as pjoin
import numpy as np
import vtk
import helios

import vtk.util.numpy_support as VN
from fury import actor, window, ui, colormap as cmap
import fury.primitive as fp
from fury.utils import (get_actor_from_polydata, numpy_to_vtk_colors,
                        set_polydata_triangles, set_polydata_vertices,
                        set_polydata_colors, colors_from_actor,
                        vertices_from_actor, update_actor)


###############################################################################
# Now, let's define some auxiliary functions

def vtk_vertices_from_actor(actor):
    """Access to vtk vertices from actor.

    Parameters
    ----------
    actor : actor

    Returns
    -------
    vertices : vtkarray

    """
    return actor.GetMapper().GetInput().GetPoints().GetData()


def vtk_array_from_actor(actor, array_name):
    """Access vtk array from actor which uses polydata.

    Parameters
    ----------
    actor : actor

    Returns
    -------
    output : vtkarray

    """
    vtk_array = \
        actor.GetMapper().GetInput().GetPointData().GetArray(array_name)
    if vtk_array is None:
        return None

    return vtk_array


###############################################################################
# Let's generate a Watts-Strogatz random network as an example

vertices_count = 10000
ws_neighs = 3
ws_rewiring = 0.01
view_size = 100

edgesList = []
for i in range(vertices_count):
    for k in range(ws_neighs):
        if(random.random() >= ws_rewiring):
            edgesList.append((i, (i+k+1) % vertices_count))
        else:
            while(True):
                randomFrom = random.randint(0, vertices_count)
                randomTo = random.randint(0, vertices_count)
                if(randomFrom != randomTo):
                    break
            edgesList.append((randomFrom, randomTo))
edges = np.ascontiguousarray(edgesList, dtype=np.uint64)


###############################################################################
# Defining visual properties, such as scale and colors of nodes

labels = None

positions = view_size * \
    np.random.random((vertices_count, 3)) - view_size / 2.0

positions = np.ascontiguousarray(positions, dtype=np.float32)
colors = np.array(cmap.cm.inferno(
    np.arange(0, vertices_count)/(vertices_count-1)))
radii = 1 + np.random.rand(len(positions))

edges_colors = []
for source, target in edges:
    edges_colors.append(np.array([colors[source], colors[target]]))

edges_colors = np.average(np.array(edges_colors), axis=1)


###############################################################################
# Our data preparation is ready, it is time to visualize them all. We start to
# build 2 actors that represent our data : nodes_actor for the nodes and
# lines_actor for the edges.

n_points = colors.shape[0]
np.random.seed(42)
centers = np.zeros(positions.shape)
radius = np.ones(n_points)

lines_actor = actor.line(np.zeros((len(edges), 2, 3)),
                         colors=edges_colors, lod=False,
                         fake_tube=False, linewidth=3,
                         opacity=0.1
                         )


###############################################################################
# We use the impostor technique to render the spheres for nodes.
# this is accomplished by instanciating a billboard actor
# and setting the shader to draw a sphere.

fs_dec = \
    """
    uniform mat4 MCDCMatrix;
    uniform mat4 MCVCMatrix;
    """

fake_sphere = \
    """
// camera right
vec3 uu = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
//  camera up
vec3 vv = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
// camera direction
vec3 ww = vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]);
vec3 ro = MCVCMatrix[3].xyz * mat3(MCVCMatrix);  // camera position

// create view ray
vec3 rd = normalize( point.x*-uu + point.y*-vv + 7*ww);
vec3 col = vec3(0.0);

float len = length(point);
float radius = 1.;
if(len > radius){
    discard;
}

vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
vec3 direction = normalize(vec3(1., 1., 1.));
float ddf = max(0, dot(direction, normalizedPoint));
float ssf = pow(ddf, 24);
fragOutput0 = vec4(max(color*0.5+ddf * color, ssf * vec3(1)), 1);
"""


nodes_actor = actor.billboard(centers,
                              colors=colors,
                              scales=1.0,
                              fs_dec=fs_dec,
                              fs_impl=fake_sphere
                              )

###############################################################################
# Preparing editable geometry for the nodes

vtk_centers_geometry = vtk_array_from_actor(nodes_actor, array_name="center")
centers_geometry = VN.vtk_to_numpy(vtk_centers_geometry)
centers_geometryOrig = np.array(centers_geometry)
centers_length = centers_geometry.shape[0] / positions.shape[0]


vtk_verts_geometry = vtk_vertices_from_actor(nodes_actor)
verts_geometry = VN.vtk_to_numpy(vtk_verts_geometry)
verts_geometryOrig = np.array(verts_geometry)
verts_length = verts_geometry.shape[0] / positions.shape[0]


###############################################################################
# Defining timer callback and layout iterator. Helios-core
# is used to dynamically layout the network

velocities = np.zeros((vertices_count, 3), dtype=np.float32)
edgesArray = np.ascontiguousarray(np.array(edges), dtype=np.uint64)
layout = None


def new_layout_timer(showm, edges_list, vertices_count,
                     max_iterations=1000, vertex_initial_positions=None):
    global layout
    counter = 0
    if(vertex_initial_positions is not None):
        positions[:] = np.array(vertex_initial_positions)
    else:
        positions[:] = view_size * \
            np.random.random((vertices_count, 3)) - view_size / 2.0

    viscosity = 0.30
    a = 0.0005
    b = 1.0

    layout = helios.FRLayout(edgesArray, positions,
                             velocities, a, b, viscosity)
    layout.start()
    framesPerSecond = []

    def _timer(_obj, _event):
        nonlocal counter, framesPerSecond
        counter += 1
        framesPerSecond.append(scene.frame_rate)

        centers_geometry[:] = np.repeat(positions, centers_length, axis=0)
        verts_geometry[:] = verts_geometryOrig + centers_geometry

        edges_positions = VN.vtk_to_numpy(
            lines_actor.GetMapper().GetInput().GetPoints().GetData())
        edges_positions[::2] = positions[edges_list[:, 0]]
        edges_positions[1::2] = positions[edges_list[:, 1]]

        lines_actor.GetMapper().GetInput().GetPoints().GetData().Modified()
        lines_actor.GetMapper().GetInput().ComputeBounds()
        vtk_verts_geometry.Modified()
        vtk_centers_geometry.Modified()
        update_actor(nodes_actor)

        if(selectedNode is not None):
            selectedActor.SetPosition(positions[selectedNode])

        nodes_actor.GetMapper().GetInput().GetPoints().GetData().Modified()
        nodes_actor.GetMapper().GetInput().ComputeBounds()
        selectedActor.GetMapper().GetInput().ComputeBounds()
        selectedActor.GetMapper().GetInput().GetPoints().GetData().Modified()
        showm.scene.ResetCameraClippingRange()
        showm.render()

        if counter >= max_iterations:
            showm.exit()
    return _timer

###############################################################################
# Defining interactions with hardware selector


scene = window.Scene()
camera = scene.camera()

selectedNode = None
selectedActor = actor.label(
    "Origin", pos=centers[0], color=(1, 1, 1), scale=(4, 4, 4),)
selectedActor.PickableOff()
selectedActor.SetCamera(scene.GetActiveCamera())
lines_actor.PickableOff()


def left_click_callback(obj, event):
    global selectedNode
    event_pos = showm.iren.GetEventPosition()
    pickingArea = 4
    res = hsel.Select()
    hsel.SetArea(event_pos[0]-pickingArea, event_pos[1]-pickingArea,
                 event_pos[0]+pickingArea, event_pos[1]+pickingArea)
    res = hsel.Select()

    numNodes = res.GetNumberOfNodes()
    if (numNodes < 1):
        selectedNode = None
    else:
        sel_node = res.GetNode(0)
        selectedNodes = set(np.floor(VN.vtk_to_numpy(
            sel_node.GetSelectionList())/2).astype(int))
        selectedNode = list(selectedNodes)[0]

    if(selectedNode is not None):
        if(labels is not None):
            selectedActor.text.SetText(labels[selectedNode])
        else:
            selectedActor.text.SetText("#%d" % selectedNode)
        selectedActor.SetPosition(positions[selectedNode])

    else:
        selectedActor.text.SetText("")
    timer_callback(None, None)

###############################################################################
# We add observers to pick the nodes and enable hardware selector


nodes_actor.AddObserver('LeftButtonPressEvent', left_click_callback, 1)

hsel = vtk.vtkHardwareSelector()
hsel.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
hsel.SetRenderer(scene)

###############################################################################
# All actors need to be added in a scene, so we build one and add our
# lines_actor and nodes_actor.

scene.add(lines_actor)
scene.add(nodes_actor)
scene.add(selectedActor)


###############################################################################
# The final step ! Visualize the result of our creation! Also, we need to move
# the camera a little bit farther from the network. you can increase the
# parameter max_iteractions of the timer callback to let the animation run for
# more time.

showm = window.ShowManager(scene, reset_camera=False, size=(
    2080, 1500), order_transparent=False, multi_samples=2,)


showm.initialize()
scene.set_camera(position=(0, 0, -750))

timer_callback = new_layout_timer(
    showm, edges, vertices_count,
    max_iterations=500,
    vertex_initial_positions=positions)


showm.iren.AddObserver("MouseMoveEvent", left_click_callback)

# Run every 16 milliseconds
showm.add_timer_callback(True, 16, timer_callback)
showm.start()

window.record(showm.scene, size=(900, 768),
              out_path="viz_animated_networks.png")

###############################################################################
# Cleaning up Helios-Core thread

if(layout):
    layout.stop()
