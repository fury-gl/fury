"""
============================================================
Nested Stochastic Block inference and Graph-tool integration
============================================================

Graph-tool it's a superb lib to perform a myriad of network inference methods.
However, graph-tool uses cairo to render the plots.
Here, I'll show to you how to use fury  to draw a nested stochastic block model
"""
from graph_tool.all import *
import graph_tool as gt
import numpy as np

from fury import colormap as cmap

from scipy import interpolate

from colorsys import rgb_to_hsv, hsv_to_rgb

import matplotlib.pyplot as plt
from fury import actor, window
###############################################################################
# First, we need to load the network data and extract the largest component


g = gt.collection.data["celegansneural"]
g = gt.GraphView(g, vfilt=gt.topology.label_largest_component(g))
g.purge_vertices()


###############################################################################
# Here is a simple function to extract the information of the inferred nested
# block model of  any graph-tool instance

def getNestedInference(g):
    '''Given a graph-tool instance this function extract the results
    obtained from  nested block model inference algorithm

    Parameters:
    -----------
        g: Graph or GraphView instance
            A graph with |V| vertices and |E| edges
    Returns:
    --------
        cts: list of arrays (float)
            A lists containing |E| arrays of different sizes. Each array
            represents a set of control points for each edge
        pos: array (float)
            A 2d array of |V|x2 size. Each row represents a  node positions
            in a circular layour
        labels: list (int)
            a list of  |V| int numbers. Each number represents the comunitiy
            which the node belongs
    '''
    state = gt.inference.minimize_nested_blockmodel_dl(g)
    # cts represent the control points of the Beizer curves used for graph-tool
    # to draw the edges
    tg = gt.inference.nested_blockmodel.get_hierarchy_tree(state)[0]
    tpos = pos = gt.draw.radial_tree_layout(
        tg, tg.vertex(tg.num_vertices() - 1), weighted=True)
    cts = gt.draw.get_hierarchy_control_points(g, tg, tpos)

    cts = list(cts)
    pos = np.array(list(g.own_property(tpos)))
    labels = list(state.levels[0].b)

    return cts, pos, labels


# Now, we can perform the nested block model inference
cts, pos, labels = getNestedInference(g)


###############################################################################
# Each different comunity obtained from the nested block model  algorithm will
# have a different color

comunityColors = cmap.distinguishable_colormap(
    nb_colors=len(np.unique(labels)))

nodeColors = np.array([comunityColors[c] for c in labels])


###############################################################################
# To plot the resuts of the inference we will need to draw curved lines
# in FURY. To do so, we  will use the BSpline interpolation with the  control's
# points  obtained from the function  getNestedInference.


def bSplineInterpolation(ctsCoord, points):
    tck, u = interpolate.splprep(ctsCoord.T)

    coord = interpolate.splev(points, tck)
    coord = np.array(coord)
    return coord

###############################################################################
# Graph-tool stores the control point coordinates  relative to the nodes
# positions. In order to use this control point to preform the interpolation we
# need first to convert those control to points into real coordinates


def gt2real(posSource, posTarget, ctp):
    '''
    Parameters:
    -----------
        posSource: ndarray (float)
            2d numpy array representing the source node position
        posTarget: ndarray (float)
            2d numpy array representing the target node position
        ctp: ndarray (float)
            2d numpy array representing the relative coordinates of
            a given control point

    Return:
    -------
        realPos: ndarray (float)
            2d numpy array representing the real coordinates of 
            a given control point (rel)

    '''
    xRel = posTarget - posSource
    norm = np.sqrt(np.dot(xRel, xRel))
    y = np.array([
        [0, -1],
        [1.0, 0]
    ])/norm
    yRel = np.dot(y, xRel)
    realPos = posSource + xRel*ctp[0] + yRel*ctp[1]
    return realPos


###############################################################################
# Almost finished. Now, just use the previous functions to compute the
# positions to draw the curved lines

edgePositions = []
edgeColors = []
for ctsByEdge, e in zip(cts, g.edges()):
    s = g.vertex_index[e.source()]
    t = g.vertex_index[e.target()]

    cs = nodeColors[s]
    ct = nodeColors[t]
    cs = np.array(rgb_to_hsv(*cs))
    ct = np.array(rgb_to_hsv(*ct))
    pS = pos[s]
    pT = pos[t]

    ctsByEdge = ctsByEdge.a
    # Here is another trick! To use the data given by graph-tool
    # remove the boundary conditions
    ctsByEdge = ctsByEdge[10:ctsByEdge.shape[0]-10]

    ctsByEdge = ctsByEdge.reshape((ctsByEdge.shape[0]//2, 2))
    controlsRealCoord = np.array([
        gt2real(pS, pT, rel)
        for rel in ctsByEdge
    ])

    tInterp = np.linspace(0, 1, num=50, endpoint=False)
    out = bSplineInterpolation(controlsRealCoord, tInterp)
    ids = range(0, out.shape[1], 2)
    numPoints = len(ids)
    for i in ids:
        fac = (i)/numPoints
        cl = (ct - cs) * fac + cs
        cl = hsv_to_rgb(*cl)
        edgeColors.append(cl)

        start = [out[0][i], out[1][i], 0]
        end = [out[0][i+1], out[1][i+1], 0]

        edgePositions.append([start, end])
        if i < np.max(ids):
            ss = [out[0][i+2], out[1][i+2], 0]
            edgePositions.append([end, ss])
            fac = (i+1)/numPoints
            cl = (ct - cs) * fac + cs
            cl = hsv_to_rgb(*cl)
            edgeColors.append(cl)

edgePositions = np.array(edgePositions)

###############################################################################
# Finally, just call Fury and see the results
pos3d = np.append(pos, np.zeros((pos.shape[0], 1)), axis=1)
sphere_actor = actor.sphere(centers=pos3d,
                            colors=nodeColors,
                            radii=0.05,
                            theta=8,
                            phi=8,
                            )


lines_actor = actor.line(edgePositions,
                         colors=edgeColors,
                         opacity=0.1,
                         )

scene = window.Scene()

scene.add(lines_actor)
scene.add(sphere_actor)

arr = window.snapshot(scene, "nbm.png", size=(600, 600))


plt.figure(dpi=600)
plt.imshow(arr)
plt.show()

# Your plot should look like this: https://ibb.co/dDkC3f2
