"""
================================
Flock simulation in a box
================================

This is an example of boids in a box using FURY.
"""

##############################################################################
# Explanation:

import numpy as np
import math
from fury import window, actor, ui, utils, primitive
import itertools


def vec2vec_rotmat(u, v):
    r""" rotation matrix from 2 unit vectors
    u, v being unit 3d vectors return a 3x3 rotation matrix R than aligns u to
    v.
    In general there are many rotations that will map u to v. If S is any
    rotation using v as an axis then R.S will also map u to v since (S.R)u =
    S(Ru) = Sv = v.  The rotation R returned by vec2vec_rotmat leaves fixed the
    perpendicular to the plane spanned by u and v.
    The transpose of R will align v to u.

    Parameters
    -----------
    u : array, shape(3,)
    v : array, shape(3,)

    Returns
    ---------
    R : array, shape(3,3)

    Examples
    ---------
    >>> import numpy as np
    >>> from dipy.core.geometry import vec2vec_rotmat
    >>> u=np.array([1,0,0])
    >>> v=np.array([0,1,0])
    >>> R=vec2vec_rotmat(u,v)
    >>> np.dot(R,u)
    array([ 0.,  1.,  0.])
    >>> np.dot(R.T,v)
    array([ 1.,  0.,  0.])
    """

    # Cross product is the first step to find R
    # Rely on numpy instead of manual checking for failing
    # cases
    w = np.cross(u, v)
    wn = np.linalg.norm(w)

    # Check that cross product is OK and vectors
    # u, v are not collinear (norm(w)>0.0)
    if np.isnan(wn) or wn < np.finfo(float).eps:
        norm_u_v = np.linalg.norm(u - v)
        # This is the case of two antipodal vectors:
        # ** former checking assumed norm(u) == norm(v)
        if norm_u_v > np.linalg.norm(u):
            return -np.eye(3)
        return np.eye(3)

    # if everything ok, normalize w
    w = w / wn

    # vp is in plane of u,v,  perpendicular to u
    vp = (v - (np.dot(u, v) * u))
    vp = vp / np.linalg.norm(vp)

    # (u vp w) is an orthonormal basis
    P = np.array([u, vp, w])
    Pt = P.T
    cosa = np.dot(u, v)
    sina = np.sqrt(1 - cosa ** 2)
    R = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])
    Rp = np.dot(Pt, np.dot(R, P))

    # make sure that you don't return any Nans
    # check using the appropriate tool in numpy
    if np.any(np.isnan(Rp)):
        return np.eye(3)

    return Rp

global centers, center_leader
num_particles = 2
num_leaders = 1
steps = 1000
dt = 0.05
vel = np.random.rand(2, 3)
colors = np.array([[0.5, 0.5, 0.5],
                   [0.5, 0, 0.5]])
centers = 0 * np.array([[10, 0, 0.],
                        [13 + 3, 0, 0.]])
directions = np.array([[-np.sqrt(2)/2, np.sqrt(2)/2, 0],
                       [np.sqrt(2)/2, np.sqrt(2)/2, 0.]])
scene = window.Scene()
arrow_actor = actor.arrow(centers=centers,
                          directions=directions, colors=colors, heights=3,
                          resolution=10, vertices=None, faces=None)
scene.add(arrow_actor)

color_leader = np.array([[0, 1, 0.]])
center_leader = 0 * np.array([[10 + 3, 0 , 0.]])
direction_leader = np.array([[0, 1., 0]])
arrow_actor_leader = actor.arrow(centers=center_leader,
                          directions=direction_leader, colors=color_leader, heights=3,
                          resolution=10, vertices=None, faces=None)


scene.add(arrow_actor_leader)
axes_actor = actor.axes(scale=(1, 1, 1), colorx=(1, 0, 0), colory=(0, 1, 0), colorz=(0, 0, 1), opacity=1)
scene.add(axes_actor)
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=True,
                           order_transparent=True)
showm.initialize()

tb = ui.TextBlock2D(bold=True)
counter = itertools.count()
vertices = utils.vertices_from_actor(arrow_actor)
no_vertices_per_arrow = len(vertices)/num_particles
initial_vertices = vertices.copy() - \
    np.repeat(centers, no_vertices_per_arrow, axis=0)
vertices_leader = utils.vertices_from_actor(arrow_actor_leader)
initial_vertices_leader = vertices_leader.copy() - \
    np.repeat(center_leader, no_vertices_per_arrow, axis=0)


scene.zoom(0.8)

def timer_callback(_obj, _event):
    global centers, center_leader
    turnfraction = 0.01
    cnt = next(counter)
    dst = 5
    angle = 2 * np.pi * turnfraction * cnt
    x1 = dst #* np.cos(angle)
    x2 = (dst + 5) * np.cos(angle)
    x3 = (dst + 10) #* np.cos(angle)
    y1 = dst #* np.sin(angle)
    y2 = (dst + 5) * np.sin(angle)
    y3 = (dst + 10) #* np.sin(angle)

    angle_1 = 2 * np.pi * turnfraction * (cnt+1)
    x1_1 = dst #* np.cos(angle_1)
    x2_1 = (dst + 5) * np.cos(angle_1)
    x3_1 = (dst + 10) #* np.cos(angle_1)
    y1_1 = dst #* np.sin(angle_1)
    y2_1 = (dst + 5) * np.sin(angle_1)
    y3_1 = (dst + 10) #* np.sin(angle_1)

    xyz = np.array([[x1, y1, 0.],
                   [x3, y3, 0.]])
    xyz_1 = np.array([[x1_1, y1_1, 0.],
                     [x3_1, y3_1, 0.]])

    xyz_leader = np.array([[x2, y2, 0.]])
    xyz_1_leader = np.array([[x2_1, y2_1, 0.]])

    tb.message = "Let's count up to 1000 and exit :" + str(cnt)
    R = vec2vec_rotmat(np.array((xyz_1_leader[0,:] - xyz_leader[0,:])/np.linalg.norm(xyz_1_leader[0,:] - xyz_leader[0,:])), np.array([0, 1., 0]))
    xyz = xyz + vel * cnt

    vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_arrow, axis=0)
    utils.update_actor(arrow_actor)

    vertices_leader[:] = np.dot(initial_vertices_leader, R) + \
        np.repeat(xyz_leader, no_vertices_per_arrow, axis=0)
    utils.update_actor(arrow_actor_leader)
    scene.reset_clipping_range()
    showm.render()
    if cnt == steps:
        showm.exit()

scene.add(tb)

showm.add_timer_callback(True, 50, timer_callback)
showm.start()
