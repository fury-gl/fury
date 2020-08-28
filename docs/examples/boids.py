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
    cosa = np.clip(np.dot(u, v), -1, 1)

    sina = np.sqrt(1 - cosa ** 2)
    if (cosa > 1):
        print("cosa is greater than 1!!")
    R = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])
    Rp = np.dot(Pt, np.dot(R, P))

    # make sure that you don't return any Nans
    # check using the appropriate tool in numpy
    if np.any(np.isnan(Rp)):
        return np.eye(3)

    return Rp

def box_edges(box_lx, box_ly, box_lz):

    edge1 = 0.5 * np.array([[box_lx, box_ly, box_lz],
                            [box_lx, box_ly, -box_lz],
                            [-box_lx, box_ly, -box_lz],
                            [-box_lx, box_ly, box_lz],
                            [box_lx, box_ly, box_lz]])
    edge2 = 0.5 * np.array([[box_lx, box_ly, box_lz],
                            [box_lx, -box_ly, box_lz]])
    edge3 = 0.5 * np.array([[box_lx, box_ly, -box_lz],
                            [box_lx, -box_ly, -box_lz]])
    edge4 = 0.5 * np.array([[-box_lx, box_ly, -box_lz],
                            [-box_lx, -box_ly, -box_lz]])
    edge5 = 0.5 * np.array([[-box_lx, box_ly, box_lz],
                            [-box_lx, -box_ly, box_lz]])
    lines = [edge1, -edge1, edge2, edge3, edge4, edge5]
    return lines


##############################################################################
# Here we define collision between walls-particles and particle-particle.
# When collision happens, the particle with lower velocity gets the
# color of the particle with higher velocity

def collision():
    global xyz, center_leader, vel
    num_vertices = vertices.shape[0]
    sec = np.int(num_vertices / num_particles)

    for i, j in np.ndindex(num_particles, num_particles):

        if (i == j):
            continue
        distance = np.linalg.norm(xyz[i] - xyz[j])
        vel_mag_i = np.linalg.norm(vel[i])
        vel_mag_j = np.linalg.norm(vel[j])
        # Collision happens if the distance between the centars of two
        # particles is less or equal to the sum of their radii
        if (distance <= (radii[i] + radii[j])):
            vel[i] = -vel[i]
            vel[j] = -vel[j]
            if vel_mag_j > vel_mag_i:
                vcolors[i * sec: i * sec + sec] = \
                    vcolors[j * sec: j * sec + sec]
            if vel_mag_i > vel_mag_j:
                vcolors[j * sec: j * sec + sec] = \
                    vcolors[i * sec: i * sec + sec]
            xyz[i] = xyz[i] + vel[i] * dt
            xyz[j] = xyz[j] + vel[j] * dt
    # Collision between particles-walls;
    vel[:, 0] = np.where(((xyz[:, 0] <= - 0.5 * box_lx + radii[:]) |
                          (xyz[:, 0] >= (0.5 * box_lx - radii[:]))),
                         - vel[:, 0], vel[:, 0])
    vel[:, 1] = np.where(((xyz[:, 1] <= - 0.5 * box_ly + radii[:]) |
                          (xyz[:, 1] >= (0.5 * box_ly - radii[:]))),
                         - vel[:, 1], vel[:, 1])
    vel[:, 2] = np.where(((xyz[:, 2] <= -0.5 * box_lz + radii[:]) |
                          (xyz[:, 2] >= (0.5 * box_lz - radii[:]))),
                         - vel[:, 2], vel[:, 2])


global xyz, center_leader, vel
num_particles = 20
num_leaders = 1
steps = 1000
dt = 0.05
box_lx = 50
box_ly = 50
box_lz = 50
vel = 4 * (np.random.rand(num_particles, 3) - 0.5)
colors = np.random.rand(num_particles, 3)
xyz = np.array([box_lx, box_ly, box_lz]) * (np.random.rand(num_particles, 3)
                                            - 0.5) * 0.6
directions = np.random.rand(num_particles, 3)
scene = window.Scene()
arrow_actor = actor.arrow(centers=xyz,
                          directions=directions, colors=colors, heights=3,
                          resolution=10, vertices=None, faces=None)
scene.add(arrow_actor)

color_leader = np.array([[0, 1, 0.]])
center_leader = np.array([[10, 0 , 0.]])
direction_leader = np.array([[0, 1., 0]])
arrow_actor_leader = actor.arrow(centers=center_leader,
                          directions=direction_leader, colors=color_leader, heights=3,
                          resolution=10, vertices=None, faces=None)


scene.add(arrow_actor_leader)
axes_actor = actor.axes(scale=(1, 1, 1), colorx=(1, 0, 0), colory=(0, 1, 0), colorz=(0, 0, 1), opacity=1)
scene.add(axes_actor)
showm = window.ShowManager(scene,
                           size=(1200, 1000), reset_camera=True,
                           order_transparent=True)
showm.initialize()

tb = ui.TextBlock2D(bold=True)
counter = itertools.count()
vertices = utils.vertices_from_actor(arrow_actor)
no_vertices_per_arrow = len(vertices)/num_particles
initial_vertices = vertices.copy() - \
    np.repeat(xyz, no_vertices_per_arrow, axis=0)
vertices_leader = utils.vertices_from_actor(arrow_actor_leader)
initial_vertices_leader = vertices_leader.copy() - \
    np.repeat(center_leader, no_vertices_per_arrow, axis=0)


scene.zoom(0.4)

def timer_callback(_obj, _event):
    global xyz, center_leader, vel
    alpha = 0.5

    turnfraction = 0.01
    cnt = next(counter)
    dst = 5
    angle = 2 * np.pi * turnfraction * cnt
    x2 = (dst + 5) * np.cos(angle)
    y2 = (dst + 5) * np.sin(angle)

    angle_1 = 2 * np.pi * turnfraction * (cnt+1)
    x2_1 = (dst + 5) * np.cos(angle_1)
    y2_1 = (dst + 5) * np.sin(angle_1)

    xyz_leader = np.array([[x2, y2, 0.]])
    xyz_1_leader = np.array([[x2_1, y2_1, 0.]])

    tb.message = "Let's count up to 1000 and exit :" + str(cnt)
    mag_vel_leader = np.linalg.norm(xyz_1_leader - xyz_leader)

    vel_leader = np.array((xyz_1_leader[ 0,:] - xyz_leader[ 0,:])/np.linalg.norm(xyz_1_leader[0,:] - xyz_leader[0,:]))
    R = vec2vec_rotmat(vel_leader, np.array([0, 1., 0]))

    vel = vel / np.linalg.norm(vel, axis=1).reshape((num_particles, 1))
    vel = alpha * vel + (1 - alpha) * vel_leader
    vel = vel / np.linalg.norm(vel, axis=1).reshape((num_particles, 1))
    xyz = xyz + vel * mag_vel_leader

    # It rotates arrow at origin and then shifts to position;
    num_vertices = vertices.shape[0]
    sec = np.int(num_vertices / num_particles)

    for i in range(num_particles):
        R_followers = vec2vec_rotmat(vel[i], directions[i])
        vertices[i * sec: i * sec + sec] = np.dot(initial_vertices[i * sec: i * sec + sec], R_followers) + \
            np.repeat(xyz[i: i+1], no_vertices_per_arrow, axis=0)
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
