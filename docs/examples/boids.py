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
        # Collision happens if the distance between the centars of two
        # particles is less or equal to the sum of their radii
        if (distance <= 3): #(radii[i] + radii[j])):
            vel[i] = -vel[i]
            vel[j] = -vel[j]
            xyz[i] = xyz[i] + vel[i] * dt
            xyz[j] = xyz[j] + vel[j] * dt
    # Collision between particles-walls;
    vel[:, 0] = np.where(((xyz[:, 0] <= - 0.5 * box_lx + 3) |
                          (xyz[:, 0] >= (0.5 * box_lx - 3))),
                         - vel[:, 0], vel[:, 0])
    vel[:, 1] = np.where(((xyz[:, 1] <= - 0.5 * box_ly + 3) |
                          (xyz[:, 1] >= (0.5 * box_ly - 3))),
                         - vel[:, 1], vel[:, 1])
    vel[:, 2] = np.where(((xyz[:, 2] <= -0.5 * box_lz + 3) |
                          (xyz[:, 2] >= (0.5 * box_lz - 3))),
                         - vel[:, 2], vel[:, 2])


global xyz, center_leader, vel
num_particles = 200
num_leaders = 1
steps = 1000
dt = 0.05
box_lx = 100
box_ly = 100
box_lz = 100
vel = 4 * (np.random.rand(num_particles, 3) - 0.5)
colors = np.random.rand(num_particles, 3)
xyz = np.array([box_lx, box_ly, box_lz]) * (np.random.rand(num_particles, 3)
                                            - 0.5) * 0.6
directions = np.random.rand(num_particles, 3)
scene = window.Scene()
box_centers = np.array([[0, 0, 0]])
box_directions = np.array([[0, 1, 0]])
box_colors = np.array([[255, 255, 255]])
box_actor = actor.box(box_centers, box_directions, box_colors,
                      scale=(box_lx, box_ly, box_lz))
utils.opacity(box_actor, 0.2)
scene.add(box_actor)

lines = box_edges(box_lx, box_ly, box_lz)
line_actor = actor.streamtube(lines, colors=(1, 0.5, 0), linewidth=0.1)
scene.add(line_actor)
cone_actor = actor.cone(centers=xyz,
                          directions=directions, colors=colors, heights=3,
                          resolution=10, vertices=None, faces=None)
scene.add(cone_actor)

color_leader = np.array([[0, 1, 0.]])
center_leader = np.array([[10, 0 , 0.]])
direction_leader = np.array([[0, 1., 0]])
cone_actor_leader = actor.cone(centers=center_leader,
                          directions=direction_leader, colors=color_leader, heights=3,
                          resolution=10, vertices=None, faces=None)


scene.add(cone_actor_leader)
axes_actor = actor.axes(scale=(1, 1, 1), colorx=(1, 0, 0), colory=(0, 1, 0), colorz=(0, 0, 1), opacity=1)
scene.add(axes_actor)
showm = window.ShowManager(scene,
                           size=(1200, 1000), reset_camera=True,
                           order_transparent=True)
showm.initialize()

tb = ui.TextBlock2D(bold=True)
counter = itertools.count()
vertices = utils.vertices_from_actor(cone_actor)
no_vertices_per_cone = len(vertices)/num_particles
initial_vertices = vertices.copy() - \
    np.repeat(xyz, no_vertices_per_cone, axis=0)
vertices_leader = utils.vertices_from_actor(cone_actor_leader)
initial_vertices_leader = vertices_leader.copy() - \
    np.repeat(center_leader, no_vertices_per_cone, axis=0)


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
    # R = vec2vec_rotmat(vel_leader, np.array([0, 1., 0]))

    # Alignment: Alignment is a behavior that causes a particular agent(leader) to line up with agents(followers) close by.
    # Cohesion: Cohesion is a behavior that causes agents(followers) to steer towards the "center of mass" - that is, the average position of the agents(followers) within a certain radius.

    for i in range(num_particles):
        neighborCount = 0
        vel_in_sphere = np.array([0, 0, 0.])
        center_mass_in_sphere = np.array([0, 0, 0.])
        separation = np.array([0, 0, 0.])
        for j in range(num_particles):
            if (i == j):
                continue
            distance = np.linalg.norm(xyz[i] - xyz[j])
            if (distance <= 10):
                vel_in_sphere += (vel[i] + vel[j])  # Alignment
                center_mass_in_sphere += (xyz[i] + xyz[j])  # Cohesion
                separation += (xyz[i] - xyz[j])
                neighborCount = neighborCount + 1 # number of neighbors in sphere

        if (neighborCount > 0):
            vel_in_sphere = (vel_in_sphere / neighborCount) # Alignment
            center_mass_in_sphere = xyz[i] - center_mass_in_sphere / neighborCount # Cohesion
            separation = -1 * (separation / neighborCount)
            vel_in_sphere = vel_in_sphere / np.linalg.norm(vel_in_sphere)
            center_mass_in_sphere = center_mass_in_sphere / np.linalg.norm(center_mass_in_sphere)
            separation = separation / np.linalg.norm(separation)

            # print('before', np.linalg.norm(vel[i]))
            vel[i] += 0.333 * vel_in_sphere +  0.0 * center_mass_in_sphere  +  0.666 * separation
            # print('after', np.linalg.norm(vel[i]))
            vel[i] = vel[i] / np.linalg.norm(vel[i])
            #print(np.linalg.norm(vel[i]), np.linalg.norm(vel_in_sphere), np.linalg.norm(center_mass_in_sphere), np.linalg.norm(separation))
            #xyz[i] = xyz[i] + vel[i] / np.linalg.norm(vel[i]) # equation of motion
        #else:
            #xyz[i] = xyz[i] + vel[i] / np.linalg.norm(vel[i])

    R = vec2vec_rotmat(vel_leader, np.array([0, 1., 0]))
    # vel = alpha * vel + (1 - alpha) * vel_leader
    vel = vel / np.linalg.norm(vel, axis=1).reshape((num_particles, 1))
    xyz = xyz + vel
    collision()

    # It rotates arrow at origin and then shifts to position;
    num_vertices = vertices.shape[0]
    sec = np.int(num_vertices / num_particles)
    for i in range(num_particles):
        R_followers = vec2vec_rotmat(vel[i], directions[i])
        vertices[i * sec: i * sec + sec] = np.dot(initial_vertices[i * sec: i * sec + sec], R_followers) + \
            np.repeat(xyz[i: i+1], no_vertices_per_cone, axis=0)
        utils.update_actor(cone_actor)

    vertices_leader[:] = np.dot(initial_vertices_leader, R) + \
        np.repeat(xyz_leader, no_vertices_per_cone, axis=0)
    utils.update_actor(cone_actor_leader)
    scene.reset_clipping_range()
    showm.render()
    if cnt == steps:
        showm.exit()

scene.add(tb)

showm.add_timer_callback(True, 10, timer_callback)
showm.start()
