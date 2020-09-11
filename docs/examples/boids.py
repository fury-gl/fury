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
from fury import window, actor, ui, utils, primitive, disable_warnings
import itertools

disable_warnings()

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

def normalize_all_vels(num_particles):
    global vel
    vel = vel / np.linalg.norm(vel, axis=1).reshape((num_particles, 1))


##############################################################################
# Here we define collision between walls-particles and particle-particle.
# When collision happens, the particle with lower velocity gets the
# color of the particle with higher velocity

# def collision_walls(particle_size):
#     global pos, box_centers, vel
#     periodic_boundary = False
#     if periodic_boundary == True:
#         pos[:, 0] = np.where((pos[:, 0] < - 0.5 * box_lx), (0.5 * box_lx), pos[:, 0])
#         pos[:, 0] = np.where((pos[:, 0] > 0.5 * box_lx), (-0.5 * box_lx), pos[:, 0])

#         pos[:, 1] = np.where((pos[:, 1] < - 0.5 * box_lx), (0.5 * box_lx), pos[:, 1])
#         pos[:, 1] = np.where((pos[:, 1] > 0.5 * box_lx), (-0.5 * box_lx), pos[:, 1])

#         pos[:, 2] = np.where((pos[:, 2] < - 0.5 * box_lx), (0.5 * box_lx), pos[:, 2])
#         pos[:, 2] = np.where((pos[:, 2] > 0.5 * box_lx), (-0.5 * box_lx), pos[:, 2])

#     else:
#         vel[:, 0] = np.where(((pos[:, 0] <= - 0.5 * box_lx + particle_size) |
#                             (pos[:, 0] >= 0.5 * box_lx - particle_size)),
#                             - vel[:, 0], vel[:, 0])
#         vel[:, 1] = np.where(((pos[:, 1] <= - 0.5 * box_ly + particle_size) |
#                             (pos[:, 1] >= 0.5 * box_ly - particle_size)),
#                             - vel[:, 1], vel[:, 1])
#         vel[:, 2] = np.where(((pos[:, 2] <= -0.5 * box_lz + particle_size) |
#                             (pos[:, 2] >= 0.5 * box_lz - particle_size)),
#                             - vel[:, 2], vel[:, 2])
#         pos = pos + vel

def keepWithinBounds():
    particle_size = 2
    turnfactor = 1
    vel[:, 0] = np.where((pos[:, 0] <= (-0.5 * box_lx + particle_size)), vel[:, 0] + turnfactor, vel[:, 0])
    vel[:, 0] = np.where((pos[:, 0] >= (0.5 * box_lx - particle_size)), vel[:, 0] - turnfactor, vel[:, 0])

    vel[:, 1] = np.where((pos[:, 1] <= (-0.5 * box_ly + particle_size)), vel[:, 1] + turnfactor, vel[:, 1])
    vel[:, 1] = np.where((pos[:, 1] >= (0.5 * box_ly - particle_size)), vel[:, 1] - turnfactor, vel[:, 1])

    vel[:, 2] = np.where((pos[:, 2] <= (-0.5 * box_lz + particle_size)), vel[:, 2] + turnfactor, vel[:, 2])
    vel[:, 2] = np.where((pos[:, 2] >= (0.5 * box_lz - particle_size)), vel[:, 2] - turnfactor, vel[:, 2])


def cohesion_alignment_separation():
    global pos, vel
    centeringFactor = 1
    minDistance = 2
    avoidFactor = 1
    speedLimit = 15

    for i in range(num_particles):
        neighborCount = 0
        neighborCount_collision = 0

        cohesion = np.array([0, 0, 0.])
        separation = np.array([0, 0, 0.])
        aligment = vel[i].copy() #np.array([0, 0, 0.])
        collision = np.array([0, 0, 0.])
        for j in range(num_particles):
            if (i == j):
                continue
            distance = np.linalg.norm(pos[i] - pos[j])
            if (distance <= 7 ):
                aligment += vel[j]  # Alignment
                cohesion += pos[j]  # Cohesion
                separation += pos[i] - pos[j]
                neighborCount += 1 # number of neighbors in sphere
                # print (neighborCount)

        if neighborCount > 0:
            aligment = (aligment / neighborCount) # Alignment
            aligment = (aligment - vel[i])
            cohesion = cohesion / neighborCount # Cohesion
            cohesion = (cohesion - pos[i])
            separation = separation / neighborCount
            # separation = separation / np.linalg.norm(separation)
            # aligment = aligment / np.linalg.norm(aligment)
            # cohesion = cohesion / np.linalg.norm(cohesion)
            # vel[i] += vel[i] * avoidFactor
            # collision = collision / np.linalg.norm(collision)

            # if np.linalg.norm(separation) > 0:
            vel[i] += 0.333 * separation + 0.333 * aligment  + 0.333 * cohesion #+ 0.0 * collision
            vel[i] = vel[i] / np.linalg.norm (vel[i])
        else:
            vel[i] = vel[i].copy() #np.array([0, 0, 0.])
            vel[i] = vel[i] / np.linalg.norm (vel[i])
            # velocity = np.linalg.norm(vel[i])


        # if velocity > speedLimit:
        #     vel[i] = (vel[i]/velocity) * speedLimit


global pos, vel
num_particles = 50
num_leaders = 1
steps = 10000
dt = 0.05
box_lx = 100
box_ly = 100
box_lz = 100
test_rules = False
colors = np.random.rand(num_particles, 3)
if test_rules == True:
    vel = np.array([[-np.sqrt(2)/2, np.sqrt(2)/2, 0], [0, 1., 0], [np.sqrt(2)/2, np.sqrt(2)/2, 0]])
    pos = np.array([[-5, 0., 0], [0, 0., 0], [5, 0., 0]])
    directions = np.array([[-np.sqrt(2)/2, np.sqrt(2)/2, 0], [0, 1., 0], [np.sqrt(2)/2, np.sqrt(2)/2, 0]])
else:
    vel = (np.random.rand(num_particles, 3))
    pos = np.array([box_lx, box_ly, box_lz]) * (np.random.rand(num_particles, 3) - 0.5) * 0.6
    directions = np.random.rand(num_particles, 3)


scene = window.Scene()
box_centers = np.array([[0, 0, 0]])
box_directions = np.array([[0, 1, 0]])
box_colors = np.array([[255, 255, 255]])
box_actor = actor.box(box_centers, box_directions, box_colors,
                      scales=(box_lx, box_ly, box_lz))
utils.opacity(box_actor, 0.)
scene.add(box_actor)

lines = box_edges(box_lx, box_ly, box_lz)
line_actor = actor.streamtube(lines, colors=(1, 0.5, 0), linewidth=0.1)
scene.add(line_actor)
# cone_actor = actor.sphere(centers=pos,
#                             colors=colors,
#                             radii=1)
# scene.add(cone_actor)

cone_actor = actor.cone(centers=pos,
                          directions=directions, colors=colors, heights=2,
                          resolution=10, vertices=None, faces=None)
scene.add(cone_actor)

sphere_actor = actor.sphere(centers=box_centers,
                            colors=np.array([[0, 0, 1]]),
                            radii=10)
# scene.add(sphere_actor)
# color_leader = np.array([[0, 1, 0.]])
# center_leader = np.array([[10, 0 , 0.]])
# direction_leader = np.array([[0, 1., 0]])
# cone_actor_leader = actor.cone(centers=center_leader,
#                           directions=direction_leader, colors=color_leader, heights=2,
#                           resolution=10, vertices=None, faces=None)


# scene.add(cone_actor_leader)
axes_actor = actor.axes(scale=(1, 1, 1), colorx=(1, 0, 0), colory=(0, 1, 0), colorz=(0, 0, 1), opacity=1)
scene.add(axes_actor)
showm = window.ShowManager(scene,
                           size=(3000, 2000), reset_camera=True,
                           order_transparent=True)
showm.initialize()

tb = ui.TextBlock2D(bold=True)
counter = itertools.count()
vertices = utils.vertices_from_actor(cone_actor)
no_vertices_per_cone = len(vertices)/num_particles
initial_vertices = vertices.copy() - \
    np.repeat(pos, no_vertices_per_cone, axis=0)

# vertices_leader = utils.vertices_from_actor(cone_actor_leader)
# initial_vertices_leader = vertices_leader.copy() - \
#     np.repeat(center_leader, no_vertices_per_cone, axis=0)
scene.zoom(1.2)

def timer_callback(_obj, _event):
    global pos, vel
    # alpha = 0.5

    # turnfraction = 0.01
    cnt = next(counter)
    # dst = 5
    # angle = 2 * np.pi * turnfraction * cnt
    # x2 = (dst + 5) * np.cos(angle)
    # y2 = (dst + 5) * np.sin(angle)
    # angle_1 = 2 * np.pi * turnfraction * (cnt+1)
    # x2_1 = (dst + 5) * np.cos(angle_1)
    # y2_1 = (dst + 5) * np.sin(angle_1)
    # pos_leader = np.array([[x2, y2, 0.]])
    # pos_1_leader = np.array([[x2_1, y2_1, 0.]])

    tb.message = "Let's count up to 1000 and exit :" + str(cnt)
    # mag_vel_leader = np.linalg.norm(pos_1_leader - pos_leader)
    # vel_leader = np.array((pos_1_leader[ 0,:] - pos_leader[ 0,:])/np.linalg.norm(pos_1_leader[0,:] - pos_leader[0,:]))
    # R = vec2vec_rotmat(vel_leader, np.array([0, 1., 0]))
    # vel = alpha * vel + (1 - alpha) * vel_leader


    # velocity normalization
    cohesion_alignment_separation()

    normalize_all_vels(num_particles)
    pos = pos + vel
    keepWithinBounds()
    vel = vel / np.linalg.norm(vel, axis=1).reshape((num_particles, 1))

    # It rotates arrow at origin and then shifts to position;
    num_vertices = vertices.shape[0]
    sec = np.int(num_vertices / num_particles)
    for i in range(num_particles):
        R_followers = vec2vec_rotmat(vel[i], directions[i])
        vertices[i * sec: i * sec + sec] = np.dot(initial_vertices[i * sec: i * sec + sec], R_followers) + \
            np.repeat(pos[i: i+1], no_vertices_per_cone, axis=0)
        utils.update_actor(cone_actor)


    # vertices[:] = initial_vertices + \
    #     np.repeat(pos, no_vertices_per_cone, axis=0)
    # utils.update_actor(cone_actor)

    # vertices_leader[:] = np.dot(initial_vertices_leader, R) + \
    #     np.repeat(pos_leader, no_vertices_per_cone, axis=0)
    # utils.update_actor(cone_actor_leader)
    scene.reset_clipping_range()
    showm.render()
    if cnt == steps:
        showm.exit()

scene.add(tb)

showm.add_timer_callback(True, 100, timer_callback)
showm.start()
