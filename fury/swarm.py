"""
================================
Flock simulation in a box
================================

This is an example of boids in a box using FURY.
"""

##############################################################################
# Explanation:

import numpy as np
from fury import disable_warnings

disable_warnings()

# The GlobalMemory has the initial parameters of simulation time, box,
# particles, attractors and abstacles


class GlobalMemory(object):

    def __init__(self):
        self.cnt = 0
        self.steps = 10000
        self.tm_step = 100

        # Initial parameters for box
        self.box_lx = 100
        self.box_ly = 100
        self.box_lz = 100
        self.box_centers = np.array([[0, 0, 0]])
        self.box_directions = np.array([[0, 1, 0]])
        self.box_colors = np.array([[255, 255, 255]])
        self.size_box = [self.box_lx, self.box_ly, self.box_lz]

        # Initial parameters for particles. The particles are shown by cones
        self.num_particles = 100
        self.height_cones = 1
        self.turnfactor = 1
        self.vel = np.array([0, 0, 0.])
        self.pos = np.array([0, 0, 0.])
        self.directions = np.array([0, 0, 0.])
        self.colors = np.random.rand(self.num_particles, 3)
        self.vertices = None

        # Initial parameters for obstacles. The obstacles are shown by spheres
        self.num_obstacles = 0
        self.radii_obstacles = 2.0
        self.pos_obstacles = np.array(self.size_box) * \
                                     (np.random.rand(self.num_obstacles, 3) -
                                      0.5) * 0.6
        self.vel_obstacles = np.random.rand(self.num_obstacles, 3)
        self.color_obstacles = np.random.rand(self.num_obstacles, 3) * 0.5

        # Initial parameters for attractors. The attractors are shown by cones
        self.num_attractors = 0
        self.radii_attractors = 2
        self.pos_attractors = np.array(self.size_box) * \
                                      (np.random.rand(self.num_attractors, 3) -
                                       0.5) * 0.6
        self.vel_attractors = np.random.rand(self.num_attractors, 3)
        self.color_attractors = np.random.rand(self.num_attractors, 3)


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


def box_edges(box_size):

    box_lx, box_ly, box_lz = box_size

    upper_frame = 0.5 * np.array([[box_lx, box_ly, box_lz],
                                 [box_lx, box_ly, -box_lz],
                                 [-box_lx, box_ly, -box_lz],
                                 [-box_lx, box_ly, box_lz],
                                 [box_lx, box_ly, box_lz]])
    column_1 = 0.5 * np.array([[box_lx, box_ly, box_lz],
                              [box_lx, -box_ly, box_lz]])
    column_2 = 0.5 * np.array([[box_lx, box_ly, -box_lz],
                              [box_lx, -box_ly, -box_lz]])
    column_3 = 0.5 * np.array([[-box_lx, box_ly, -box_lz],
                              [-box_lx, -box_ly, -box_lz]])
    column_4 = 0.5 * np.array([[-box_lx, box_ly, box_lz],
                              [-box_lx, -box_ly, box_lz]])
    lower_frame = - upper_frame
    lines = [upper_frame, lower_frame, column_1, column_2, column_3, column_4]
    return lines


##############################################################################
# Here we define collision between walls-particles.
def collision_particle_walls(gm, normalize=True):
    gm.vel[:, 0] = np.where((gm.pos[:, 0] <= (-0.5 * gm.box_lx +
                            gm.height_cones)), gm.vel[:, 0]
                            + gm.turnfactor, gm.vel[:, 0])
    gm.vel[:, 0] = np.where((gm.pos[:, 0] >= (0.5 * gm.box_lx
                            - gm.height_cones)), gm.vel[:, 0] -
                            gm.turnfactor, gm.vel[:, 0])
    gm.vel[:, 1] = np.where((gm.pos[:, 1] <= (-0.5 * gm.box_ly +
                            gm.height_cones)), gm.vel[:, 1] + gm.turnfactor,
                            gm.vel[:, 1])
    gm.vel[:, 1] = np.where((gm.pos[:, 1] >= (0.5 * gm.box_ly -
                            gm.height_cones)), gm.vel[:, 1] - gm.turnfactor,
                            gm.vel[:, 1])
    gm.vel[:, 2] = np.where((gm.pos[:, 2] <= (-0.5 * gm.box_lz +
                            gm.height_cones)), gm.vel[:, 2] + gm.turnfactor,
                            gm.vel[:, 2])
    gm.vel[:, 2] = np.where((gm.pos[:, 2] >= (0.5 * gm.box_lz -
                            gm.height_cones)), gm.vel[:, 2] - gm.turnfactor,
                            gm.vel[:, 2])
    if normalize:
        gm.vel = gm.vel / np.linalg.norm(gm.vel, axis=1).reshape(
                                        (gm.vel.shape[0], 1))


def boids_rules(gm, vertices, vcolors):

    num_vertices = vertices.shape[0]
    sec = np.int(num_vertices / gm.num_particles)

    for i in range(gm.num_particles):
        neighborCount = 0
        cohesion = np.array([0, 0, 0.])
        separation = np.array([0, 0, 0.])
        alignment = gm.vel[i].copy()
        avoid_collision = np.array([0, 0, 0.])
        follow_attractor = np.array([0, 0, 0.])
        distance_particle_attractors = np.array([0, 0, 0.])
        avoid_obstacles = np.array([0, 0, 0.])
        count_attracts = 0
        count_repuls = 0

        for a in range(gm.num_obstacles):
            distance_particle_obstacle = np.linalg.norm(gm.pos[i] -
                                                        gm.pos_obstacles[a])
            if (distance_particle_obstacle > 0.0) and (
                distance_particle_obstacle <= (gm.radii_obstacles +
                                               gm.height_cones)):
                diff = gm.pos[i] - gm.pos_obstacles[a]
                inv_sqr_magnitude = 1 / ((np.linalg.norm(diff) -
                                         (gm.radii_obstacles +
                                          gm.height_cones)) ** 2)
                avoid_obstacles += gm.vel[i] + (inv_sqr_magnitude * diff)
                count_repuls += 1
        if count_repuls > 0:
            avoid_obstacles = avoid_obstacles/count_repuls

        for k in range(gm.num_attractors):
            distance_particle_attractors = np.linalg.norm(gm.pos[i] -
                                                          gm.pos_attractors[k])
            if ((distance_particle_attractors > 0.0) and
                    (distance_particle_attractors <= gm.box_lx / 4)):
                follow_attractor += (gm.pos_attractors[k] - gm.pos[i])
                count_attracts += 1
        if count_attracts > 0:
            follow_attractor = follow_attractor/count_attracts

        for j in range(gm.num_particles):
            if (i == j):
                continue
            distance_particle_particle = np.linalg.norm(gm.pos[i] - gm.pos[j])
            if distance_particle_particle <= (gm.height_cones * 2):
                diff = gm.pos[i] - gm.pos[j]
                inv_sqr_magnitude = 1 / ((np.linalg.norm(diff) -
                                          gm.height_cones) ** 2)
                avoid_collision = avoid_collision + (inv_sqr_magnitude * diff)

            if ((distance_particle_particle > 0.0) and
                    (distance_particle_particle <= gm.box_lx / 8)):
                vcolors[i * sec: i * sec + sec] = \
                    vcolors[j * sec: j * sec + sec]
                alignment += gm.vel[j]
                cohesion += gm.pos[j]
                separation += gm.pos[i] - gm.pos[j]
                neighborCount += 1  # number of neighbors in sphere

        if neighborCount > 0:
            cohesion = cohesion / (neighborCount)
            cohesion = (cohesion - gm.pos[i])
            alignment = (alignment / (neighborCount + 1))
            alignment = alignment - gm.vel[i]
            separation = separation / neighborCount

        gm.vel[i] += avoid_collision + separation + alignment + cohesion + \
            follow_attractor + avoid_obstacles


def collision_obstacle_attractors_walls(gm):
    # Collosion between attractors-attractors, obstacle-obstacle,
    # obstacle-walls and attractors-walls:
    # Obstacle-obstacle:
    for i, j in np.ndindex(gm.num_obstacles, gm.num_obstacles):
        if (i == j):
            continue
        distance_obstacles = np.linalg.norm(gm.pos_obstacles -
                                            gm.pos_obstacles)
        if (distance_obstacles <= (gm.radii_obstacles + gm.radii_obstacles)):
            gm.vel_obstacles[i] = -gm.vel_obstacles[i]
            gm.vel_obstacles[j] = -gm.vel_obstacles[j]
    # attractors-attractors:
    for i, j in np.ndindex(gm.num_attractors, gm.num_attractors):
        if (i == j):
            continue
        distance_attractors = np.linalg.norm(gm.pos_attractors -
                                             gm.pos_attractors)
        if distance_attractors <= (gm.radii_attractors + gm.radii_attractors):
            gm.vel_attractors[i] = -gm.vel_attractors[i]
            gm.vel_attractors[j] = -gm.vel_attractors[j]
#  attractors-obstacle:
    for i, j in np.ndindex(gm.num_attractors, gm.num_obstacles):
        distance_attractors_obstacle = np.linalg.norm(gm.pos_attractors[i] -
                                                      gm.pos_obstacles[j])
        if (distance_attractors_obstacle <= (gm.radii_attractors +
                                             gm.radii_obstacles)):
            gm.vel_attractors[i] = -gm.vel_attractors[i]
            gm.vel_obstacles[j] = -gm.vel_obstacles[j]
#  Obstacle-walls;
    gm.vel_obstacles[:, 0] = np.where(
        ((gm.pos_obstacles[:, 0] <= - 0.5 * gm.box_lx + gm.radii_obstacles) |
         (gm.pos_obstacles[:, 0] >= 0.5 * gm.box_lx - gm.radii_obstacles)), -
        gm.vel_obstacles[:, 0], gm.vel_obstacles[:, 0])
    gm.vel_obstacles[:, 1] = np.where(
        ((gm.pos_obstacles[:, 1] <= - 0.5 * gm.box_ly + gm.radii_obstacles) |
         (gm.pos_obstacles[:, 1] >= 0.5 * gm.box_ly - gm.radii_obstacles)), -
        gm.vel_obstacles[:, 1], gm.vel_obstacles[:, 1])
    gm.vel_obstacles[:, 2] = np.where(
        ((gm.pos_obstacles[:, 2] <= -0.5 * gm.box_lz + gm.radii_obstacles) |
         (gm.pos_obstacles[:, 2] >= 0.5 * gm.box_lz - gm.radii_obstacles)), -
        gm.vel_obstacles[:, 2], gm.vel_obstacles[:, 2])
# attractors-walls;
    gm.vel_attractors[:, 0] = np.where(
        ((gm.pos_attractors[:, 0] <= - 0.5 * gm.box_lx + gm.radii_attractors) |
         (gm.pos_attractors[:, 0] >= 0.5 * gm.box_lx - gm.radii_attractors)), -
        gm.vel_attractors[:, 0], gm.vel_attractors[:, 0])
    gm.vel_attractors[:, 1] = np.where(
        ((gm.pos_attractors[:, 1] <= - 0.5 * gm.box_ly + gm.radii_attractors) |
         (gm.pos_attractors[:, 1] >= 0.5 * gm.box_ly - gm.radii_attractors)), -
        gm.vel_attractors[:, 1], gm.vel_attractors[:, 1])
    gm.vel_attractors[:, 2] = np.where(
        ((gm.pos_attractors[:, 2] <= -0.5 * gm.box_lz + gm.radii_attractors) |
         (gm.pos_attractors[:, 2] >= 0.5 * gm.box_lz - gm.radii_attractors)), -
        gm.vel_attractors[:, 2], gm.vel_attractors[:, 2])
