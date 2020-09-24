import numpy as np
import numpy.testing as npt
from fury import window, actor, ui, utils, swarm
import itertools


def test_vec2vec_rotmat():

    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])

    real_R = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 1]], 'f8')

    R = swarm.vec2vec_rotmat(u, v)
    npt.assert_array_almost_equal(R, real_R)


def test_box_edges():

    gm = swarm.GlobalMemory()
    gm.box_lx = gm.box_ly = gm.box_lz = 50

    lines = swarm.box_edges((gm.box_lx, gm.box_ly, gm.box_lz))
    upper_frame, lower_frame, column_1, column_2, column_3, column_4 = lines

    expected_upper_frame = np.array([[25.,  25.,  25.],
                               [25.,  25., -25.],
                               [-25.,  25., -25.],
                               [-25.,  25.,  25.],
                               [25.,  25.,  25.]])

    npt.assert_array_almost_equal(upper_frame, expected_upper_frame)


def test_collision_particle_walls():
    gm = swarm.GlobalMemory()
    gm.height_cones = 1
    gm.num_particles = 1
    gm.turnfactor = 1
    gm.box_lx = gm.box_ly = gm.box_lz = 50
    gm.pos = np.array([[0, 26, 0.]])
    gm.vel = np.array([[0, 4, 0.]])
    swarm.collision_particle_walls(gm)
    expected_vel = np.array([[0, 1., 0.]])
    npt.assert_array_almost_equal(gm.vel, expected_vel)


def test_boids_rules():
    gm = swarm.GlobalMemory()
    gm.height_cones = 1
    gm.num_particles = 1
    gm.turnfactor = 1
    gm.box_lx = gm.box_ly = gm.box_lz = 50
    gm.pos = np.array([[0, 26, 0.]])
    gm.vel = np.array([[0, 4, 0.]])
    cone_actor = actor.cone(centers=gm.pos,
                        directions=gm.vel.copy(), colors=gm.colors,
                        heights=gm.height_cones,
                        resolution=10, vertices=None, faces=None)
    gm.vertices = utils.vertices_from_actor(cone_actor)
    swarm.boids_rules(gm)
    expected_vel = [[0. ,8., 0.]]
    npt.assert_array_almost_equal(gm.vel, expected_vel)


def test_collision_obstacle_leader_walls():
    gm = swarm.GlobalMemory()
    gm.radii_obstacles = 1
    gm.num_obstacles = 1
    gm.pos_obstacles = np.array([[0, 26, 0.]])
    gm.vel_obstacles = np.array([[0, 26, 1.]])

    gm.radii_leaders = 1
    gm.num_leaders = 1
    gm.pos_leaders = np.array([[0, 26, 0.]])
    gm.vel_leaders = np.array([[0, 4, 4.]])

    gm.box_lx = gm.box_ly = gm.box_lz = 50
    swarm.collision_obstacle_leader_walls(gm)
    expected_vel_leaders = [[0., 4., -4.]]
    print(gm.vel_obstacles)
    print(gm.vel_leaders)
    npt.assert_array_almost_equal(gm.vel_leaders, expected_vel_leaders)
    # npt.assert_array_almost_equal(gm.vel_obstacles, expected_vel_obstacle)


# test_vec2vec_rotmat()
# test_box_edges()
# test_collision_particle_walls()
# test_boids_rules()
test_collision_obstacle_leader_walls()
