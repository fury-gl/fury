import numpy as np
import numpy.testing as npt
from fury import actor, utils, swarm


def test_vec2vec_rotmat():

    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])

    real_R_test1 = np.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 1]], 'f8')

    R = swarm.vec2vec_rotmat(u, v)
    npt.assert_array_almost_equal(R, real_R_test1)

    u = np.array([1, 0, 0])
    v = np.array([1, 0, 0])

    real_R_test2 = np.array([[1., 0., 0.],
                             [0., 1., 0.],
                             [0., 0., 1.]])

    R = swarm.vec2vec_rotmat(u, v)
    npt.assert_array_almost_equal(R, real_R_test2)

    u = np.array([0, 0, 0])
    v = np.array([1, 1, 1])

    real_R_test3 = np.array([[-1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., -1.]])

    R = swarm.vec2vec_rotmat(u, v)
    npt.assert_array_almost_equal(R, real_R_test3)


def test_box_edges():

    gm = swarm.GlobalMemory()
    gm.box_lx = gm.box_ly = gm.box_lz = 50

    lines = swarm.box_edges((gm.box_lx, gm.box_ly, gm.box_lz))
    upper_frame, lower_frame, column_1, column_2, column_3, column_4 = lines

    expected_upper_frame = np.array([[25.,  25.,  25.], [25., 25., -25.],
                                     [-25., 25., -25.], [-25., 25., 25.],
                                     [25., 25., 25.]])
    expected_lower_frame = np.array([[-25., -25., -25.], [-25., -25.,  25.],
                                     [25., -25.,  25.], [25., -25., -25.],
                                     [-25., -25., -25.]])
    expected_column_1 = np.array([[25.,  25.,  25.], [25., -25.,  25.]])
    expected_column_2 = np.array([[25.,  25., -25.], [25., -25., -25.]])
    expected_column_3 = np.array([[-25.,  25., -25.], [-25., -25., -25.]])
    expected_column_4 = np.array([[-25.,  25.,  25.], [-25., -25.,  25.]])

    npt.assert_array_almost_equal(upper_frame, expected_upper_frame)
    npt.assert_array_almost_equal(lower_frame, expected_lower_frame)
    npt.assert_array_almost_equal(column_1, expected_column_1)
    npt.assert_array_almost_equal(column_2, expected_column_2)
    npt.assert_array_almost_equal(column_3, expected_column_3)
    npt.assert_array_almost_equal(column_4, expected_column_4)


def test_collision_particle_walls():
    gm = swarm.GlobalMemory()
    gm.height_cones = 1
    gm.num_particles = 1
    gm.turnfactor = 1
    gm.box_lx = gm.box_ly = gm.box_lz = 50
    gm.pos = np.array([[0, -27, 27.]])
    gm.vel = np.array([[0, 1, 0.]])
    swarm.collision_particle_walls(gm)
    expected_vel = np.array([[0., 0.89442719, -0.4472136]])
    npt.assert_array_almost_equal(gm.vel, expected_vel)


def test_boids_rules():
    gm = swarm.GlobalMemory()
    gm.height_cones = 1
    gm.num_particles = 2
    gm.turnfactor = 1
    gm.num_obstacles = 1
    gm.num_attractors = 1
    gm.box_lx = gm.box_ly = gm.box_lz = 50
    gm.pos = np.array([[0, -27, 27.], [0, -25, 27.]])
    gm.vel = np.array([[0, 1, 0.], [0, 3, 0.]])
    gm.pos_attractors = np.array([[0, -20, 27]])
    gm.pos_obstacles = np.array([[0, -26, 27.]])
    cone_actor = actor.cone(centers=gm.pos,
                            directions=gm.vel.copy(), colors=gm.colors,
                            heights=gm.height_cones,
                            resolution=10, vertices=None, faces=None)
    gm.vertices = utils.vertices_from_actor(cone_actor)
    vcolors = utils.colors_from_actor(cone_actor, 'colors')
    swarm.boids_rules(gm, gm.vertices, vcolors)
    expected_vel = np.array([[0., 7.75, 0], [0., 15.625, 0]])
    npt.assert_array_almost_equal(gm.vel, expected_vel)


def test_collision_obstacle_attractors_walls():
    gm = swarm.GlobalMemory()
    gm.radii_obstacles = 1
    gm.num_obstacles = 3
    expected_distance_obstacles = 0
    expected_distance_attractors = 0
    distance_obstacles = 0
    distance_attractors = 0
    gm.pos_obstacles = np.array([[0, 26, 0.], [0, 25, 0.], [0, 2, 0.]])
    gm.vel_obstacles = np.array([[0, 26, 0.], [0, 25, 0.], [0., 2., 0.]])

    gm.radii_attractors = 1
    gm.num_attractors = 3
    gm.pos_attractors = np.array([[23, 0, 0.], [24, 0, 0.], [0., 1., 0.]])
    gm.vel_attractors = np.array([[23, 0, 0.], [24, 0, 0.], [0., 1., 0.]])

    gm.box_lx = gm.box_ly = gm.box_lz = 50
    distance_attractors = np.linalg.norm(gm.pos_attractors[0] -
                                         gm.pos_attractors[1])
    distance_obstacles = np.linalg.norm(gm.pos_obstacles[0] -
                                        gm.pos_obstacles[1])
    expected_distance_obstacles = 1
    expected_distance_attractors = 1

    swarm.collision_obstacle_attractors_walls(gm)
    expected_vel_obstacle = np.array([[0, -26, 0.], [0, -25, 0.],
                                      [0., -2., 0.]])
    expected_vel_attractors = np.array([[23, 0, 0.], [-24, 0, 0.],
                                        [0., -1., 0.]])
    npt.assert_array_almost_equal(distance_attractors,
                                  expected_distance_attractors)
    npt.assert_array_almost_equal(distance_obstacles,
                                  expected_distance_obstacles)
    npt.assert_array_almost_equal(gm.vel_attractors, expected_vel_attractors)
    npt.assert_array_almost_equal(gm.vel_obstacles, expected_vel_obstacle)


test_vec2vec_rotmat()
test_box_edges()
test_collision_particle_walls()
test_boids_rules()
test_collision_obstacle_attractors_walls()
