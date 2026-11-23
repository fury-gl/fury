import numpy as np
import numpy.testing as npt
import pytest

from fury.swarm import Swarm, box_edges, vec2vec_rotmat


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _assert_rotations(rot):
    npt.assert_allclose(
        rot @ np.swapaxes(rot, -1, -2),
        np.broadcast_to(np.eye(3), rot.shape),
        atol=1e-10,
    )
    npt.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-10)


def test_vec2vec_rotmat_aligns_random_vectors():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(200, 3))
    v = rng.normal(size=(200, 3)) * 5
    rot = vec2vec_rotmat(u, v)
    assert rot.shape == (200, 3, 3)
    _assert_rotations(rot)
    npt.assert_allclose(np.einsum("nij,nj->ni", rot, _unit(u)), _unit(v), atol=1e-10)


def test_vec2vec_rotmat_special_cases():
    rot = vec2vec_rotmat([1, 0, 0], [0, 2, 0])
    assert rot.shape == (3, 3)
    npt.assert_allclose(rot @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    # Broadcasting one source direction to many targets.
    assert vec2vec_rotmat([1, 0, 0], np.eye(3)).shape == (3, 3, 3)

    npt.assert_allclose(vec2vec_rotmat([0, 0, 3], [0, 0, 1]), np.eye(3), atol=1e-12)
    npt.assert_array_equal(vec2vec_rotmat([0, 0, 0], [0, 0, 1]), np.eye(3))
    npt.assert_array_equal(vec2vec_rotmat([1, 0, 0], [0, 0, 0]), np.eye(3))

    # Antiparallel vectors need a proper rotation, not a reflection.
    u = np.array([[1, 0, 0], [0, 1, 0], [1, 2, 3]], dtype=float)
    rot = vec2vec_rotmat(u, -u)
    _assert_rotations(rot)
    npt.assert_allclose(np.einsum("nij,nj->ni", rot, _unit(u)), -_unit(u), atol=1e-12)


def test_box_edges():
    size = np.array([4.0, 6.0, 10.0])
    edges = box_edges(size)
    assert edges.shape == (12, 2, 3)

    offsets = edges[:, 1] - edges[:, 0]
    axis = np.argmax(np.abs(offsets), axis=1)
    npt.assert_array_equal(np.bincount(axis), [4, 4, 4])
    npt.assert_allclose(np.abs(offsets).sum(axis=1), size[axis])
    npt.assert_allclose(np.abs(edges), np.broadcast_to(size / 2, edges.shape))


def _swarm(positions, velocities, **kwargs):
    params = {
        "cohesion_weight": 0.0,
        "alignment_weight": 0.0,
        "separation_weight": 0.0,
        "box_size": (1000, 1000, 1000),
    }
    params.update(kwargs)
    return Swarm(positions, velocities, **params)


def test_swarm_lone_boid_moves_straight_at_constant_speed():
    swarm = _swarm([[0, 0, 0]], [[0, 0, 3]], speed=2.0, cohesion_weight=1.0)
    for _ in range(5):
        swarm.step()
    npt.assert_allclose(swarm.positions, [[0, 0, 10]])
    npt.assert_allclose(swarm.velocities, [[0, 0, 2]])
    assert not swarm.neighbors.any()

    swarm.step(dt=0.25)
    npt.assert_allclose(swarm.positions, [[0, 0, 10.5]])


def test_swarm_cohesion_pulls_neighbors_together():
    swarm = _swarm(
        [[-4, 0, 0], [4, 0, 0]], [[0, 0, 1], [0, 0, 1]], cohesion_weight=0.05
    )
    swarm.step()
    npt.assert_array_equal(swarm.neighbors, [[False, True], [True, False]])
    assert np.ptp(swarm.positions[:, 0]) < 8

    far = _swarm(
        [[-40, 0, 0], [40, 0, 0]], [[0, 0, 1], [0, 0, 1]], cohesion_weight=0.05
    )
    far.step()
    npt.assert_allclose(far.positions[:, 0], [-40, 40])


def test_swarm_alignment_matches_headings():
    swarm = _swarm(
        [[-1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]], alignment_weight=0.5
    )
    cos_before = 0.0
    swarm.step()
    cos_after = np.dot(*_unit(swarm.velocities))
    assert cos_after > cos_before
    npt.assert_allclose(np.linalg.norm(swarm.velocities, axis=1), 1.0)


def test_swarm_separation_pushes_close_boids_apart():
    swarm = _swarm(
        [[-0.5, 0, 0], [0.5, 0, 0]], [[0, 0, 1], [0, 0, 1]], separation_weight=0.5
    )
    swarm.step()
    assert np.ptp(swarm.positions[:, 0]) > 1


def test_swarm_stays_inside_box():
    rng = np.random.default_rng(1)
    swarm = Swarm(
        rng.uniform(-8, 8, (40, 3)),
        rng.normal(size=(40, 3)),
        box_size=(20, 20, 20),
        obstacle_positions=[[0, 0, 0]],
        obstacle_velocities=[[0.7, 0.3, 0.0]],
        attractor_positions=[[3, 3, 3]],
        attractor_velocities=[[-0.4, 0.9, 0.2]],
    )
    for _ in range(300):
        swarm.step()
        assert np.all(np.abs(swarm.positions) <= 10)
        assert np.all(np.abs(swarm.obstacle_positions) <= 10 - swarm.obstacle_radius)
        assert np.all(np.abs(swarm.attractor_positions) <= 10 - swarm.attractor_radius)


def test_swarm_boids_avoid_obstacles():
    swarm = _swarm(
        [[-20, 0.1, 0]],
        [[1, 0, 0]],
        separation_weight=0.5,
        obstacle_positions=[[0, 0, 0]],
        obstacle_radius=5.0,
    )
    for _ in range(40):
        swarm.step()
        assert np.linalg.norm(swarm.positions[0]) > 5.0


def test_swarm_boids_follow_attractors():
    swarm = _swarm(
        [[0, 0, 0]],
        [[0, 1, 0]],
        attraction_weight=0.2,
        attraction_radius=30.0,
        attractor_positions=[[20, 0, 0]],
    )
    for _ in range(30):
        swarm.step()
    assert np.linalg.norm(swarm.positions[0] - [20, 0, 0]) < 5.0

    ignored = _swarm(
        [[0, 0, 0]],
        [[0, 1, 0]],
        attraction_weight=0.2,
        attraction_radius=10.0,
        attractor_positions=[[20, 0, 0]],
    )
    ignored.step()
    npt.assert_allclose(ignored.positions, [[0, 1, 0]])


def test_swarm_movers_bounce_off_each_other():
    swarm = _swarm(
        np.zeros((1, 3)),
        [[0, 0, 1]],
        obstacle_positions=[[-1.5, 0, 0]],
        obstacle_velocities=[[1, 0, 0]],
        obstacle_radius=1.0,
        attractor_positions=[[1.5, 0, 0]],
        attractor_velocities=[[-1, 0, 0]],
        attractor_radius=1.0,
    )
    swarm.step()
    npt.assert_allclose(swarm.obstacle_velocities, [[-1, 0, 0]])
    npt.assert_allclose(swarm.attractor_velocities, [[1, 0, 0]])
    # Separating objects are not reversed again.
    swarm.step()
    npt.assert_allclose(swarm.obstacle_velocities, [[-1, 0, 0]])


def test_swarm_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Swarm(np.zeros((3, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        Swarm(np.zeros((3, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        Swarm(np.zeros((3, 3)), np.zeros((3, 3)), obstacle_positions=[1, 2])
    with pytest.raises(ValueError):
        Swarm(
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            attractor_positions=np.zeros((2, 3)),
            attractor_velocities=np.zeros((1, 3)),
        )
