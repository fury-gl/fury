"""Flocking (boids) simulation of particles moving inside a box."""

import itertools

import numpy as np
from scipy.spatial.distance import cdist

# Distances below this are clamped to avoid dividing by zero.
_MIN_DISTANCE = 1e-6


def vec2vec_rotmat(u, v):
    """
    Compute the rotation matrices that rotate directions ``u`` onto ``v``.

    The rotation returned is the smallest one aligning the two directions: a
    rotation around the axis ``u x v``. When ``u`` and ``v`` are antiparallel,
    the result is a half turn around an axis perpendicular to ``u``. When one
    of the vectors has zero length, the identity is returned.

    Parameters
    ----------
    u : array_like, shape (3,) or (N, 3)
        Source directions. They do not need to be unit vectors.
    v : array_like, shape (3,) or (N, 3)
        Target directions. They do not need to be unit vectors.

    Returns
    -------
    ndarray, shape (3, 3) or (N, 3, 3)
        Rotation matrices ``R`` such that ``R @ u`` points along ``v``. A single
        matrix is returned when both ``u`` and ``v`` have shape (3,).

    Examples
    --------
    >>> import numpy as np
    >>> from fury.swarm import vec2vec_rotmat
    >>> rot = vec2vec_rotmat([1, 0, 0], [0, 1, 0])
    >>> np.round(rot @ np.array([1, 0, 0]), 6) + 0.0
    array([0., 1., 0.])
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    single = u.ndim == 1 and v.ndim == 1
    u, v = np.broadcast_arrays(np.atleast_2d(u), np.atleast_2d(v))

    u_norm = np.linalg.norm(u, axis=1, keepdims=True)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    valid = (u_norm[:, 0] > _MIN_DISTANCE) & (v_norm[:, 0] > _MIN_DISTANCE)
    u = np.divide(u, u_norm, out=np.zeros_like(u), where=u_norm > _MIN_DISTANCE)
    v = np.divide(v, v_norm, out=np.zeros_like(v), where=v_norm > _MIN_DISTANCE)

    cos = np.einsum("ij,ij->i", u, v)
    rot = np.tile(np.eye(3), (len(u), 1, 1))

    # Rodrigues' formula: R = I + K + K^2 / (1 + cos), where K is the
    # cross-product matrix of w = u x v.
    general = valid & (cos > -1 + 1e-8)
    w = np.cross(u[general], v[general])
    k = np.zeros((len(w), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -w[:, 2], w[:, 1]
    k[:, 1, 0], k[:, 1, 2] = w[:, 2], -w[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -w[:, 1], w[:, 0]
    rot[general] += k + (k @ k) / (1 + cos[general])[:, None, None]

    # Antiparallel: half turn around an axis perpendicular to u.
    opposite = valid & ~general
    if np.any(opposite):
        u_opp = u[opposite]
        helper = np.eye(3)[np.argmin(np.abs(u_opp), axis=1)]
        axis = np.cross(u_opp, helper)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        rot[opposite] = 2 * axis[:, :, None] * axis[:, None, :] - np.eye(3)

    return rot[0] if single else rot


def box_edges(box_size):
    """
    Return the 12 edges of an axis-aligned box centered at the origin.

    Parameters
    ----------
    box_size : array_like, shape (3,)
        Box lengths along x, y and z.

    Returns
    -------
    ndarray, shape (12, 2, 3)
        Start and end point of each edge. It can be passed directly to
        :func:`fury.actor.line` to draw the box wireframe.
    """
    half = 0.5 * np.asarray(box_size, dtype=float)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    edges = [
        (i, j)
        for i, j in itertools.combinations(range(8), 2)
        if np.count_nonzero(signs[i] != signs[j]) == 1
    ]
    return (signs * half)[np.array(edges)]


def _as_points(points, name):
    """
    Validate and copy an array of 3D points.

    Parameters
    ----------
    points : array_like, shape (N, 3) or None
        Points to validate. ``None`` gives an empty array.
    name : str
        Name used in the error message.

    Returns
    -------
    ndarray, shape (N, 3)
        Float copy of ``points``.
    """
    if points is None:
        return np.zeros((0, 3))
    points = np.array(points, dtype=float)
    if points.size == 0:
        return np.zeros((0, 3))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (M, 3).")
    return points


class Swarm:
    """
    Flock of boids moving inside a box, following Reynolds' boids rules.

    Each boid steers using its neighbors, the boids within
    ``neighbor_radius``:

    * cohesion: move toward the neighbors' average position;
    * alignment: match the neighbors' average velocity;
    * separation: move away from boids closer than ``separation_radius``,
      more strongly the closer they are.

    Boids also turn back when they get within ``wall_margin`` of a wall,
    avoid spherical obstacles and are drawn toward attractors. Obstacles and
    attractors move at constant velocity, bounce off the walls and off each
    other. All boids move at the constant ``speed``.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Initial boid positions, inside the box.
    velocities : array_like, shape (N, 3)
        Initial boid velocities.
    box_size : array_like, shape (3,), optional
        Lengths of the box, centered at the origin, along x, y and z.
    speed : float, optional
        Boid speed, in distance per unit of time.
    neighbor_radius : float, optional
        Distance within which boids see each other.
    separation_radius : float, optional
        Distance within which boids, and obstacle surfaces, push a boid away.
    cohesion_weight : float, optional
        Strength of the steering toward the neighbors' center.
    alignment_weight : float, optional
        Strength of the steering toward the neighbors' velocity.
    separation_weight : float, optional
        Strength of the push away from close boids and obstacles.
    wall_margin : float, optional
        Distance from the walls at which boids start turning back.
    turn_factor : float, optional
        Strength of the turn away from the walls.
    obstacle_positions : array_like, shape (M, 3), optional
        Centers of spherical obstacles.
    obstacle_velocities : array_like, shape (M, 3), optional
        Obstacle velocities. Obstacles are static by default.
    obstacle_radius : float, optional
        Radius of the obstacles.
    attractor_positions : array_like, shape (K, 3), optional
        Positions of the attractors.
    attractor_velocities : array_like, shape (K, 3), optional
        Attractor velocities. Attractors are static by default.
    attractor_radius : float, optional
        Collision radius of the attractors.
    attraction_radius : float, optional
        Distance within which boids are drawn toward an attractor.
    attraction_weight : float, optional
        Strength of the steering toward the attractors.

    Attributes
    ----------
    positions, velocities : ndarray, shape (N, 3)
        Current boid state, updated in place by :meth:`step`.
    obstacle_positions, obstacle_velocities : ndarray, shape (M, 3)
        Current obstacle state, updated in place by :meth:`step`.
    attractor_positions, attractor_velocities : ndarray, shape (K, 3)
        Current attractor state, updated in place by :meth:`step`.
    neighbors : ndarray, shape (N, N)
        ``neighbors[i, j]`` is True when boid ``j`` was a neighbor of boid
        ``i`` during the last step. A boid is never its own neighbor.

    Examples
    --------
    >>> import numpy as np
    >>> from fury.swarm import Swarm
    >>> rng = np.random.default_rng(0)
    >>> swarm = Swarm(rng.uniform(-30, 30, (50, 3)), rng.uniform(-1, 1, (50, 3)))
    >>> swarm.step()
    >>> swarm.positions.shape
    (50, 3)
    """

    def __init__(
        self,
        positions,
        velocities,
        *,
        box_size=(100, 100, 100),
        speed=1.0,
        neighbor_radius=12.5,
        separation_radius=3.0,
        cohesion_weight=0.005,
        alignment_weight=0.1,
        separation_weight=0.5,
        wall_margin=5.0,
        turn_factor=0.2,
        obstacle_positions=None,
        obstacle_velocities=None,
        obstacle_radius=2.0,
        attractor_positions=None,
        attractor_velocities=None,
        attractor_radius=2.0,
        attraction_radius=25.0,
        attraction_weight=0.02,
    ):
        """
        Initialize the swarm.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Initial boid positions, inside the box.
        velocities : array_like, shape (N, 3)
            Initial boid velocities.
        box_size : array_like, shape (3,), optional
            Lengths of the box, centered at the origin, along x, y and z.
        speed : float, optional
            Boid speed, in distance per unit of time.
        neighbor_radius : float, optional
            Distance within which boids see each other.
        separation_radius : float, optional
            Distance within which boids, and obstacle surfaces, push a boid
            away.
        cohesion_weight : float, optional
            Strength of the steering toward the neighbors' center.
        alignment_weight : float, optional
            Strength of the steering toward the neighbors' velocity.
        separation_weight : float, optional
            Strength of the push away from close boids and obstacles.
        wall_margin : float, optional
            Distance from the walls at which boids start turning back.
        turn_factor : float, optional
            Strength of the turn away from the walls.
        obstacle_positions : array_like, shape (M, 3), optional
            Centers of spherical obstacles.
        obstacle_velocities : array_like, shape (M, 3), optional
            Obstacle velocities. Obstacles are static by default.
        obstacle_radius : float, optional
            Radius of the obstacles.
        attractor_positions : array_like, shape (K, 3), optional
            Positions of the attractors.
        attractor_velocities : array_like, shape (K, 3), optional
            Attractor velocities. Attractors are static by default.
        attractor_radius : float, optional
            Collision radius of the attractors.
        attraction_radius : float, optional
            Distance within which boids are drawn toward an attractor.
        attraction_weight : float, optional
            Strength of the steering toward the attractors.
        """
        self.positions = np.array(positions, dtype=float)
        self.velocities = np.array(velocities, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3).")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities must have the same shape as positions.")

        self.box_size = np.broadcast_to(np.asarray(box_size, dtype=float), (3,))
        self.speed = speed
        self.neighbor_radius = neighbor_radius
        self.separation_radius = separation_radius
        self.cohesion_weight = cohesion_weight
        self.alignment_weight = alignment_weight
        self.separation_weight = separation_weight
        self.wall_margin = wall_margin
        self.turn_factor = turn_factor

        self.obstacle_positions = _as_points(obstacle_positions, "obstacle_positions")
        self.obstacle_velocities = self._mover_velocities(
            obstacle_velocities, self.obstacle_positions, "obstacle_velocities"
        )
        self.obstacle_radius = obstacle_radius

        self.attractor_positions = _as_points(
            attractor_positions, "attractor_positions"
        )
        self.attractor_velocities = self._mover_velocities(
            attractor_velocities, self.attractor_positions, "attractor_velocities"
        )
        self.attractor_radius = attractor_radius
        self.attraction_radius = attraction_radius
        self.attraction_weight = attraction_weight

        n_boids = len(self.positions)
        self.neighbors = np.zeros((n_boids, n_boids), dtype=bool)

    @staticmethod
    def _mover_velocities(velocities, positions, name):
        """
        Validate the velocities of obstacles or attractors.

        Parameters
        ----------
        velocities : array_like, shape (M, 3) or None
            Velocities, or None for static objects.
        positions : ndarray, shape (M, 3)
            Positions of the same objects.
        name : str
            Name used in the error message.

        Returns
        -------
        ndarray, shape (M, 3)
            Float copy of ``velocities``.
        """
        if velocities is None:
            return np.zeros_like(positions)
        velocities = np.array(velocities, dtype=float)
        if velocities.shape != positions.shape:
            raise ValueError(f"{name} must have the same shape as its positions.")
        return velocities

    def step(self, dt=1.0):
        """
        Advance the simulation by ``dt`` units of time.

        Steering and motion both scale with ``dt``, so the simulation runs at
        the same pace whatever the frame rate: e.g. 6 steps with ``dt=1/6``
        roughly match one step with ``dt=1``.

        Parameters
        ----------
        dt : float, optional
            Time step.
        """
        self.velocities[:] = self._steer(dt)
        self.positions += dt * self.velocities
        self._keep_inside(self.positions, self.velocities, 0.0)

        self.obstacle_positions += dt * self.obstacle_velocities
        self.attractor_positions += dt * self.attractor_velocities
        self._bounce_movers()
        self._keep_inside(
            self.obstacle_positions, self.obstacle_velocities, self.obstacle_radius
        )
        self._keep_inside(
            self.attractor_positions, self.attractor_velocities, self.attractor_radius
        )

    def _steer(self, dt):
        """
        Compute the boid velocities for the next step.

        Parameters
        ----------
        dt : float
            Time step.

        Returns
        -------
        ndarray, shape (N, 3)
            New velocities, with norm ``speed``.
        """
        pos, vel = self.positions, self.velocities

        dist = cdist(pos, pos)
        np.fill_diagonal(dist, np.inf)

        self.neighbors = dist <= self.neighbor_radius
        count = self.neighbors.sum(axis=1, keepdims=True)
        has_neighbors = count > 0
        weights = self.neighbors / np.maximum(count, 1)
        cohesion = np.where(has_neighbors, weights @ pos - pos, 0.0)
        alignment = np.where(has_neighbors, weights @ vel - vel, 0.0)

        too_close = dist <= self.separation_radius
        push = np.where(too_close, np.maximum(dist, _MIN_DISTANCE) ** -2, 0.0)
        # sum_j push[i, j] * (pos[i] - pos[j]), without building the
        # (N, N, 3) array of differences.
        separation = push.sum(axis=1, keepdims=True) * pos - push @ pos

        steer = (
            self.cohesion_weight * cohesion
            + self.alignment_weight * alignment
            + self.separation_weight * (separation + self._avoid_obstacles())
            + self.attraction_weight * self._follow_attractors()
        )

        limit = 0.5 * self.box_size - self.wall_margin
        steer += self.turn_factor * ((pos < -limit).astype(float) - (pos > limit))

        new_vel = vel + dt * steer
        norm = np.linalg.norm(new_vel, axis=1, keepdims=True)
        return np.where(
            norm > _MIN_DISTANCE,
            self.speed * new_vel / np.maximum(norm, _MIN_DISTANCE),
            vel,
        )

    def _avoid_obstacles(self):
        """
        Compute the push away from the obstacles.

        Returns
        -------
        ndarray, shape (N, 3)
            Sum, over the obstacles whose surface is within
            ``separation_radius``, of the unit vector away from the obstacle
            divided by the squared distance to its surface.
        """
        if not len(self.obstacle_positions):
            return np.zeros_like(self.positions)
        diff = self.positions[:, None, :] - self.obstacle_positions[None, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=2), _MIN_DISTANCE)
        gap = np.maximum(dist - self.obstacle_radius, _MIN_DISTANCE)
        push = np.where(gap <= self.separation_radius, gap**-2 / dist, 0.0)
        return np.einsum("ij,ijk->ik", push, diff)

    def _follow_attractors(self):
        """
        Compute the pull toward the attractors.

        Returns
        -------
        ndarray, shape (N, 3)
            Average offset from each boid to the attractors within
            ``attraction_radius``, zero when there are none.
        """
        if not len(self.attractor_positions):
            return np.zeros_like(self.positions)
        diff = self.attractor_positions[None, :, :] - self.positions[:, None, :]
        near = np.linalg.norm(diff, axis=2) <= self.attraction_radius
        count = np.maximum(near.sum(axis=1, keepdims=True), 1)
        return np.einsum("ij,ijk->ik", near, diff) / count

    def _bounce_movers(self):
        """Reverse obstacles and attractors that collide while approaching."""
        pos = np.concatenate([self.obstacle_positions, self.attractor_positions])
        if len(pos) < 2:
            return
        vel = np.concatenate([self.obstacle_velocities, self.attractor_velocities])
        radii = np.concatenate(
            [
                np.full(len(self.obstacle_positions), self.obstacle_radius),
                np.full(len(self.attractor_positions), self.attractor_radius),
            ]
        )
        diff = pos[:, None, :] - pos[None, :, :]
        overlap = np.linalg.norm(diff, axis=2) <= radii[:, None] + radii[None, :]
        np.fill_diagonal(overlap, False)
        approaching = np.einsum("ijk,ijk->ij", diff, vel[:, None] - vel[None]) < 0
        vel[(overlap & approaching).any(axis=1)] *= -1

        n_obstacles = len(self.obstacle_positions)
        self.obstacle_velocities[:] = vel[:n_obstacles]
        self.attractor_velocities[:] = vel[n_obstacles:]

    def _keep_inside(self, positions, velocities, radius):
        """
        Clamp objects inside the box and point their velocity back inward.

        Parameters
        ----------
        positions : ndarray, shape (M, 3)
            Positions, modified in place.
        velocities : ndarray, shape (M, 3)
            Velocities, modified in place.
        radius : float
            Object radius kept clear of the walls.
        """
        limit = 0.5 * self.box_size - radius
        low = positions < -limit
        high = positions > limit
        velocities[low] = np.abs(velocities[low])
        velocities[high] = -np.abs(velocities[high])
        np.clip(positions, -limit, limit, out=positions)
