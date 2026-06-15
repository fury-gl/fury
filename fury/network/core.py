"""Network actors."""

import numpy as np

from fury.actor import Actor, Line
from fury.geometry import Geometry, buffer_to_geometry
from fury.lib import (
    Buffer,
    PointsShader,
    WorldObject,
    register_wgpu_render_function,
)
from fury.material import LineMaterial, NetworkMaterial
from fury.shader import NetworkComputeShader


class Network(WorldObject, Actor):
    """
    Network actor that simulates force-directed layout on the GPU.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
        Initial positions of the nodes.
    edges : ndarray, shape (M, 2)
        Indices of connected nodes.
    colors : ndarray, shape (N, 4), optional
        Colors of the nodes.
    k : float, optional
        Optimal distance constant (affects equilibrium length).
    damping : float, optional
        Damping factor for velocity (0.0 to 1.0).
    repulsion_strength : float, optional
        Multiplier for repulsive forces.
    speed : float, optional
        Simulation speed factor.
    point_size : float, optional
        Size of the rendered nodes.
    edge_opacity : float, optional
        Opacity of the edges (0.0 to 1.0).
    """

    def __init__(
        self,
        nodes,
        edges,
        colors=None,
        k=10.0,
        damping=0.9,
        repulsion_strength=1.0,
        speed=1.0,
        point_size=15.0,
        edge_opacity=0.5,
    ):
        """
        Initialize the network actor instance.

        Parameters
        ----------
        nodes : ndarray, shape (N, 3)
            Initial positions of the nodes.
        edges : ndarray, shape (M, 2)
            Indices of connected nodes.
        colors : ndarray, shape (N, 4), optional
            Colors of the nodes.
        k : float, optional
            Optimal distance constant (affects equilibrium length).
        damping : float, optional
            Damping factor for velocity (0.0 to 1.0).
        repulsion_strength : float, optional
            Multiplier for repulsive forces.
        speed : float, optional
            Simulation speed factor.
        point_size : float, optional
            Size of the rendered nodes.
        edge_opacity : float, optional
            Opacity of the edges (0.0 to 1.0).
        """
        super().__init__()

        if not isinstance(nodes, np.ndarray) or nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes must be a (N, 3) numpy array")

        if not isinstance(edges, np.ndarray) or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be a (M, 2) numpy array")

        if np.any(edges < 0):
            raise ValueError(
                f"edges cannot contain negative indices. \
                    Found minimum value: {np.min(edges)}"
            )

        self.n_nodes = nodes.shape[0]

        if np.any(edges >= self.n_nodes):
            raise ValueError(
                f"edges cannot contain indices >= number of nodes ({self.n_nodes}). \
                    Found maximum value: {np.max(edges)}"
            )

        self.n_edges = edges.shape[0]

        velocities_data = np.zeros((self.n_nodes, 4), dtype=np.float32)

        adj = [[] for _ in range(self.n_nodes)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        flat_adj = []
        offsets = np.zeros(self.n_nodes, dtype=np.int32)
        counts = np.zeros(self.n_nodes, dtype=np.int32)

        current_offset = 0
        for i in range(self.n_nodes):
            neighbors = adj[i]
            offsets[i] = current_offset
            counts[i] = len(neighbors)
            flat_adj.extend(neighbors)
            current_offset += len(neighbors)

        flat_adj = np.array(flat_adj, dtype=np.int32)

        if colors is None:
            colors = np.ones((self.n_nodes, 4), dtype=np.float32)

        self.geometry = buffer_to_geometry(
            positions=nodes.astype(np.float32),
            colors=colors.astype(np.float32),
        )

        self.velocities_buffer = Buffer(velocities_data)
        self.adj_buffer = Buffer(flat_adj)
        self.offsets_buffer = Buffer(offsets)
        self.counts_buffer = Buffer(counts)

        self.material = NetworkMaterial(
            k=k,
            damping=damping,
            repulsion_strength=repulsion_strength,
            speed=speed,
            size=point_size,
            color_mode="vertex",
        )

        edge_geometry = Geometry()
        edge_geometry.positions = self.geometry.positions

        edge_geometry.indices = Buffer(edges.astype(np.int32).ravel())

        edge_material = LineMaterial(
            color=np.array([1.0, 1.0, 1.0, edge_opacity], dtype=np.float32),
            opacity=edge_opacity,
        )

        edge_actor = Line(edge_geometry, edge_material)

        self.add(edge_actor)


@register_wgpu_render_function(Network, NetworkMaterial)
def register_network_shaders(wobject):
    """
    Register and return compute and rendering shader steps for the actor pipeline.

    Parameters
    ----------
    wobject : Network
        The target network object component context.

    Returns
    -------
    tuple
        The configured compute shader and render shader pair layout tuple.
    """
    compute_shader = NetworkComputeShader(wobject)

    render_shader = PointsShader(wobject)
    return compute_shader, render_shader
