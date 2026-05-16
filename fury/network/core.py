"""Network actors."""

from math import ceil

import numpy as np

from fury.actor import Line, Mesh
from fury.geometry import Geometry, buffer_to_geometry
from fury.lib import (
    BaseShader,
    Binding,
    Buffer,
    PointsShader,
    register_wgpu_render_function,
)
from fury.material import LineMaterial, PointsMaterial


class Network(Mesh):
    """Network actor that simulates force-directed layout on the GPU.

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
        """Initialize the network actor instance.

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

        self.n_nodes = nodes.shape[0]
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


class NetworkMaterial(PointsMaterial):
    """Material handling simulation parameters for the Network.

    Parameters
    ----------
    k : float, optional
        Optimal distance constant.
    damping : float, optional
        Damping factor for velocity.
    speed : float, optional
        Simulation speed factor.
    repulsion_strength : float, optional
        Multiplier for repulsive forces.
    size : float, optional
        Size of the rendered nodes.
    **kwargs : dict
        Additional parameters passed to PointsMaterial.
    """

    uniform_type = dict(
        PointsMaterial.uniform_type,
        k="f4",
        damping="f4",
        speed="f4",
        repulsion_strength="f4",
    )

    def __init__(
        self,
        k=10.0,
        damping=0.9,
        speed=1.0,
        repulsion_strength=1.0,
        size=10.0,
        **kwargs,
    ):
        """Initialize the material instance.

        Parameters
        ----------
        k : float, optional
            Optimal distance constant.
        damping : float, optional
            Damping factor for velocity.
        speed : float, optional
            Simulation speed factor.
        repulsion_strength : float, optional
            Multiplier for repulsive forces.
        size : float, optional
            Size of the rendered nodes.
        **kwargs : dict
            Additional parameters passed to PointsMaterial.
        """
        super().__init__(size=size, **kwargs)
        self.k = k
        self.damping = damping
        self.speed = speed
        self.repulsion_strength = repulsion_strength

    @property
    def k(self):
        """Get the optimal distance constant.

        Returns
        -------
        float
            The optimal distance constant.
        """
        return self.uniform_buffer.data["k"]

    @k.setter
    def k(self, v):
        """Set the optimal distance constant.

        Parameters
        ----------
        v : float
            The new optimal distance value.
        """
        self.uniform_buffer.data["k"] = v
        self.uniform_buffer.update_full()

    @property
    def damping(self):
        """Get the velocity damping factor.

        Returns
        -------
        float
            The damping factor.
        """
        return self.uniform_buffer.data["damping"]

    @damping.setter
    def damping(self, v):
        """Set the velocity damping factor.

        Parameters
        ----------
        v : float
            The new damping factor value.
        """
        self.uniform_buffer.data["damping"] = v
        self.uniform_buffer.update_full()

    @property
    def speed(self):
        """Get the simulation speed factor.

        Returns
        -------
        float
            The simulation speed factor.
        """
        return self.uniform_buffer.data["speed"]

    @speed.setter
    def speed(self, v):
        """Set the simulation speed factor.

        Parameters
        ----------
        v : float
            The new simulation speed value.
        """
        self.uniform_buffer.data["speed"] = v
        self.uniform_buffer.update_full()

    @property
    def repulsion_strength(self):
        """Get the repulsive force multiplier.

        Returns
        -------
        float
            The repulsion strength multiplier.
        """
        return self.uniform_buffer.data["repulsion_strength"]

    @repulsion_strength.setter
    def repulsion_strength(self, v):
        """Set the repulsive force multiplier.

        Parameters
        ----------
        v : float
            The new repulsion strength multiplier.
        """
        self.uniform_buffer.data["repulsion_strength"] = v
        self.uniform_buffer.update_full()


class NetworkComputeShader(BaseShader):
    """Compute Shader implementing Fruchterman-Reingold layout.

    Parameters
    ----------
    wobject : Network
        The target network actor instance utilizing this shader.
    """

    type = "compute"

    def __init__(self, wobject):
        """Initialize the compute shader instance.

        Parameters
        ----------
        wobject : Network
            The target network actor instance utilizing this shader.
        """
        super().__init__(wobject)
        self["workgroup_size"] = (64, 1, 1)
        self["n_nodes"] = wobject.n_nodes

    def get_render_info(self, wobject, _shared):
        """Get workgroup dispatch sizing metrics for the compute pipeline.

        Parameters
        ----------
        wobject : Network
            The network object instance being simulated.
        _shared : Any
            Shared context pipeline references.

        Returns
        -------
        dict
            A dictionary containing workgroup dimension sizes.
        """
        n = int(ceil(wobject.n_nodes / 64))
        return {
            "indices": (n, 1, 1),
        }

    def get_bindings(self, wobject, _shared, _scene):
        """Define and structure pipeline data resources for compute stages.

        Parameters
        ----------
        wobject : Network
            The network object instance providing buffer bindings.
        _shared : Any
            Shared context pipeline references.
        _scene : Any
            The scene object context.

        Returns
        -------
        dict
            A layout dictionary index mapping configured binding structures.
        """
        bindings = {
            0: Binding(
                "u_material",
                "buffer/uniform",
                wobject.material.uniform_buffer,
                "COMPUTE",
            ),
            1: Binding(
                "s_positions", "buffer/storage", wobject.geometry.positions, "COMPUTE"
            ),
            2: Binding(
                "s_velocities", "buffer/storage", wobject.velocities_buffer, "COMPUTE"
            ),
            3: Binding("s_adj", "buffer/storage", wobject.adj_buffer, "COMPUTE"),
            4: Binding(
                "s_offsets", "buffer/storage", wobject.offsets_buffer, "COMPUTE"
            ),
            5: Binding("s_counts", "buffer/storage", wobject.counts_buffer, "COMPUTE"),
        }
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, _wobject, _shared):
        """Return an empty pipeline structure map configuration.

        Parameters
        ----------
        _wobject : Network
            The target network instance.
        _shared : Any
            Shared pipeline context metadata.

        Returns
        -------
        dict
            An empty dictionary.
        """
        return {}

    def get_code(self):
        """Get the structural WGSL computer execution source block string.

        Returns
        -------
        str
            The layout compute kernel source code string.
        """
        return """
        {{ bindings_code }}

        const N_NODES = i32({{ n_nodes }});

        fn get_pos(i: i32) -> vec3<f32> {
            let idx = i * 3;
            return vec3<f32>(s_positions[idx], s_positions[idx+1], s_positions[idx+2]);
        }

        fn set_pos(i: i32, p: vec3<f32>) {
            let idx = i * 3;
            s_positions[idx] = p.x;
            s_positions[idx+1] = p.y;
            s_positions[idx+2] = p.z;
        }

        @compute @workgroup_size{{ workgroup_size }}
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = i32(global_id.x);
            if (i >= N_NODES) { return; }

            let pos_i = get_pos(i);
            var force = vec3<f32>(0.0);

            let k = u_material.k;
            let k_sq = k * k;

            // 1. Repulsion
            for (var j: i32 = 0; j < N_NODES; j++) {
                if (i == j) { continue; }

                let pos_j = get_pos(j);
                let delta = pos_i - pos_j;
                let dist_sq = dot(delta, delta);
                let dist = sqrt(dist_sq);

                if (dist > 0.001) {
                    force += (delta / dist_sq) * k_sq * u_material.repulsion_strength;
                } else {
                    force += vec3<f32>(1.0, 0.0, 0.0) * k_sq;
                }
            }

            // 2. Attraction
            let start = s_offsets[i];
            let count = s_counts[i];

            for (var idx: i32 = 0; idx < count; idx++) {
                let neighbor_idx = s_adj[start + idx];
                let pos_j = get_pos(neighbor_idx);
                let delta = pos_i - pos_j;
                let dist_sq = dot(delta, delta);
                let dist = sqrt(dist_sq);

                force -= delta * dist / k;
            }

            // 3. Integration
            var vel = s_velocities[i].xyz;
            vel = (vel + force * 0.01) * u_material.damping;

            let current_speed = length(vel);
            let max_speed = k * 2.0;
            if (current_speed > max_speed) {
                vel = normalize(vel) * max_speed;
            }

            let new_pos = pos_i + vel * u_material.speed;

            set_pos(i, new_pos);
            s_velocities[i] = vec4<f32>(vel, 0.0);
        }
        """


@register_wgpu_render_function(Network, NetworkMaterial)
def register_network_shaders(wobject):
    """Register and return compute and rendering shader steps for the actor pipeline.

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
