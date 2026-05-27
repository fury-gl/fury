import numpy as np
import numpy.testing as npt
import pytest

from fury.actor import Line
from fury.lib import Buffer, PointsShader
from fury.network import Network, NetworkMaterial, register_network_shaders


@pytest.fixture
def valid_nodes():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def valid_edges():
    return np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)


@pytest.fixture
def valid_colors():
    return np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_network_initialization_valid(valid_nodes, valid_edges):
    """Test standard initialization with default parameters."""
    network = Network(nodes=valid_nodes, edges=valid_edges)

    assert network.n_nodes == 3
    assert network.n_edges == 3

    assert network.geometry.positions.data.shape == (3, 3)
    assert network.geometry.colors.data.shape == (3, 4)

    assert isinstance(network.material, NetworkMaterial)
    assert network.material.k == 10.0
    assert network.material.damping == 0.9

    assert isinstance(network.velocities_buffer, Buffer)
    assert isinstance(network.adj_buffer, Buffer)
    assert isinstance(network.offsets_buffer, Buffer)
    assert isinstance(network.counts_buffer, Buffer)

    assert len(network.children) == 1
    assert isinstance(network.children[0], Line)


def test_network_initialization_with_colors(valid_nodes, valid_edges, valid_colors):
    """Test initialization with explicit color mapping."""
    network = Network(nodes=valid_nodes, edges=valid_edges, colors=valid_colors)
    npt.assert_array_almost_equal(network.geometry.colors.data, valid_colors)


def test_network_initialization_invalid_nodes():
    """Test constraints on node array input shape."""
    invalid_nodes = np.array([0.0, 1.0, 2.0])
    edges = np.array([[0, 1]])

    with pytest.raises(ValueError, match="nodes must be a \\(N, 3\\) numpy array"):
        Network(nodes=invalid_nodes, edges=edges)

    invalid_nodes_2d = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="nodes must be a \\(N, 3\\) numpy array"):
        Network(nodes=invalid_nodes_2d, edges=edges)


def test_network_initialization_invalid_edges(valid_nodes):
    """Test constraints on edge array input shape."""
    invalid_edges = np.array([0, 1])

    with pytest.raises(ValueError, match="edges must be a \\(M, 2\\) numpy array"):
        Network(nodes=valid_nodes, edges=invalid_edges)

    invalid_edges_2d = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="edges must be a \\(M, 2\\) numpy array"):
        Network(nodes=valid_nodes, edges=invalid_edges_2d)


def test_network_adjacency_flattening(valid_nodes):
    """Test that the adjacency list is flattened correctly for the compute shader."""

    nodes = np.zeros((4, 3))
    edges = np.array([[0, 1], [1, 2], [1, 3]])

    network = Network(nodes=nodes, edges=edges)

    expected_counts = [1, 3, 1, 1]
    npt.assert_array_equal(network.counts_buffer.data, expected_counts)

    expected_offsets = [0, 1, 4, 5]
    npt.assert_array_equal(network.offsets_buffer.data, expected_offsets)

    assert len(network.adj_buffer.data) == 6


def test_network_material_properties():
    """Test that Material property setters update the inner uniform_buffer."""
    mat = NetworkMaterial(
        k=5.0, damping=0.8, speed=2.0, repulsion_strength=1.5, size=12.0
    )

    assert mat.k == 5.0
    assert mat.damping == 0.8
    assert mat.speed == 2.0
    assert mat.repulsion_strength == 1.5

    mat.k = 12.0
    mat.damping = 0.5
    mat.speed = 3.0
    mat.repulsion_strength = 2.5

    assert mat.k == 12.0
    assert mat.uniform_buffer.data["k"] == 12.0

    assert mat.damping == 0.5
    assert mat.uniform_buffer.data["damping"] == 0.5

    assert mat.speed == 3.0
    assert mat.uniform_buffer.data["speed"] == 3.0

    assert mat.repulsion_strength == 2.5
    assert mat.uniform_buffer.data["repulsion_strength"] == 2.5


def test_register_network_shaders(valid_nodes, valid_edges):
    """Test that the shader registration function binds compute and points correctly."""
    network = Network(nodes=valid_nodes, edges=valid_edges)

    compute_shader, render_shader = register_network_shaders(network)

    assert compute_shader.type == "compute"
    assert isinstance(render_shader, PointsShader)
    assert compute_shader["n_nodes"] == network.n_nodes
