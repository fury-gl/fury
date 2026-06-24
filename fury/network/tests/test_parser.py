import numpy as np
import numpy.testing as npt
import pytest

from fury.network.parser import parse_network, stringify_network


@pytest.fixture
def sample_gexf_data():
    return """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz"\
      version="1.2">
    <graph mode="static" defaultedgetype="directed">
        <nodes>
            <node id="0" label="Node 1">
                <viz:color r="255" g="0" b="0" a="1.0"/>
                <viz:position x="10.0" y="20.0" z="5.0"/>
            </node>
            <node id="1" label="Node 2">
                <viz:color r="0" g="255" b="0" a="0.5"/>
                <viz:position x="-5.0" y="0.0" z="0.0"/>
            </node>
        </nodes>
        <edges>
            <edge id="e1" source="0" target="1" weight="2.5" type="directed"/>
        </edges>
    </graph>
</gexf>"""


@pytest.fixture
def sample_gml_data():
    return """graph [
  directed 1
  node [
    id 0
    graphics [
        x 15.0
        y 25.0
        z 0.0
        color 0.5
        color 0.5
        color 0.5
        color 1.0
    ]
  ]
  node [
    id 1
    graphics [
        x -10.0
        y -10.0
        z -10.0
        color 1.0
        color 0.0
        color 0.0
        color 1.0
    ]
  ]
  edge [
    source 0
    target 1
  ]
]"""


@pytest.fixture
def sample_xnet_data():
    return """#vertices 2
"0"
"1"
#edges nonweighted undirected
0 1
#v "position" v3
10 20 30
-10 -20 -30
#v "color" v4
1 0 0 1
0 1 0 0.5
"""


def test_gexf_parse_valid(sample_gexf_data):
    nodes_xyz, edges_indices, colors = parse_network(sample_gexf_data, "gexf")

    assert nodes_xyz.shape == (2, 3)
    assert edges_indices.shape == (1, 2)
    assert colors.shape == (2, 4)

    npt.assert_array_equal(nodes_xyz[0], [10.0, 20.0, 5.0])
    npt.assert_array_almost_equal(colors[0], [1.0, 0.0, 0.0, 1.0])

    npt.assert_array_equal(nodes_xyz[1], [-5.0, 0.0, 0.0])
    npt.assert_array_almost_equal(colors[1], [0.0, 1.0, 0.0, 0.5])

    npt.assert_array_equal(edges_indices[0], [0, 1])


def test_gexf_parse_invalid_xml():
    invalid_xml = "<gexf><unclosed_tag>"
    with pytest.raises(ValueError, match="Invalid GEXF XML Data"):
        parse_network(invalid_xml, "gexf")


def test_gexf_parse_missing_graph_tag():
    xml = "<gexf><meta></meta></gexf>"
    with pytest.raises(ValueError, match="No <graph> tag found in GEXF"):
        parse_network(xml, "gexf")


def test_gexf_roundtrip(sample_gexf_data):
    original_data = parse_network(sample_gexf_data, "gexf")
    gexf_str = stringify_network(original_data, "gexf")
    final_data = parse_network(gexf_str, "gexf")

    npt.assert_array_almost_equal(original_data[0], final_data[0])
    npt.assert_array_equal(original_data[1], final_data[1])
    npt.assert_array_almost_equal(original_data[2], final_data[2])


def test_gml_parse_valid(sample_gml_data):
    nodes_xyz, edges_indices, colors = parse_network(sample_gml_data, "gml")

    assert nodes_xyz.shape == (2, 3)
    assert edges_indices.shape == (1, 2)
    assert colors.shape == (2, 4)

    npt.assert_array_equal(nodes_xyz[0], [15.0, 25.0, 0.0])
    npt.assert_array_almost_equal(colors[0], [0.5, 0.5, 0.5, 1.0])

    npt.assert_array_equal(edges_indices[0], [0, 1])


def test_gml_parse_missing_graph():
    gml = "node [ id 1 ]"
    with pytest.raises(ValueError, match="GML must contain a 'graph' key"):
        parse_network(gml, "gml")


def test_gml_roundtrip(sample_gml_data):
    original_data = parse_network(sample_gml_data, "gml")
    gml_str = stringify_network(original_data, "gml")
    final_data = parse_network(gml_str, "gml")

    npt.assert_array_almost_equal(original_data[0], final_data[0])
    npt.assert_array_equal(original_data[1], final_data[1])
    npt.assert_array_almost_equal(original_data[2], final_data[2])


def test_xnet_parse_valid(sample_xnet_data):
    nodes_xyz, edges_indices, colors = parse_network(sample_xnet_data, "xnet")

    assert nodes_xyz.shape == (2, 3)
    assert edges_indices.shape == (1, 2)
    assert colors.shape == (2, 4)

    npt.assert_array_equal(nodes_xyz[0], [10.0, 20.0, 30.0])
    npt.assert_array_almost_equal(colors[1], [0.0, 1.0, 0.0, 0.5])

    npt.assert_array_equal(edges_indices[0], [0, 1])


def test_xnet_parse_malformed_headers():
    data = "#invalid_header\n..."
    with pytest.raises(ValueError, match="Malformed XNET: Missing #vertices header"):
        parse_network(data, "xnet")


def test_xnet_roundtrip(sample_xnet_data):
    original_data = parse_network(sample_xnet_data, "xnet")
    xnet_str = stringify_network(original_data, "xnet")
    final_data = parse_network(xnet_str, "xnet")

    npt.assert_array_almost_equal(original_data[0], final_data[0])
    npt.assert_array_equal(original_data[1], final_data[1])
    npt.assert_array_almost_equal(original_data[2], final_data[2])


def test_parse_network_invalid_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        parse_network("data", "invalid_fmt")


def test_stringify_network_invalid_format():
    dummy_data = (np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 4)))
    with pytest.raises(ValueError, match="Unsupported format"):
        stringify_network(dummy_data, "invalid_fmt")


def test_empty_network_roundtrip():
    empty_data = (
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 2), dtype=np.int32),
        np.zeros((0, 4), dtype=np.float32),
    )

    for fmt in ["gexf", "gml", "xnet"]:
        s = stringify_network(empty_data, fmt)
        res = parse_network(s, fmt)

        assert res[0].shape == (0, 3)
        assert res[1].shape == (0, 2)
        assert res[2].shape == (0, 4)
