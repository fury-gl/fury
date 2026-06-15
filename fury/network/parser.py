"""Network Parsing Functionality."""

import re
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np


class BaseParser:
    """Abstract base class for all network file format parsers."""

    def parse(self, data):
        """
        Parse raw content string into a Network instance.

        Parameters
        ----------
        data : str
            The raw text string sequence content to parse.

        Returns
        -------
        tuple
            A tuple containing:
            - nodes_xyz (np.ndarray): Shape (N, 3) float32 array of node positions.
            - edges_indices (np.ndarray): Shape (E, 2) int32 array of edge connections.
            - colors (np.ndarray): Shape (N, 4) float32 array of node colors (RGBA).
        """
        raise NotImplementedError("Parse method not implemented")

    def stringify(self, parsed_data):
        """
        Serialize a parsed network data into a formatted string layout.

        Parameters
        ----------
        parsed_data : tuple
            The output tuple (nodes_xyz, edges_indices, colors) from self.parse().

        Returns
        -------
        str
            The formatted exported string file layout representation.
        """
        raise NotImplementedError("Stringify method not implemented")


class GEXFParser(BaseParser):
    """Parses and Writes GEXF XML format natively into NumPy arrays."""

    def parse(self, xml_string):
        """
        Parse raw GEXF XML directly into NumPy arrays.

        Parameters
        ----------
        xml_string : str
            The raw GEXF XML text context.

        Returns
        -------
        tuple
            A tuple containing:
            - nodes_xyz (np.ndarray): Shape (N, 3) float32 array of node positions.
            - edges_indices (np.ndarray): Shape (E, 2) int32 array of edge connections.
            - colors (np.ndarray): Shape (N, 4) float32 array of node colors (RGBA).
        """
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            raise ValueError("Invalid GEXF XML Data") from e

        def get_tag_name(element):
            """
            Strip the XML namespace prefix URI away from an element tag identifier.

            Parameters
            ----------
            element : Element
                The element tree layout node object instance.

            Returns
            -------
            str
                The clean target literal tag dictionary key name token string.
            """
            return element.tag.split("}")[-1] if "}" in element.tag else element.tag

        graph_node = None
        for el in root.iter():
            if get_tag_name(el).lower() == "graph":
                graph_node = el
                break

        if graph_node is None:
            raise ValueError("No <graph> tag found in GEXF")

        nodes_node = None
        for child in graph_node:
            if get_tag_name(child).lower() == "nodes":
                nodes_node = child
                break

        node_list = []
        if nodes_node is not None:
            node_list = [n for n in nodes_node if get_tag_name(n).lower() == "node"]

        n_nodes = len(node_list)
        nodes_xyz = np.zeros((n_nodes, 3), dtype=np.float32)
        colors = np.ones((n_nodes, 4), dtype=np.float32)

        id_to_idx = {}

        for i, n in enumerate(node_list):
            n_id = n.get("id")
            if n_id is None:
                n_id = str(i)
            id_to_idx[n_id] = i

            has_pos = False

            for child in n:
                tag = get_tag_name(child).lower()

                if tag == "color":
                    r, g, b = child.get("r"), child.get("g"), child.get("b")
                    a = child.get("a")

                    if r is not None and g is not None and b is not None:
                        colors[i, 0] = float(r) / 255.0
                        colors[i, 1] = float(g) / 255.0
                        colors[i, 2] = float(b) / 255.0
                        if a is not None:
                            colors[i, 3] = float(a)

                elif tag == "position":
                    x, y, z = child.get("x"), child.get("y"), child.get("z")
                    nodes_xyz[i, 0] = float(x) if x is not None else 0.0
                    nodes_xyz[i, 1] = float(y) if y is not None else 0.0
                    nodes_xyz[i, 2] = float(z) if z is not None else 0.0
                    has_pos = True

            if not has_pos:
                nodes_xyz[i] = (np.random.rand(3) - 0.5) * 100

        edges_node = None
        for child in graph_node:
            if get_tag_name(child).lower() == "edges":
                edges_node = child
                break

        edges_list = []
        if edges_node is not None:
            for edj in edges_node:
                if get_tag_name(edj).lower() == "edge":
                    source = edj.get("source")
                    target = edj.get("target")

                    if source in id_to_idx and target in id_to_idx:
                        edges_list.append([id_to_idx[source], id_to_idx[target]])

        edges_indices = np.array(edges_list, dtype=np.int32)
        if edges_indices.ndim == 1 and len(edges_indices) == 0:
            edges_indices = np.zeros((0, 2), dtype=np.int32)

        return nodes_xyz, edges_indices, colors

    def stringify(self, parsed_data):
        """
        Serialize extracted arrays back into a valid GEXF XML string format.

        This makes `stringify(parse(gexf_file))` output a validated equivalent.

        Parameters
        ----------
        parsed_data : tuple
            The output tuple (nodes_xyz, edges_indices, colors) from self.parse().

        Returns
        -------
        str
            The exported XML GEXF string context.
        """
        if not isinstance(parsed_data, tuple) or len(parsed_data) != 3:
            raise ValueError(
                "stringify expects the tuple emitted by parse(): (nodes, edges, colors)"
            )

        nodes_xyz, edges_indices, colors = parsed_data

        gexf = ET.Element(
            "gexf",
            {
                "xmlns": "http://www.gexf.net/1.2draft",
                "xmlns:viz": "http://www.gexf.net/1.2draft/viz",
                "version": "1.2",
            },
        )

        graph = ET.SubElement(
            gexf,
            "graph",
            {"mode": "static", "defaultedgetype": "undirected"},
        )

        nodes_el = ET.SubElement(graph, "nodes")
        for i in range(len(nodes_xyz)):
            n_id = str(i)
            n_el = ET.SubElement(nodes_el, "node", {"id": n_id, "label": f"Node {i}"})

            x, y, z = nodes_xyz[i]
            r, g, b, a = colors[i]

            ET.SubElement(
                n_el,
                "viz:color",
                {
                    "r": str(int(max(0, min(255, r * 255)))),
                    "g": str(int(max(0, min(255, g * 255)))),
                    "b": str(int(max(0, min(255, b * 255)))),
                    "a": str(round(a, 2)),
                },
            )

            ET.SubElement(
                n_el,
                "viz:position",
                {
                    "x": f"{x:g}",
                    "y": f"{y:g}",
                    "z": f"{z:g}",
                },
            )

        edges_el = ET.SubElement(graph, "edges")
        for i, (source, target) in enumerate(edges_indices):
            ET.SubElement(
                edges_el,
                "edge",
                {
                    "id": str(i),
                    "source": str(source),
                    "target": str(target),
                },
            )

        raw_str = ET.tostring(gexf, encoding="utf-8")
        return minidom.parseString(raw_str).toprettyxml(indent="  ")


class GMLParser(BaseParser):
    """Parses and Writes GML (Graph Modeling Language) efficiently."""

    _TOKEN_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|[\[\]]|[^\s\[\]]+')

    def parse(self, data):
        """
        Parse raw GML structured content tokens directly into NumPy arrays.

        Parameters
        ----------
        data : str
            The raw text string block processing context data content.

        Returns
        -------
        tuple
            A tuple containing:
            - nodes_xyz (np.ndarray): Shape (N, 3) float32 array of node positions.
            - edges_indices (np.ndarray): Shape (E, 2) int32 array of edge connections.
            - colors (np.ndarray): Shape (N, 4) float32 array of node colors (RGBA).
        """

        token_iter = (m.group(0) for m in self._TOKEN_PATTERN.finditer(data))

        def parse_gml_level(iterator):
            """
            Parse a nested level of GML tokens recursively into a dictionary object.

            Parameters
            ----------
            iterator : iterator
                The active iterator streaming raw GML string tokens.

            Returns
            -------
            dict
                The parsed property mapping tree for the current structural level.
            """
            obj = {}
            for key in iterator:
                if key == "]":
                    return obj

                try:
                    value_token = next(iterator)
                except StopIteration:
                    break

                if value_token == "[":
                    val = parse_gml_level(iterator)
                elif value_token.startswith('"'):
                    val = value_token[1:-1]
                else:
                    try:
                        val = int(value_token)
                    except ValueError:
                        try:
                            val = float(value_token)
                        except ValueError:
                            val = value_token

                if key in obj:
                    if isinstance(obj[key], list):
                        obj[key].append(val)
                    else:
                        obj[key] = [obj[key], val]
                else:
                    obj[key] = val
            return obj

        parsed_root = parse_gml_level(token_iter)

        if "graph" not in parsed_root:
            raise ValueError("GML must contain a 'graph' key")

        g_data = parsed_root["graph"]
        if isinstance(g_data, list):
            g_data = g_data[0]

        nodes = g_data.get("node", [])
        if not isinstance(nodes, list):
            nodes = [nodes]

        n_nodes = len(nodes)
        nodes_xyz = np.zeros((n_nodes, 3), dtype=np.float32)
        colors = np.ones((n_nodes, 4), dtype=np.float32)

        id_to_idx = {}

        for i, n in enumerate(nodes):
            nid = str(n.get("id", ""))
            id_to_idx[nid] = i

            graphics = n.get("graphics", {})

            if "x" in graphics and "y" in graphics:
                nodes_xyz[i, 0] = float(graphics.get("x", 0.0))
                nodes_xyz[i, 1] = float(graphics.get("y", 0.0))
                nodes_xyz[i, 2] = float(graphics.get("z", 0.0))
            else:
                nodes_xyz[i] = (np.random.rand(3) - 0.5) * 100

            c = graphics.get("color")
            if c is not None:
                if not isinstance(c, list):
                    c = [c]
                for j in range(min(len(c), 4)):
                    try:
                        colors[i, j] = float(c[j])
                    except (ValueError, TypeError):
                        pass

        edges = g_data.get("edge", [])
        if not isinstance(edges, list):
            edges = [edges]

        edges_list = []
        for e in edges:
            sid = str(e.get("source", ""))
            tid = str(e.get("target", ""))

            if sid in id_to_idx and tid in id_to_idx:
                edges_list.append([id_to_idx[sid], id_to_idx[tid]])

        edges_indices = np.array(edges_list, dtype=np.int32)
        if edges_indices.ndim == 1 and len(edges_indices) == 0:
            edges_indices = np.zeros((0, 2), dtype=np.int32)

        return nodes_xyz, edges_indices, colors

    def stringify(self, parsed_data):
        """
        Convert extracted arrays back into an explicit GML structural text layout.

        This accepts the exact output of `parse()`, effectively making:
        `stringify(parse(gml_file))` output a cleanly validated GML equivalent.

        Parameters
        ----------
        parsed_data : tuple
            The output tuple (nodes_xyz, edges_indices, colors) from self.parse().

        Returns
        -------
        str
            The exported plain GML string representation document block.
        """
        if not isinstance(parsed_data, tuple) or len(parsed_data) != 3:
            raise ValueError(
                "stringify expects the tuple emitted by parse(): (nodes, edges, colors)"
            )

        nodes_xyz, edges_indices, colors = parsed_data

        lines = ["graph ["]
        indent = "  "

        for i in range(len(nodes_xyz)):
            x, y, z = nodes_xyz[i]
            r, g, b, a = colors[i]

            lines.append(f"{indent}node [")
            lines.append(f"{indent}  id {i}")
            lines.append(f"{indent}  graphics [")

            lines.append(f"{indent}    x {x:g}")
            lines.append(f"{indent}    y {y:g}")
            lines.append(f"{indent}    z {z:g}")

            lines.append(f"{indent}    color {r:g}")
            lines.append(f"{indent}    color {g:g}")
            lines.append(f"{indent}    color {b:g}")
            lines.append(f"{indent}    color {a:g}")

            lines.append(f"{indent}  ]")
            lines.append(f"{indent}]")

        for source, target in edges_indices:
            lines.append(f"{indent}edge [")
            lines.append(f"{indent}  source {source}")
            lines.append(f"{indent}  target {target}")
            lines.append(f"{indent}]")

        lines.append("]")
        return "\n".join(lines)


class XNETParser(BaseParser):
    """Parses and Writes XNET format (Line-based format) directly into NumPy arrays."""

    def parse(self, data):
        """
        Parse raw XNET format tokens directly into NumPy arrays.

        Parameters
        ----------
        data : str
            The raw text string block processing context data content.

        Returns
        -------
        tuple
            A tuple containing:
            - nodes_xyz (np.ndarray): Shape (N, 3) float32 array of node positions.
            - edges_indices (np.ndarray): Shape (E, 2) int32 array of edge connections.
            - colors (np.ndarray): Shape (N, 4) float32 array of node colors (RGBA).
        """

        lines = [line.strip() for line in data.splitlines() if line.strip()]
        if not lines:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0, 4), dtype=np.float32),
            )

        idx = 0

        if not lines[idx].lower().startswith("#vertices"):
            raise ValueError("Malformed XNET: Missing #vertices header")

        parts = lines[idx].split()
        n_nodes = int(parts[1]) if len(parts) > 1 else 0
        idx += 1

        labels = []
        while idx < len(lines) and not lines[idx].startswith("#"):
            labels.append(lines[idx])
            idx += 1

        if n_nodes == 0:
            n_nodes = len(labels)

        nodes_xyz = (np.random.rand(n_nodes, 3).astype(np.float32) - 0.5) * 100
        colors = np.ones((n_nodes, 4), dtype=np.float32)
        edges_list = []

        if idx < len(lines) and lines[idx].lower().startswith("#edges"):
            idx += 1
            while idx < len(lines) and not lines[idx].startswith("#"):
                e_parts = lines[idx].split()
                if len(e_parts) >= 2:
                    try:
                        edges_list.append([int(e_parts[0]), int(e_parts[1])])
                    except ValueError:
                        pass
                idx += 1

        edges_indices = np.array(edges_list, dtype=np.int32)
        if edges_indices.ndim == 1:
            edges_indices = np.zeros((0, 2), dtype=np.int32)

        while idx < len(lines):
            line = lines[idx]
            match = re.match(r'^#([ve])\s+"([^"]+)"\s+(.+)$', line)
            idx += 1

            if not match:
                continue

            prop_type, prop_name, prop_fmt = match.groups()
            prop_name = prop_name.lower()

            values = []
            while idx < len(lines) and not lines[idx].startswith("#"):
                values.append(lines[idx])
                idx += 1

            if prop_type.lower() == "v":
                if prop_name == "position":
                    for i, val in enumerate(values):
                        if i >= n_nodes:
                            break
                        v_parts = val.replace('"', "").split()
                        try:
                            if len(v_parts) >= 2:
                                nodes_xyz[i, 0] = float(v_parts[0])
                                nodes_xyz[i, 1] = float(v_parts[1])
                            if len(v_parts) >= 3:
                                nodes_xyz[i, 2] = float(v_parts[2])
                        except ValueError:
                            pass

                elif prop_name == "color":
                    for i, val in enumerate(values):
                        if i >= n_nodes:
                            break
                        v_parts = val.replace('"', "").split()
                        try:
                            if len(v_parts) >= 3:
                                colors[i, 0] = float(v_parts[0])
                                colors[i, 1] = float(v_parts[1])
                                colors[i, 2] = float(v_parts[2])
                            if len(v_parts) >= 4:
                                colors[i, 3] = float(v_parts[3])
                        except ValueError:
                            pass

        return nodes_xyz, edges_indices, colors

    def stringify(self, parsed_data):
        """
        Serialize extracted arrays back into a valid structural XNET format.

        This accepts the exact output of `parse()`, effectively making:
        `stringify(parse(xnet_file))` output a cleanly validated XNET equivalent.

        Parameters
        ----------
        parsed_data : tuple
            The output tuple (nodes_xyz, edges_indices, colors) from self.parse().

        Returns
        -------
        str
            The exported plain XNET string document mapping.
        """
        if not isinstance(parsed_data, tuple) or len(parsed_data) != 3:
            raise ValueError(
                "stringify expects the tuple emitted by parse(): (nodes, edges, colors)"
            )

        nodes_xyz, edges_indices, colors = parsed_data
        n_nodes = len(nodes_xyz)

        lines = [f"#vertices {n_nodes}"]

        lines.extend([f'"{i}"' for i in range(n_nodes)])

        lines.append("#edges nonweighted undirected")
        lines.extend([f"{src} {tgt}" for src, tgt in edges_indices])

        lines.append('#v "position" v3')
        for x, y, z in nodes_xyz:
            lines.append(f"{x:g} {y:g} {z:g}")

        lines.append('#v "color" v4')
        for r, g, b, a in colors:
            lines.append(f"{r:g} {g:g} {b:g} {a:g}")

        return "\n".join(lines)


_parsers = {"gexf": GEXFParser(), "gml": GMLParser(), "xnet": XNETParser()}


def parse_network(data, format):
    """
    Parse string data into network arrays.

    Parameters
    ----------
    data : str
        The raw data containing network details.
    format : str
        The file layout format ('gexf', 'gml', or 'xnet').

    Returns
    -------
    tuple
        A tuple containing:
        - nodes_xyz (np.ndarray): Shape (N, 3) float32 array of node positions.
        - edges_indices (np.ndarray): Shape (E, 2) int32 array of edge connections.
        - colors (np.ndarray): Shape (N, 4) float32 array of node colors (RGBA).
    """
    fmt = format.lower().strip()
    if fmt not in _parsers:
        raise ValueError(
            f"Unsupported format: {fmt}. Supported: {list(_parsers.keys())}"
        )

    return _parsers[fmt].parse(data)


def stringify_network(network_data, format):
    """
    Convert network arrays into a string of the specified format.

    Parameters
    ----------
    network_data : tuple
        A tuple containing:
        - nodes_xyz (np.ndarray): Shape (N, 3) float32 array of node positions.
        - edges_indices (np.ndarray): Shape (E, 2) int32 array of edge connections.
        - colors (np.ndarray): Shape (N, 4) float32 array of node colors (RGBA).
    format : str
        The exported configuration selector string ('gexf', 'gml', or 'xnet').

    Returns
    -------
    str
        The formatted serial layout text output stream.
    """
    fmt = format.lower().strip()
    if fmt not in _parsers:
        raise ValueError(
            f"Unsupported format: {fmt}. Supported: {list(_parsers.keys())}"
        )

    return _parsers[fmt].stringify(network_data)
