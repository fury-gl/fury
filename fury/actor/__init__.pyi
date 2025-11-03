"""Actor module for FURY."""

__all__ = [
    "actor_from_primitive",
    "line",
    "sphere",
    "ellipsoid",
    "cylinder",
    "cone",
    "arrow",
    "axes",
    "box",
    "tetrahedron",
    "icosahedron",
    "triangularprism",
    "pentagonalprism",
    "octagonalprism",
    "rhombicuboctahedron",
    "frustum",
    "superquadric",
    "streamtube",
    "disk",
    "image",
    "marker",
    "point",
    "ring",
    "square",
    "star",
    "text",
    "triangle",
    "billboard",
    "sph_glyph",
    "vector_field",
    "vector_field_slicer",
    "surface",
    "SphGlyph",
    "data_slicer",
    "VectorField",
    "volume_slicer",
    "peaks_slicer",
    "streamlines",
    "line_projection",
]

from .billboard import billboard
from .bio import peaks_slicer, volume_slicer
from .core import actor_from_primitive, arrow, axes, line
from .curved import cone, cylinder, ellipsoid, sphere, streamlines, streamtube
from .planar import (
    disk,
    image,
    line_projection,
    marker,
    point,
    ring,
    square,
    star,
    text,
    triangle,
)
from .polyhedron import (
    box,
    frustum,
    icosahedron,
    octagonalprism,
    pentagonalprism,
    rhombicuboctahedron,
    superquadric,
    tetrahedron,
    triangularprism,
)
from .slicer import (
    SphGlyph,
    VectorField,
    data_slicer,
    sph_glyph,
    vector_field,
    vector_field_slicer,
)
from .surface import surface
