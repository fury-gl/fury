from fury.lib import (
    Geometry,
    Mesh,
    MeshBasicMaterial,
    MeshPhongMaterial,
    Points,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsMaterial,
)


def buffer_to_geometry(positions, **kwargs):
    """
    Convert a buffer to a geometry object.

    Parameters
    ----------
    positions : array_like
        The positions buffer.
    kwargs : dict
        A dict of attributes to define on the geometry object. Keys can be
        "colors", "normals", "texcoords",
        "indices", ...

    Returns
    -------
    geo : Geometry
        The geometry object.
    """
    if positions is None or positions.size == 0:
        raise ValueError("positions array cannot be empty or None.")

    geo = Geometry(positions=positions, **kwargs)
    return geo


def create_mesh(geometry, material):
    """
    Create a mesh object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object.

    Returns
    -------
    mesh : Mesh
        The mesh object.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be an instance of Geometry.")

    if not isinstance(material, (MeshPhongMaterial, MeshBasicMaterial)):
        raise TypeError(
            "material must be an instance of MeshPhongMaterial or MeshBasicMaterial."
        )

    mesh = Mesh(geometry=geometry, material=material)
    return mesh


def create_point(geometry, material):
    """
    Create a point object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object.

    Returns
    -------
    points : Points
        The point object.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be an instance of Geometry.")

    if not isinstance(
        material, (PointsMaterial, PointsGaussianBlobMaterial, PointsMarkerMaterial)
    ):
        raise TypeError(
            "material must be an instance of PointsMaterial, "
            "PointsGaussianBlobMaterial or PointsMarkerMaterial."
        )

    point = Points(geometry=geometry, material=material)
    return point
