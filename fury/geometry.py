from pygfx import Geometry, Mesh


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
    mesh = Mesh(geometry=geometry, material=material)
    return mesh
