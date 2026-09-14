"""Reading of glTF 2.0 assets."""

from fury.actor import Mesh
from fury.lib import gfx
from fury.optpkg import TripWireError, optional_package

gltf_msg = (
    "You do not have gltflib installed. glTF files cannot be read without it. "
    "Please install or upgrade gltflib using pip install -U fury[gltf]"
)
_, have_gltflib, _ = optional_package("gltflib", trip_msg=gltf_msg)


def _check_gltflib():
    """
    Check that the optional ``gltflib`` dependency is importable.

    Raises
    ------
    TripWireError
        If ``gltflib`` is not installed.
    """
    if not have_gltflib:
        raise TripWireError(gltf_msg)


def _to_actor(obj):
    """
    Retype a PyGfx mesh as a FURY actor.

    ``Mesh`` subclasses ``gfx.Mesh`` without adding state, so rebinding the
    class exposes the FURY actor methods on the object the importer built,
    leaving it usable as the same node of the imported scene graph.

    Parameters
    ----------
    obj : gfx.Mesh
        Mesh produced by the PyGfx glTF importer.

    Returns
    -------
    Mesh
        The same object, retyped as a FURY mesh actor.
    """
    obj.__class__ = Mesh
    return obj


def load_gltf(fname, *, quiet=True, remote_ok=False):
    """
    Read a glTF 2.0 asset.

    Parameters
    ----------
    fname : str
        Path of the ``.gltf`` or ``.glb`` file.
    quiet : bool, optional
        Whether to suppress the warnings raised for unsupported glTF features.
    remote_ok : bool, optional
        Whether ``fname`` is allowed to be a URL.

    Returns
    -------
    object
        Imported asset, exposing the ``scene``, ``scenes``, ``cameras``,
        ``lights`` and ``animations`` it declares.

    Raises
    ------
    TripWireError
        If ``gltflib`` is not installed.
    """
    _check_gltflib()

    return gfx.load_gltf(fname, quiet=quiet, remote_ok=remote_ok)


def load_gltf_mesh(fname, *, materials=True, quiet=True, remote_ok=False):
    """
    Read the meshes of a glTF 2.0 asset.

    Node transformations are not applied and skeletons are not read. Use
    :func:`load_gltf` to read a positioned scene.

    Parameters
    ----------
    fname : str
        Path of the ``.gltf`` or ``.glb`` file.
    materials : bool, optional
        Whether to read the materials the meshes refer to.
    quiet : bool, optional
        Whether to suppress the warnings raised for unsupported glTF features.
    remote_ok : bool, optional
        Whether ``fname`` is allowed to be a URL.

    Returns
    -------
    list of Mesh
        Mesh actors found in the asset.

    Raises
    ------
    TripWireError
        If ``gltflib`` is not installed.
    """
    _check_gltflib()

    meshes = gfx.load_gltf_mesh(
        fname, materials=materials, quiet=quiet, remote_ok=remote_ok
    )

    return [_to_actor(mesh) for mesh in meshes]
