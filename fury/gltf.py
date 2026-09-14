"""Reading of glTF 2.0 assets."""

from fury.actor import Line, Mesh, Points, SkinnedMesh
from fury.lib import gfx
from fury.optpkg import TripWireError, optional_package

gltf_msg = (
    "You do not have gltflib installed. glTF files cannot be read without it. "
    "Please install or upgrade gltflib using pip install -U fury[gltf]"
)
_, have_gltflib, _ = optional_package("gltflib", trip_msg=gltf_msg)

# Drawables the PyGfx glTF importer builds, mapped to their FURY counterpart.
_ACTOR_TYPES = {
    gfx.Mesh: Mesh,
    gfx.SkinnedMesh: SkinnedMesh,
    gfx.Points: Points,
    gfx.Line: Line,
}


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
    Retype a PyGfx drawable as a FURY actor.

    Each FURY actor class subclasses its PyGfx counterpart without adding
    state, so rebinding the class exposes the FURY actor methods on the object
    the importer built, leaving it usable as the same node of the imported
    scene graph.

    Parameters
    ----------
    obj : gfx.WorldObject
        Drawable produced by the PyGfx glTF importer.

    Returns
    -------
    Actor
        The same object, retyped as the matching FURY actor.
    """
    obj.__class__ = _ACTOR_TYPES[type(obj)]
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


class glTF:
    """
    Reader for the contents of a glTF 2.0 asset.

    Parameters
    ----------
    filename : str
        Path of the ``.gltf`` or ``.glb`` file.
    quiet : bool, optional
        Whether to suppress the warnings raised for unsupported glTF features.
    remote_ok : bool, optional
        Whether ``filename`` is allowed to be a URL.

    Attributes
    ----------
    scene : gfx.Group
        Default scene of the asset.
    scenes : list of gfx.Group
        Every scene the asset declares.
    cameras : list of gfx.Camera
        Cameras the asset declares.
    lights : list of gfx.Light
        Punctual lights the asset declares.
    """

    def __init__(self, filename, *, quiet=True, remote_ok=False):
        """Read a glTF 2.0 asset."""
        self._gltf = load_gltf(filename, quiet=quiet, remote_ok=remote_ok)
        self._actors = None

    @property
    def scene(self):
        """
        Get the default scene of the asset.

        Returns
        -------
        gfx.Group
            Scene the asset nominates as its default.
        """
        return self._gltf.scene

    @property
    def scenes(self):
        """
        Get every scene the asset declares.

        Returns
        -------
        list of gfx.Group
            Scenes found in the asset.
        """
        return self._gltf.scenes

    @property
    def cameras(self):
        """
        Get the cameras the asset declares.

        Returns
        -------
        list of gfx.Camera
            Cameras found in the asset, positioned by their node transform.
        """
        return self._gltf.cameras

    @property
    def lights(self):
        """
        Get the punctual lights the asset declares.

        Returns
        -------
        list of gfx.Light
            Lights found in the asset, positioned by their node transform.
        """
        return self._gltf.lights

    def actors(self):
        """
        Get the drawables of the default scene as FURY actors.

        The actors are the nodes of :attr:`scene` itself, so their node
        transforms are already applied and moving an actor moves the scene
        with it.

        Returns
        -------
        list of Actor
            Mesh, skinned mesh, point and line actors of the default scene.
        """
        if self._actors is None:
            self._actors = [
                _to_actor(obj)
                for obj in self.scene.iter(lambda x: type(x) in _ACTOR_TYPES)
            ]

        return self._actors
