"""Core actor functionality for FURY."""

import inspect

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from fury.actor import set_opacity
from fury.colormap import normalize_colors
from fury.geometry import (
    buffer_to_geometry,
    line_buffer_separator,
)
from fury.io import load_image
from fury.lib import (
    Buffer,
    Geometry,
    ImageBasicMaterial,
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsMaterial,
    TextMaterial,
    Texture,
    gfx,
)
from fury.material import (
    PBRMaterial,
    _create_line_material,
    _create_mesh_material,
    _pbr_material_like,
    _set_pbr_properties,
)
import fury.primitive as fp
from fury.transform import rotate, scale, translate
from fury.utils import _anisotropy_tangents


class Actor:
    """Base Actor class for making APIs user-friendly."""

    def rotate(self, rotation):
        """
        Rotate the actor by the given rotation.

        Parameters
        ----------
        rotation : tuple
            Rotation angles (in degrees) around the x, y, and z axes.
        """
        rotation = np.asarray(rotation, dtype=np.float32)
        if rotation.shape != (3,):
            raise ValueError("Rotation must contain three angles (degrees).")

        quaternion = self._euler_to_quaternion(np.radians(rotation))
        rotate(quaternion, actor=self)

    def translate(self, translation):
        """
        Translate the actor by the given translation vector.

        Parameters
        ----------
        translation : tuple
            Translation vector along the x, y, and z axes.
        """
        if not isinstance(translation, (list, tuple, np.ndarray)):
            raise ValueError("Translation must be a sequence of three values.")
        translation = np.asarray(translation, dtype=np.float32)
        if translation.shape != (3,):
            raise ValueError("Translation must contain three values.")
        translate(translation, actor=self)

    def scale(self, scales):
        """
        Scale the actor by the given scale factors.

        Parameters
        ----------
        scales : tuple or float
            Scale factors along the x, y, and z axes. If a single float
            is provided, uniform scaling is applied.
        """
        if isinstance(scales, (int, float)):
            scales = (scales, scales, scales)
        elif not isinstance(scales, (list, tuple, np.ndarray)):
            raise ValueError(
                "Scale must be a sequence of three values or a single float."
            )

        scales = np.asarray(scales, dtype=np.float32)
        if scales.shape != (3,):
            raise ValueError("Scale must contain three values.")
        scale(scales, actor=self)

    def transform(self, matrix):
        """
        Apply a transformation matrix to the actor.

        This transformation replaces any existing transformations.

        Parameters
        ----------
        matrix : ndarray, shape (4, 4)
            Transformation matrix to be applied to the actor.
        """
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Transformation matrix must be a numpy array.")
        elif matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be of shape (4, 4).")
        self.local.matrix = matrix

    @property
    def opacity(self):
        """
        Get the opacity of the actor.

        Returns
        -------
        float
            Opacity value between 0 (fully transparent) and 1 (fully opaque).
        """
        if isinstance(self, Group):
            if len(self.children) == 0:
                return 1.0
            return self.children[0].material.opacity
        return self.material.opacity

    @opacity.setter
    def opacity(self, opacity):
        """
        Set the opacity of the actor.

        Parameters
        ----------
        opacity : float
            Opacity value between 0 (fully transparent) and 1 (fully opaque).
        """
        set_opacity(self, opacity)

    @staticmethod
    def _euler_to_quaternion(rotation):
        """
        Convert XYZ Euler angles (radians) to a quaternion.

        Parameters
        ----------
        rotation : tuple or ndarray
            Rotation angles (in radians) around the x, y, and z axes.

        Returns
        -------
        ndarray
            Quaternion representing the rotation.
        """
        return Rot.from_euler("xyz", rotation).as_quat()


class Mesh(gfx.Mesh, Actor):
    """Mesh actor class."""


class Points(gfx.Points, Actor):
    """Points actor class."""


class LineActor(gfx.Line, Actor):
    """Line actor class."""


Line = LineActor


class TextActor(gfx.Text, Actor):
    """Text actor class."""


Text = TextActor


class ImageActor(gfx.Image, Actor):
    """Image actor class."""


Image = ImageActor


class Volume(gfx.Volume, Actor):
    """Volume actor class."""


class Group(gfx.Group, Actor):
    """Group actor class."""


def create_mesh(geometry, material):
    """
    Create a mesh object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object. Must be an instance of MeshPhongMaterial,
        MeshBasicMaterial or MeshStandardMaterial (which MeshPhysicalMaterial
        derives from).

    Returns
    -------
    Mesh
        The mesh object.

    Raises
    ------
    TypeError
        If geometry is not an instance of Geometry or material is not an
        instance of MeshPhongMaterial, MeshBasicMaterial or
        MeshStandardMaterial.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be an instance of Geometry.")

    if not isinstance(
        material, (MeshPhongMaterial, MeshBasicMaterial, MeshStandardMaterial)
    ):
        raise TypeError(
            "material must be an instance of MeshPhongMaterial, "
            "MeshBasicMaterial or MeshStandardMaterial."
        )

    mesh = Mesh(geometry=geometry, material=material)
    return mesh


def create_line(geometry, material):
    """
    Create a line object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object.

    Returns
    -------
    Line
        The line object.
    """
    line = Line(geometry=geometry, material=material)
    return line


def create_point(geometry, material):
    """
    Create a point object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object. Must be either PointsMaterial, PointsGaussianBlobMaterial,
        or PointsMarkerMaterial.

    Returns
    -------
    Points
        The point object.

    Raises
    ------
    TypeError
        If geometry is not an instance of Geometry or material is not an
        instance of PointsMaterial, PointsGaussianBlobMaterial, or PointsMarkerMaterial.
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


def create_text(text, material, **kwargs):
    """
    Create a text object.

    Parameters
    ----------
    text : str
        The text content.
    material : TextMaterial
        The material object.
    **kwargs : dict
        Additional properties like font_size, anchor, etc.

    Returns
    -------
    Text
        The text object.

    Raises
    ------
    TypeError
        If text is not a string or material is not an instance of TextMaterial.

    Examples
    --------
    >>> mat = TextMaterial()
    >>> t = create_text("Hello", mat)
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not isinstance(material, TextMaterial):
        raise TypeError("material must be an instance of TextMaterial.")

    return Text(text=text, material=material, **kwargs)


def create_image(image_input, material, **kwargs):
    """
    Create an image object.

    Parameters
    ----------
    image_input : str or np.ndarray
        The image content. Can be a file path (string) to an image file,
        or a NumPy array. Supported array shapes:
        - (H, W) for grayscale images
        - (H, W, 1) for single-channel images (squeezed to grayscale)
        - (H, W, 3) for RGB images (converted to RGBA internally)
        - (H, W, 4) for RGBA images
    material : ImageBasicMaterial
        The material object.
    **kwargs : dict, optional
        Additional properties like position, visible, etc.

    Returns
    -------
    Image
        The image object.

    Raises
    ------
    TypeError
        If image_input is not a string or NumPy array, or if material is
        not an instance of ImageBasicMaterial.
    ValueError
        If the array has an unsupported number of dimensions or channels.
    """
    if isinstance(image_input, str):
        image = np.flipud(np.array(load_image(image_input)).astype(np.float32))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim not in (2, 3):
            raise ValueError("image_input must be a 2D or 3D NumPy array.")
        if image_input.ndim == 3 and image_input.shape[2] not in (1, 3, 4):
            raise ValueError("image_input must have 1, 3, or 4 channels.")
        image = image_input.astype(np.float32)
    else:
        raise TypeError("image_input must be a file path (str) or a NumPy array.")

    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        alpha = np.ones((*image.shape[:2], 1), dtype=np.float32)
        image = np.concatenate([image, alpha], axis=-1)

    if not isinstance(material, ImageBasicMaterial):
        raise TypeError("material must be an instance of ImageBasicMaterial.")

    image = Image(
        Geometry(grid=Texture(image.astype(np.float32), dim=2)), material=material
    )
    return image


def actor_from_primitive(
    vertices,
    faces,
    centers,
    *,
    colors=(1, 0, 0),
    scales=(1, 1, 1),
    directions=(1, 0, 0),
    opacity=None,
    material="phong",
    smooth=False,
    enable_picking=True,
    repeat_primitive=True,
    have_tiled_verts=False,
    wireframe=False,
    wireframe_thickness=1.0,
    material_params=None,
):
    """
    Build an actor from a primitive.

    Parameters
    ----------
    vertices : ndarray
        Vertices of the primitive.
    faces : ndarray
        Faces of the primitive.
    centers : ndarray, shape (N, 3)
        Primitive positions.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA colors. Accepts values in [0, 255] (int), [0, 1] (float),
        or hex strings (e.g. "#FF0000"). Values above 1.0 are treated as
        [0, 255] and normalized internally.
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the primitive in each dimension. If a single value is provided,
        the same size will be used for all primitives.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the primitive.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the primitive. Options are 'phong', 'basic',
        'standard' and 'physical'. The last two are physically based (PBR)
        materials.
    smooth : bool, optional
        Whether to create a smooth primitive or a faceted primitive.
    enable_picking : bool, optional
        Whether the primitive should be pickable in a 3D scene.
    repeat_primitive : bool, optional
        Whether to repeat the primitive for each center. If False,
        only one instance of the primitive is created at the first center.
    have_tiled_verts : bool, optional
        If True, vertices are already tiled (one set per center) and should
        not be duplicated again inside ``repeat_primitive``.
    wireframe : bool, optional
        Whether to render the mesh as a wireframe.
    wireframe_thickness : float, optional
        The thickness of the wireframe lines.
    material_params : dict, optional
        Extra properties forwarded to the material, e.g.
        ``{'metalness': 1.0, 'roughness': 0.2}`` for the PBR materials. See
        :func:`fury.material._create_mesh_material` for the supported keys.

    Returns
    -------
    Actor
        A mesh actor containing the generated primitive, with the specified
        material and properties.
    """
    if repeat_primitive:
        colors = normalize_colors(colors, n_points=len(centers))
    else:
        colors = normalize_colors(colors)

    if repeat_primitive:
        res = fp.repeat_primitive(
            vertices,
            faces,
            centers,
            directions=directions,
            colors=colors,
            scales=scales,
            have_tiled_verts=have_tiled_verts,
        )
        big_vertices, big_faces, big_colors, _ = res

    else:
        big_vertices = vertices
        big_faces = faces
        big_colors = colors

    prim_count = len(centers)

    if isinstance(opacity, (int, float)):
        if big_colors.shape[1] == 3:
            big_colors = np.hstack(
                (big_colors, np.full((big_colors.shape[0], 1), opacity))
            )
        else:
            big_colors[:, 3] *= opacity

    is_transparent = False
    if isinstance(opacity, (int, float)) and opacity < 1.0:
        is_transparent = True
    elif big_colors.shape[1] == 4 and np.any(big_colors[:, 3] < 1.0):
        is_transparent = True

    geo = buffer_to_geometry(
        indices=big_faces.astype("int32"),
        positions=big_vertices.astype("float32"),
        texcoords=big_vertices.astype("float32"),
        colors=big_colors.astype("float32"),
    )

    mat = _create_mesh_material(
        material=material,
        enable_picking=enable_picking,
        flat_shading=not smooth,
        wireframe=wireframe,
        wireframe_thickness=wireframe_thickness,
        alpha_mode="weighted_blend" if is_transparent else "auto",
        depth_write=not is_transparent,
        **(material_params or {}),
    )
    obj = create_mesh(geometry=geo, material=mat)
    if not repeat_primitive:
        obj.local.position = centers[0]
    obj.prim_count = prim_count
    return obj


def manifest_pbr(
    actor,
    *,
    centers=None,
    metalness=None,
    roughness=None,
    anisotropy=None,
    anisotropy_rotation=None,
    clearcoat=None,
    clearcoat_roughness=None,
    ior=None,
    anisotropy_direction=(0.0, 1.0, 0.5),
    material_params=None,
    **actor_kwargs,
):
    """
    Give an existing mesh a PBR material, or build one PBR object per center.

    Parameters
    ----------
    actor : Mesh or callable
        Either an existing mesh actor whose material is converted to a
        :class:`fury.material.PBRMaterial`, or an actor factory such as
        :func:`fury.actor.sphere` called once for all ``centers``.
    centers : array_like, shape (N, 3), optional
        Object centers, with ``N >= 1``. Required for an actor factory,
        rejected for a mesh.
    metalness : float, optional
        Metalness shared by every object.
    roughness : float, optional
        Surface roughness shared by every object.
    anisotropy : float, optional
        Anisotropy strength shared by every object.
    anisotropy_rotation : float, optional
        Rotation of the anisotropy direction in radians.
    clearcoat : float, optional
        Clearcoat layer strength shared by every object.
    clearcoat_roughness : float, optional
        Roughness of the clearcoat layer.
    ior : float, optional
        Index of refraction.
    anisotropy_direction : array_like, shape (3,), optional
        Direction used to generate tangents. Each tangent is perpendicular to
        this direction and to the surface normal. Only used when tangents are
        generated, see Notes.
    material_params : dict or list of dict, optional
        Additional ``PBRMaterial`` properties such as ``sheen``,
        ``sheen_color``, ``iridescence``, ``emissive`` or ``env_map``. The named
        PBR arguments above are shared defaults and these entries override
        them. A dict applies to every object; for an actor factory a list holds
        one dict per center.
    **actor_kwargs : dict
        Keyword arguments forwarded to the actor factory, for example ``colors``
        or ``radii``.

    Returns
    -------
    PBRMaterial or Group
        The material of the converted mesh, or a group with one mesh per center,
        each owning its own material.

    Raises
    ------
    TypeError
        If ``actor`` is neither a mesh nor a suitable actor factory, the mesh
        material cannot be converted, ``centers`` or actor keywords are given
        with a mesh, ``material`` is passed to a factory, or the factory does
        not return a mesh with one primitive per center.
    ValueError
        If ``centers`` or ``material_params`` have the wrong shape, a property
        is unknown or read-only, or the factory output cannot be split into
        equal per-center meshes.

    Notes
    -----
    Arguments left as None keep the value already on the mesh material, so
    converting a standard or physical material preserves its other properties,
    including ``env_map`` and texture maps. For a new material they take the
    ``PBRMaterial`` default.

    Anisotropy needs tangents. When the geometry has none and the material has
    no ``normal_map`` or ``clearcoat_normal_map``, tangents are generated and,
    for an existing mesh, stored on its geometry, which may be shared with other
    meshes. They are skipped with a normal map because pygfx would then build
    the normal-map frame from the tangents instead of the UVs.

    Property names map from FURY 0.x: ``metallic`` -> ``metalness``,
    ``coat_strength`` -> ``clearcoat``, ``coat_roughness`` ->
    ``clearcoat_roughness`` and ``base_ior`` -> ``ior``. ``coat_ior`` has no
    pygfx equivalent. Call this before adding the actor to a ``Scene`` with a
    skybox so the scene assigns its ``env_map`` to the new material, or pass
    ``env_map`` in ``material_params``.

    Examples
    --------
    >>> import numpy as np
    >>> from fury import actor
    >>> box = actor.box(np.zeros((1, 3)))
    >>> material = actor.manifest_pbr(box, metalness=1.0, roughness=0.2)
    >>> centers = np.array([[0, 0, 0], [3, 0, 0]])
    >>> spheres = actor.manifest_pbr(
    ...     actor.sphere,
    ...     centers=centers,
    ...     material_params=[{"metalness": 0.0}, {"metalness": 1.0}],
    ... )
    """
    shared = {
        name: value
        for name, value in {
            "metalness": metalness,
            "roughness": roughness,
            "anisotropy": anisotropy,
            "anisotropy_rotation": anisotropy_rotation,
            "clearcoat": clearcoat,
            "clearcoat_roughness": clearcoat_roughness,
            "ior": ior,
        }.items()
        if value is not None
    }

    if isinstance(actor, gfx.Mesh):
        if centers is not None or actor_kwargs:
            raise TypeError(
                "centers and actor keyword arguments are only accepted when "
                "actor is an actor factory."
            )
        if material_params is not None and not isinstance(material_params, dict):
            raise ValueError(
                "material_params must be a dict when actor is an existing mesh."
            )
        params = {**shared, **(material_params or {})}

        material = actor.material
        if type(material) is PBRMaterial:
            _set_pbr_properties(material, params)
        elif type(material) in (
            MeshPhongMaterial,
            MeshBasicMaterial,
            MeshStandardMaterial,
            MeshPhysicalMaterial,
        ):
            material = _pbr_material_like(material, params)
            actor.material = material
        else:
            raise TypeError(
                f"manifest_pbr cannot convert {type(material).__name__}; "
                "only plain mesh materials are supported."
            )

        geometry = actor.geometry
        if (
            getattr(geometry, "tangents", None) is None
            and material.normal_map is None
            and material.clearcoat_normal_map is None
        ):
            normals = getattr(geometry, "normals", None)
            geometry.tangents = Buffer(
                _anisotropy_tangents(
                    geometry.positions.data,
                    geometry.indices.data.reshape(-1, 3),
                    None if normals is None else normals.data,
                    anisotropy_direction,
                )
            )
        return material

    if not callable(actor):
        raise TypeError(
            "actor must be a mesh actor or an actor factory such as fury.actor.sphere."
        )

    name = getattr(actor, "__name__", repr(actor))
    if centers is None:
        raise ValueError("centers is required when actor is an actor factory.")
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("centers must have shape (N, 3) with N >= 1.")
    n = len(centers)

    parameters = inspect.signature(actor).parameters
    if "material" not in parameters or "material_params" not in parameters:
        raise TypeError(f"{name} does not accept material and material_params.")
    if "material" in actor_kwargs:
        raise TypeError(
            "material is set by manifest_pbr; pass PBR properties through its "
            "arguments."
        )
    if "impostor" in parameters:
        actor_kwargs.setdefault("impostor", False)

    if material_params is None:
        entries = [{}] * n
    elif isinstance(material_params, dict):
        entries = [material_params] * n
    else:
        entries = list(material_params)
        if len(entries) != n:
            raise ValueError(
                "material_params must contain one dict per center: "
                f"expected {n}, got {len(entries)}."
            )
        if not all(isinstance(entry, dict) for entry in entries):
            raise ValueError("material_params entries must be dicts.")
    per_object = [{**shared, **entry} for entry in entries]

    template = actor(centers, material="physical", **actor_kwargs)
    if not isinstance(template, gfx.Mesh) or getattr(template, "prim_count", None) != n:
        raise TypeError(f"{name} did not return a mesh with one primitive per center.")

    geometry = template.geometry
    positions = geometry.positions.data
    faces = geometry.indices.data.reshape(-1, 3)
    if len(positions) % n or len(faces) % n:
        raise ValueError(
            f"{name} did not produce the same number of vertices and faces per center."
        )
    nv, nf = len(positions) // n, len(faces) // n
    per_center = faces.reshape(n, nf, 3)
    first = np.arange(n) * nv
    if (per_center.min(axis=(1, 2)) < first).any() or (
        per_center.max(axis=(1, 2)) >= first + nv
    ).any():
        raise ValueError(f"{name} did not produce equal per-center meshes.")

    arrays = {
        key: getattr(geometry, key).data
        for key in ("positions", "normals", "colors", "texcoords")
        if getattr(geometry, key, None) is not None
    }
    materials = [_pbr_material_like(template.material, params) for params in per_object]
    normal_mapped = [
        material.normal_map is not None or material.clearcoat_normal_map is not None
        for material in materials
    ]

    factory_tangents = getattr(geometry, "tangents", None)
    if factory_tangents is not None:
        tangents = factory_tangents.data
        skip_tangents = [False] * n
    else:
        tangents = None
        if not all(normal_mapped):
            tangents = _anisotropy_tangents(
                positions, faces, arrays.get("normals"), anisotropy_direction
            )
        skip_tangents = normal_mapped

    children = []
    for i, material in enumerate(materials):
        chunk = slice(i * nv, (i + 1) * nv)
        child_arrays = {key: data[chunk] for key, data in arrays.items()}
        if tangents is not None and not skip_tangents[i]:
            child_arrays["tangents"] = tangents[chunk]
        child = create_mesh(
            geometry=buffer_to_geometry(
                indices=faces[i * nf : (i + 1) * nf] - i * nv, **child_arrays
            ),
            material=material,
        )
        child.prim_count = 1
        children.append(child)

    group = Group()
    group.local.matrix = template.local.matrix
    group.add(*children)
    return group


def arrow(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    height=1.0,
    resolution=10,
    tip_length=0.35,
    tip_radius=0.1,
    shaft_radius=0.03,
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    material_params=None,
    enable_picking=True,
):
    """
    Create one or many arrows with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Arrow positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the arrow.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA colors. Accepts values in [0, 255] (int), [0, 1] (float),
        or hex strings (e.g. "#FF0000"). Values above 1.0 are treated as
        [0, 255] and normalized internally.
    height : float or ndarray, shape (N,), optional
        The total height of the arrow, including the shaft and tip. A single
        value applies to all arrows, while an array specifies a value per arrow.
    resolution : int, optional
        The number of divisions along the arrow's circular cross-sections.
        Higher values produce smoother arrows.
    tip_length : float or ndarray, shape (N,), optional
        The length of the arrowhead tip relative to the total height. A single
        value applies to all arrows, while an array specifies a value per arrow.
    tip_radius : float or ndarray, shape (N,), optional
        The radius of the arrowhead tip. A single value applies to all arrows,
        while an array specifies a value per arrow.
    shaft_radius : float or ndarray, shape (N,), optional
        The radius of the arrow shaft. A single value applies to all arrows,
        while an array specifies a value per arrow.
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the arrow in each dimension. If a single value is
        provided, the same size will be used for all arrows.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the arrows. Options are 'phong', 'basic',
        'standard' and 'physical'. The last two are physically based (PBR)
        materials.
    material_params : dict, optional
        Extra properties forwarded to the material, e.g.
        ``{'metalness': 1.0, 'roughness': 0.2}`` for the PBR materials. See
        :func:`fury.material._create_mesh_material` for the supported keys.
    enable_picking : bool, optional
        Whether the arrows should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated arrows, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> arrow_actor = actor.arrow(centers=centers, colors=colors)
    >>> _ = scene.add(arrow_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """

    n_centers = len(centers)
    height_arr = fp._normalize_geom_param(height, n_centers, param_name="height")
    tip_length_arr = fp._normalize_geom_param(
        tip_length, n_centers, param_name="tip_length"
    )
    tip_radius_arr = fp._normalize_geom_param(
        tip_radius, n_centers, param_name="tip_radius"
    )
    shaft_radius_arr = fp._normalize_geom_param(
        shaft_radius, n_centers, param_name="shaft_radius"
    )

    all_uniform = (
        np.all(height_arr == height_arr[0])
        and np.all(tip_length_arr == tip_length_arr[0])
        and np.all(tip_radius_arr == tip_radius_arr[0])
        and np.all(shaft_radius_arr == shaft_radius_arr[0])
    )

    if all_uniform:
        vertices, faces = fp.prim_arrow(
            height=height_arr[0],
            resolution=resolution,
            tip_length=tip_length_arr[0],
            tip_radius=tip_radius_arr[0],
            shaft_radius=shaft_radius_arr[0],
        )
        return actor_from_primitive(
            vertices,
            faces,
            centers=centers,
            colors=colors,
            scales=scales,
            directions=directions,
            opacity=opacity,
            material=material,
            material_params=material_params,
            enable_picking=enable_picking,
        )

    _, faces = fp.prim_arrow(
        height=height_arr[0],
        resolution=resolution,
        tip_length=tip_length_arr[0],
        tip_radius=tip_radius_arr[0],
        shaft_radius=shaft_radius_arr[0],
    )
    all_verts = [
        fp.prim_arrow(
            height=height_arr[i],
            resolution=resolution,
            tip_length=tip_length_arr[i],
            tip_radius=tip_radius_arr[i],
            shaft_radius=shaft_radius_arr[i],
        )[0]
        for i in range(n_centers)
    ]
    vertices = np.concatenate(all_verts)
    return actor_from_primitive(
        vertices,
        faces,
        centers=centers,
        colors=colors,
        scales=scales,
        directions=directions,
        opacity=opacity,
        material=material,
        material_params=material_params,
        enable_picking=enable_picking,
        have_tiled_verts=True,
    )


def axes(
    *,
    scale=(1.0, 1.0, 1.0),
    color_x=(1.0, 0.0, 0.0),
    color_y=(0.0, 1.0, 0.0),
    color_z=(0.0, 0.0, 1.0),
    opacity=1.0,
):
    """
    Create coordinate system axes using colored arrows.

    The axes are represented as arrows with different colors:
    red = X-axis, green = Y-axis, blue = Z-axis.

    Parameters
    ----------
    scale : tuple (3,), optional
        The size (length) of each axis in the x, y, and z directions.
    color_x : str, tuple, list or ndarray, optional
        Color for the X-axis. Accepts a hex string, RGB in [0, 1], or
        RGB in [0, 255].
    color_y : str, tuple, list or ndarray, optional
        Color for the Y-axis. Same formats as ``color_x``.
    color_z : str, tuple, list or ndarray, optional
        Color for the Z-axis. Same formats as ``color_x``.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    Actor
        An axes actor representing the coordinate axes with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> axes_actor = actor.axes()
    >>> _ = scene.add(axes_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    centers = np.zeros((3, 3))
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    color_x = normalize_colors(color_x)[0][:3]
    color_y = normalize_colors(color_y)[0][:3]
    color_z = normalize_colors(color_z)[0][:3]
    colors = np.array(
        [
            [*color_x, opacity],
            [*color_y, opacity],
            [*color_z, opacity],
        ]
    )
    scales = np.asarray(scale)

    obj = arrow(centers=centers, directions=directions, colors=colors, scales=scales)
    return obj


def create_axes_helper(
    *,
    labels=None,
    colors=None,
    thickness=2,
    center_disk_radius=0.11,
    endpoint_disk_radius=0.33,
    label_font_size=0.4,
):
    """
    Create actors composing a UI axes helper.

    This returns the helper group and related actor lists so callers can
    attach callbacks and place it in scene-specific coordinate systems.

    Parameters
    ----------
    labels : list of str, optional
        Labels for [-X, +X, -Y, +Y, -Z, +Z].
    colors : list, optional
        Color for each axis endpoint. Each entry accepts a hex string,
        RGB in [0, 1], or RGB in [0, 255].
    thickness : float, optional
        Thickness for endpoint lines.
    center_disk_radius : float, optional
        Radius of the center disk.
    endpoint_disk_radius : float, optional
        Radius of endpoint disks.
    label_font_size : float, optional
        Font size for endpoint labels.

    Returns
    -------
    dict
        A dictionary containing:
        - group
        - center_disk
        - disks
        - labels
        - lines
        - line_points
        - axis_vectors
    """
    from fury.actor.curved import streamlines
    from fury.actor.planar import disk, text

    if labels is None:
        labels = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]
    elif labels is not None and len(labels) != 6:
        raise ValueError(
            "labels must be a list of 6 strings for [-X, +X, -Y, +Y, -Z, +Z]."
        )

    if colors is None:
        colors = [
            (0.9, 0.3, 0.23),
            (0.9, 0.3, 0.23),
            (0.5, 0.7, 0),
            (0.5, 0.7, 0),
            (0, 0, 0.7),
            (0, 0, 0.7),
        ]
    elif colors is not None and len(colors) != 6:
        raise ValueError(
            "colors must be a list of 6 RGB tuples for [-X, +X, -Y, +Y, -Z, +Z]."
        )

    group = Group(name="Axes Helper")
    centers = [
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]

    center_disk = disk(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        radii=center_disk_radius,
        colors=(0.5, 0.5, 0.5),
        material="basic",
    )
    group.add(center_disk)

    disks = []
    labels_actors = []
    lines = []
    line_points = []

    for i, endpoint_center in enumerate(centers):
        disk_actor = disk(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            radii=endpoint_disk_radius,
            colors=colors[i],
            material="basic",
        )
        disk_actor.local.position = endpoint_center.tolist()

        label_actor = text(
            labels[i],
            position=endpoint_center.tolist(),
            font_size=label_font_size,
        )

        axis_dir = endpoint_center / np.linalg.norm(endpoint_center)
        line_start = (axis_dir * center_disk_radius).tolist()
        line_end = (endpoint_center - axis_dir * endpoint_disk_radius).tolist()
        line_actor = streamlines(
            [[line_start, line_end]],
            colors=[colors[i]],
            thickness=thickness,
            outline_thickness=0,
        )

        disks.append(disk_actor)
        labels_actors.append(label_actor)
        lines.append(line_actor)
        line_points.append((line_start, line_end))

        # Due to the convention of the z -ve is forward.
        axis_vectors = centers[:4] + centers[5:] + centers[4:5]

        group.add(disk_actor)
        group.add(label_actor)
        group.add(line_actor)

    return {
        "group": group,
        "center_disk": center_disk,
        "disks": disks,
        "labels": labels_actors,
        "lines": lines,
        "line_points": line_points,
        "axis_vectors": axis_vectors,
    }


def line(
    lines,
    *,
    colors=(1, 0, 0),
    opacity=None,
    material="basic",
    enable_picking=True,
):
    """
    Visualize one or many lines with different colors.

    Parameters
    ----------
    lines : list of ndarray of shape (P, 3) or ndarray of shape (N, P, 3)
        Lines points.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA colors. Accepts values in [0, 255] (int), [0, 1] (float),
        or hex strings (e.g. "#FF0000"). Values above 1.0 are treated as
        [0, 255] and normalized internally.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    material : str, optional
        The material type for the lines. Options are 'basic', 'segment', 'arrow',
        'thin', and 'thin_segment'.
    enable_picking : bool, optional
        Whether the lines should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated lines, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> lines = [np.random.rand(10, 3) for _ in range(5)]
    >>> colors = np.random.rand(5, 3)
    >>> line_actor = actor.line(lines=lines, colors=colors)
    >>> _ = scene.add(line_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    if colors is not None:
        colors = normalize_colors(colors)
        if colors.ndim == 2 and len(colors) == 1:
            colors = colors[0]

    lines_positions, lines_colors = line_buffer_separator(lines, color=colors)

    geo = buffer_to_geometry(
        positions=lines_positions.astype("float32"),
        colors=lines_colors.astype("float32"),
    )

    mat = _create_line_material(
        material=material,
        enable_picking=enable_picking,
        mode="vertex",
        opacity=opacity,
    )

    obj = create_line(geometry=geo, material=mat)

    obj.local.position = lines_positions[0]

    obj.prim_count = len(lines)

    return obj
