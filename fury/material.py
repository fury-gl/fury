"""Module for creating various materials used in 3D rendering."""

import numpy as np

from fury.lib import (
    ImageBasicMaterial,
    LineArrowMaterial,
    LineMaterial,
    LineSegmentMaterial,
    LineThinMaterial,
    LineThinSegmentMaterial,
    MeshBasicMaterial,
    MeshPhongMaterial,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsMaterial,
    TextMaterial,
)


def validate_opacity(opacity):
    """Ensure opacity is between 0 and 1.

    Parameters
    ----------
    opacity : float
        Opacity value to validate.

    Returns
    -------
    float
        Validated opacity value.

    Raises
    ------
    ValueError
        If opacity is not between 0 and 1.
    """
    if opacity is None:
        return 1.0
    if not (0 <= opacity <= 1):
        raise ValueError("Opacity must be between 0 and 1.")
    return opacity


def validate_color(color, opacity, mode):
    """Validate and modify color based on opacity and mode.

    Parameters
    ----------
    color : tuple or None
        RGB or RGBA color tuple.
    opacity : float
        Opacity value between 0 and 1.
    mode : str
        Color mode, either 'auto' or 'vertex'.

    Returns
    -------
    tuple or None
        Modified color tuple with opacity applied.

    Raises
    ------
    ValueError
        If color is None when mode is 'auto' or if color has invalid length.
    """
    if color is None and mode == "auto":
        return (1, 1, 1, opacity)

    if mode == "vertex":
        return (1, 1, 1)

    if color is not None:
        if len(color) == 3:
            return (*color, opacity)
        elif len(color) == 4:
            return (*color[:3], color[3] * opacity)
        else:
            raise ValueError("Color must be a tuple of length 3 or 4.")
    return color


def _create_mesh_material(
    *,
    material="phong",
    enable_picking=True,
    color=None,
    opacity=1.0,
    mode="vertex",
    flat_shading=True,
):
    """Create a mesh material.

    Parameters
    ----------
    material : str, optional
        The type of material to create. Options are 'phong' (default) and
        'basic'.
    enable_picking : bool, optional
        Whether the material should be pickable in a scene.
    color : tuple or None, optional
        The color of the material, represented as an RGB or RGBA tuple. If None,
        the default color is used.
    opacity : float, optional
        The opacity of the material, from 0 (transparent) to 1 (opaque).
        If RGBA is provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    mode : str, optional
        The color mode of the material. Options are 'auto' and 'vertex'.
    flat_shading : bool, optional
        Whether to use flat shading (True) or smooth shading (False).

    Returns
    -------
    MeshMaterial
        A mesh material object of the specified type with the given properties.

    Raises
    ------
    ValueError
        If an unsupported material type is specified.
    """
    opacity = validate_opacity(opacity)
    color = validate_color(color, opacity, mode)

    if material == "phong":
        return MeshPhongMaterial(
            pick_write=enable_picking,
            color_mode=mode,
            color=color,
            flat_shading=flat_shading,
        )
    elif material == "basic":
        return MeshBasicMaterial(
            pick_write=enable_picking,
            color_mode=mode,
            color=color,
            flat_shading=flat_shading,
        )
    else:
        raise ValueError(f"Unsupported material type: {material}")


def _create_line_material(
    *,
    material="basic",
    enable_picking=True,
    color=None,
    opacity=1.0,
    mode="auto",
    thickness=2.0,
    thickness_space="screen",
    dash_pattern=(),
    dash_offset=0.0,
    anti_aliasing=True,
):
    """
    Create a line material.

    Parameters
    ----------
    material : str, optional
        The type of line material to create. Options are 'line' (default),
        'segment', 'arrow', 'thin', and 'thin_segment'.
    enable_picking : bool, optional
        Whether the material should be pickable in a scene.
    color : tuple or None, optional
        The color of the material, represented as an RGBA tuple. If None, the
        default color is used.
    opacity : float, optional
        The opacity of the material, from 0 (transparent) to 1 (opaque).
        If RGBA is provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    mode : str, optional
        The color mode of the material. Options are 'auto' and 'vertex'.
    thickness : float, optional
        The line thickness expressed in logical pixels.
    thickness_space : str, optional
        The coordinate space in which the thickness is
        expressed ('screen', 'world', 'model').
    dash_pattern : tuple, optional
        The pattern of the dash, e.g., [2, 3].
        meaning no dashing.
    dash_offset : float, optional
        The offset into the dash cycle to start drawing at.
    anti_aliasing : bool, optional
        Whether or not the line is anti-aliased in the shader.

    Returns
    -------
    LineMaterial
        A line material object of the specified type with the given properties.
    """

    opacity = validate_opacity(opacity)
    color = validate_color(color, opacity, mode)

    args = {
        "pick_write": enable_picking,
        "color_mode": mode,
        "color": color,
        "thickness": thickness,
        "thickness_space": thickness_space,
        "dash_pattern": dash_pattern,
        "dash_offset": dash_offset,
        "aa": anti_aliasing,
    }

    if material == "basic":
        return LineMaterial(**args)
    elif material == "segment":
        return LineSegmentMaterial(**args)
    elif material == "arrow":
        return LineArrowMaterial(**args)
    elif material == "thin":
        return LineThinMaterial(**args)
    elif material == "thin_segment":
        return LineThinSegmentMaterial(**args)
    else:
        raise ValueError(f"Unsupported material type: {material}")


def _create_vector_field_material(
    cross_section,
    *,
    material="thin_line",
    enable_picking=True,
    opacity=1.0,
    thickness=1.0,
    thickness_space="screen",
    anti_aliasing=True,
):
    """
    Create a line material.

    Parameters
    ----------
    cross_section : list or tuple, shape (3,), optional
        A list or tuple representing the cross section dimensions.
        If None, the cross section will be ignored and complete field will be shown.
    material : str, optional
        The type of vector field material to create. Options are 'thin_line' (default),
        'line', 'arrow'.
    enable_picking : bool, optional
        Whether the material should be pickable in a scene.
    opacity : float, optional
        The opacity of the material, from 0 (transparent) to 1 (opaque).
        If RGBA is provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    thickness : float, optional
        The line thickness expressed in logical pixels.
    thickness_space : str, optional
        The coordinate space in which the thickness is
        expressed ('screen', 'world', 'model').
    anti_aliasing : bool, optional
        Whether or not the line is anti-aliased in the shader.

    Returns
    -------
    LineMaterial
        A line material object of the specified type with the given properties.
    """

    opacity = validate_opacity(opacity)

    args = {
        "pick_write": enable_picking,
        "opacity": opacity,
        "thickness": thickness,
        "thickness_space": thickness_space,
        "aa": anti_aliasing,
    }

    if material == "thin_line":
        return VectorFieldThinLineMaterial(cross_section, **args)
    elif material == "line":
        return VectorFieldLineMaterial(cross_section, **args)
    elif material == "arrow":
        return VectorFieldArrowMaterial(cross_section, **args)
    else:
        raise ValueError(f"Unsupported material type: {material}")


def _create_points_material(
    *,
    material="basic",
    color=(1.0, 1.0, 1.0),
    size=4.0,
    map=None,
    aa=True,
    marker="circle",
    edge_color="black",
    edge_width=1.0,
    mode="vertex",
    opacity=1.0,
    enable_picking=True,
):
    """Create a points material.

    Parameters
    ----------
    material : str, optional
        The type of material to create. Options are 'basic' (default),
        'gaussian', and 'marker'.
    color : tuple, optional
        RGB or RGBA values in the range [0, 1].
    size : float, optional
        The size (diameter) of the points in logical pixels.
    map : TextureMap or Texture, optional
        The texture map specifying the color for each texture coordinate.
    aa : bool, optional
        Whether or not the points are anti-aliased in the shader.
    marker : str or MarkerShape, optional
        The shape of the marker.
        Options are "●": "circle", "+": "plus", "x": "cross", "♥": "heart",
        "✳": "asterix".
    edge_color : str or tuple or Color, optional
        The color of line marking the edge of the markers.
    edge_width : float, optional
        The width of the edge of the markers.
    mode : str, optional
        The color mode of the material. Options are 'auto' and 'vertex'.
    opacity : float, optional
        The opacity of the material, from 0 (transparent) to 1 (opaque).
        If RGBA is provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    enable_picking : bool, optional
        Whether the material should be pickable in a scene.

    Returns
    -------
    PointsMaterial
        A point material object of the specified type with the given properties.

    Raises
    ------
    ValueError
        If an unsupported material type is specified.
    """
    opacity = validate_opacity(opacity)
    color = validate_color(color, opacity, mode)

    if material == "basic":
        return PointsMaterial(
            color=color,
            size=size,
            color_mode=mode,
            map=map,
            aa=aa,
            pick_write=enable_picking,
        )
    elif material == "gaussian":
        return PointsGaussianBlobMaterial(
            color=color,
            size=size,
            color_mode=mode,
            map=map,
            aa=aa,
            pick_write=enable_picking,
        )
    elif material == "marker":
        return PointsMarkerMaterial(
            color=color,
            size=size,
            marker=marker,
            edge_color=edge_color,
            edge_width=edge_width,
            pick_write=enable_picking,
            color_mode=mode,
        )
    else:
        raise ValueError(f"Unsupported material type: {material}")


def _create_text_material(
    *,
    color=(0, 0, 0),
    opacity=1.0,
    outline_color=(0, 0, 0),
    outline_thickness=0.0,
    weight_offset=1.0,
    aliasing=True,
):
    """Create a text material.

    Parameters
    ----------
    color : tuple, optional
        The color of the text as RGB or RGBA tuple.
    opacity : float, optional
        The opacity of the material, from 0 (transparent) to 1 (opaque).
        If RGBA is provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    outline_color : tuple, optional
        The color of the outline of the text as RGB or RGBA tuple.
    outline_thickness : float, optional
        A value indicating the relative width of the outline. Valid values are
        between 0.0 and 0.5.
    weight_offset : float, optional
        A value representing an offset to the font weight. Font weights are in
        the range 100-900, so this value should be in the same order of
        magnitude. Can be negative to make text thinner.
    aliasing : bool, optional
        If True, use anti-aliasing while rendering glyphs. Aliasing gives
        prettier results, but may affect performance for very large texts.

    Returns
    -------
    TextMaterial
        A text material object with the specified properties.
    """
    opacity = validate_opacity(opacity)

    if color is not None:
        if len(color) == 3:
            color = (*color, opacity)
        elif len(color) == 4:
            color = color
            color = (*color[:3], color[3] * opacity)
        else:
            raise ValueError("Color must be a tuple of length 3 or 4.")

    return TextMaterial(
        color=color,
        outline_color=outline_color,
        outline_thickness=outline_thickness,
        weight_offset=weight_offset,
        aa=aliasing,
    )


def _create_image_material(
    *,
    clim=None,
    map=None,
    gamma=1.0,
    interpolation="nearest",
):
    """
    Rasterized image material.

    Parameters
    ----------
    clim : tuple, optional
        The contrast limits to scale the data values with.
    map : Texture or TextureMap, optional
        The texture map to turn the image values into its final color.
    gamma : float, optional
        The gamma correction to apply to the image data.
        Must be greater than 0.0.
    interpolation : str, optional
        The method to interpolate the image data.
        Either 'nearest' or 'linear'.

    Returns
    -------
    ImageMaterial
        A rasterized image material object with the specified properties.
    """
    return ImageBasicMaterial(
        clim=clim,
        map=map,
        gamma=gamma,
        interpolation=interpolation,
    )


class VectorFieldThinLineMaterial(LineMaterial):
    """Material for VectorFieldActor.

    Parameters
    ----------
    cross_section : {list, tuple, ndarray}
        A list or tuple  or ndarray representing the cross section dimensions.
    **kwargs : dict
            Additional keyword arguments for the material.
    """

    uniform_type = dict(
        LineThinSegmentMaterial.uniform_type,
        cross_section="4xi4",  # vec3<i32>
    )

    def __init__(self, cross_section, **kwargs):
        """Initialize the VectorFieldMaterial.

        Parameters
        ----------
        cross_section : {list, tuple, ndarray}
            A list or tuple  or ndarray representing the cross section dimensions.
        **kwargs : dict
            Additional keyword arguments for the material.
        """
        super().__init__(color_mode="vertex", **kwargs)
        self.cross_section = cross_section

    @property
    def cross_section(self):
        """Get the cross section of the vector field.

        Returns
        -------
        list
            A list representing the cross section dimensions.
        """
        return self.uniform_buffer.data["cross_section"][:3]

    @cross_section.setter
    def cross_section(self, cross_section):
        """Set the cross section of the vector field.

        Parameters
        ----------
        cross_section : list or tuple
            A list or tuple representing the cross section dimensions.
        """
        if len(cross_section) != 3:
            raise ValueError("cross_section must have exactly 3 dimensions.")
        if not all(
            isinstance(i, int) or (hasattr(i, "item") and isinstance(i.item(), int))
            for i in cross_section
        ):
            raise ValueError("cross_section must contain only integers.")

        self.uniform_buffer.data["cross_section"] = np.asarray(
            [*cross_section, 0], dtype=np.int32
        )
        self.uniform_buffer.update_full()


class VectorFieldLineMaterial(VectorFieldThinLineMaterial):
    """Material for VectorFieldActor.

    This class provides a way to distinguish the usage of right shader for
    creating a vector field.
    """


class VectorFieldArrowMaterial(VectorFieldThinLineMaterial):
    """Material for VectorFieldActor.

    This class provides a way to distinguish the usage of right shader for
    creating a vector field.
    """


class SphGlyphMaterial(MeshPhongMaterial):
    """Initialize the Spherical Glyph Material.

    Parameters
    ----------
    l_max : int, optional
        The maximum spherical harmonic degree.
    scale : int, optional
        The scale factor.
    shininess : int, optional
        The shininess factor.
    emissive : str, optional
        The emissive color.
    specular : str, optional
        The specular color.
    **kwargs : dict
            Additional keyword arguments for the material.
    """

    uniform_type = dict(
        MeshPhongMaterial.uniform_type,
        l_max="i4",
        scale="f4",
    )

    def __init__(
        self,
        l_max=4,
        scale=2,
        shininess=30,
        emissive="#000",
        specular="#494949",
        **kwargs,
    ):
        """Initialize the Spherical Glyph Material.

        Parameters
        ----------
        l_max : int, optional
            The maximum spherical harmonic degree.
        scale : int, optional
            The scale factor.
        shininess : int, optional
            The shininess factor.
        emissive : str, optional
            The emissive color.
        specular : str, optional
            The specular color.
        **kwargs : dict
            Additional keyword arguments for the material.
        """
        super().__init__(shininess, emissive, specular, **kwargs)
        self.l_max = l_max
        self.scale = scale

    @property
    def l_max(self):
        """Get the maximum spherical harmonic degree.

        Returns
        -------
        int
            The maximum spherical harmonic degree.
        """
        return self.uniform_buffer.data["l_max"]

    @l_max.setter
    def l_max(self, value):
        """Set the maximum spherical harmonic degree.

        Parameters
        ----------
        value : int
            The maximum spherical harmonic degree.
        """
        if not isinstance(value, int):
            raise ValueError("l_max must be an integer.")
        self.uniform_buffer.data["l_max"] = value
        self.uniform_buffer.update_full()

    @property
    def scale(self):
        """Get the scale factor.

        Returns
        -------
        float
            The scale factor.
        """
        return self.uniform_buffer.data["scale"]

    @scale.setter
    def scale(self, value):
        """Set the scale factor.

        Parameters
        ----------
        value : float
            The scale factor.
        """
        if not isinstance(value, (int, float)):
            raise ValueError("scale must be a number.")
        self.uniform_buffer.data["scale"] = value
        self.uniform_buffer.update_full()
