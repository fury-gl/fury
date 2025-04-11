import pygfx as gfx
from pygfx.renderers.wgpu import register_wgpu_render_function

from fury.lib import (
    Mesh,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsMaterial,
    TextMaterial,
)
from fury.shader import MeshBasicShader, MeshPhongShader


class MeshPhongMaterial(gfx.MeshPhongMaterial):
    """
    Phong material.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MeshBasicMaterial(gfx.MeshBasicMaterial):
    """
    Basic material.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# Register the custom shaders for the mesh materials
register_wgpu_render_function(Mesh, MeshPhongMaterial)(MeshPhongShader)
register_wgpu_render_function(Mesh, MeshBasicMaterial)(MeshBasicShader)


def validate_opacity(opacity):
    """Ensure opacity is between 0 and 1."""
    if not (0 <= opacity <= 1):
        raise ValueError("Opacity must be between 0 and 1.")
    return opacity


def validate_color(color, opacity, mode):
    """Validate and modify color based on opacity and mode."""
    if color is None and mode == "auto":
        raise ValueError("Color must be specified when mode is 'auto'.")

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
    """
    Create a mesh material.

    Parameters
    ----------
    material : str, optional
        The type of material to create. Options are 'phong' (default) and
        'basic'.
    enable_picking : bool, optional
        Whether the material should be pickable in a scene.
    color : tuple or None, optional
        The color of the material, represented as an RGBA tuple. If None, the
        default color is used.
    opacity : float, optional
        The opacity of the material, from 0 (transparent) to 1 (opaque).
        If RGBA is provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    mode : str, optional
        The color mode of the material. Options are 'auto' and 'vertex'.
    flat_shading : bool, optional
        Whether to use flat shading (True) or smooth shading (False).

    Returns
    -------
    MeshMaterial
        A mesh material object of the specified type with the given properties.
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
    """
    Create a points material.

    Parameters
    ----------
    material : str, optional
        The type of material to create. Options are 'basic' (default),
        'gaussian', and 'marker'.
    colors : ndarray (N,3) or (N,4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
    size : float, optional
        The size (diameter) of the points in logical pixels.
    map : TextureMap | Texture
        The texture map specifying the color for each texture coordinate.
    aa : bool, optional
        Whether or not the points are anti-aliased in the shader.
    marker : str | MarkerShape
        The shape of the marker.
        Options are "●": "circle", "+": "plus", "x": "cross", "♥": "heart",
        "✳": "asterix".
    edge_color : str | tuple | Color
        The color of line marking the edge of the markers.
    edge_width : float
        The width of the edge of the markers.
    mode : str, optional
        The color mode of the material. Options are 'auto' and 'vertex'.
    opacity : float, optional
        The opacity of the material, from 0 (transparent) to 1 (opaque).
        If RGBA is provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    enable_picking : bool, optional
        Whether the material should be pickable in a scene.

    Returns
    -------
    PointsMaterial
        A point material object of the specified type with the given properties.
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
    aa=True,
):
    """
    Create a text material.

    Parameters
    ----------
    color : tuple
        The color of the text.
    opacity : float, optional
        The opacity of the material, from 0 (transparent) to 1 (opaque).
        If RGBA is provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    outline_color : tuple
        The color of the outline of the text.
    outline_thickness : float, optional
        A value indicating the relative width of the outline. Valid values are
        between 0.0 and 0.5.
    weight_offset : float, optional
        A value representing an offset to the font weight. Font weights are in
        the range 100-900, so this value should be in the same order of
        magnitude. Can be negative to make text thinner.
    aliasing : bool
        If True, use anti-aliasing while rendering glyphs. Aliasing gives
        prettier results, but may affect performance for very large texts.


    Returns
    -------
    TextMaterial
        A text material object of the specified type with the given properties.
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
        aa=aa,
    )
