import pygfx as gfx
from pygfx import Mesh
from pygfx.renderers.wgpu import register_wgpu_render_function

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

    if not (0 <= opacity <= 1):
        raise ValueError("Opacity must be between 0 and 1.")

    if color is None and mode == "auto":
        raise ValueError("Color must be specified when mode is 'auto'.")

    elif color is not None:
        if len(color) == 3:
            color = (*color, opacity)
        elif len(color) == 4:
            color = color
            color = (*color[:3], color[3] * opacity)
        else:
            raise ValueError("Color must be a tuple of length 3 or 4.")

    if mode == "vertex":
        color = (1, 1, 1)

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
    enable_picking=True,
    color=None,
    opacity=1.0,
    mode="vertex",
):
    """
    Create a points material.

    Parameters
    ----------
    material : str, optional
        The type of material to create. Options are 'baisc' (default),
        'gaussian', and 'marker'.
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

    Returns
    -------
    PointsMaterial
        A point material object of the specified type with the given properties.
    """

    if not (0 <= opacity <= 1):
        raise ValueError("Opacity must be between 0 and 1.")

    if color is None and mode == "auto":
        raise ValueError("Color must be specified when mode is 'auto'.")

    elif color is not None:
        if len(color) == 3:
            color = (*color, opacity)
        elif len(color) == 4:
            color = color
            color = (*color[:3], color[3] * opacity)
        else:
            raise ValueError("Color must be a tuple of length 3 or 4.")

    if mode == "vertex":
        color = (1, 1, 1)

    if material == "basic":
        return gfx.PointsMaterial(
            size=4,
            pick_write=enable_picking,
            color_mode=mode,
            color=color,
        )
    elif material == "gaussian":
        return gfx.PointsGaussianBlobMaterial(
            size=4,
            pick_write=enable_picking,
            color_mode=mode,
            color=color,
        )
    elif material == "marker":
        return gfx.PointsMarkerMaterial(
            size=4,
            pick_write=enable_picking,
            color_mode=mode,
            color=color,
        )
    else:
        raise ValueError(f"Unsupported material type: {material}")
