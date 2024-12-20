import pygfx as gfx


def _create_mesh_material(
    *, material="phong", enable_picking=True, color=None, opacity=1.0, mode="vertex"
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

    Returns
    -------
    gfx.MeshMaterial
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
        return gfx.MeshPhongMaterial(
            pick_write=enable_picking,
            color_mode=mode,
            color=color,
        )
    elif material == "basic":
        return gfx.MeshBasicMaterial(
            pick_write=enable_picking,
            color_mode=mode,
            color=color,
        )
    else:
        raise ValueError(f"Unsupported material type: {material}")
