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
    if material == 'phong':
        return gfx.MeshPhongMaterial(
            pick_write=enable_picking,
            color_mode='vertex' if color is None else 'auto',
            color=color if color is not None else (1, 1, 1, opacity),
        )
    elif material == 'basic':
        return gfx.MeshBasicMaterial(
            pick_write=enable_picking,
            color_mode='vertex' if color is None else 'auto',
            color=color if color is not None else (1, 1, 1, opacity),
        )
