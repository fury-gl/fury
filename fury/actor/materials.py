import pygfx as gfx

def _create_mesh_material(material='phong', enable_picking=True, color=None, opacity=1.0):
    if material == 'phong':
        return gfx.MeshPhongMaterial(
            pick_write=enable_picking,
            color_mode='vertex' if color is None else 'auto',
            color=color if color is not None else (1, 1, 1, opacity if color is None else 1),
        )
    elif material == 'basic':
        return gfx.MeshBasicMaterial(
            pick_write=enable_picking,
            color_mode='vertex' if color is None else 'auto',
            color=color if color is not None else (1, 1, 1, opacity if color is None else 1),
        )
