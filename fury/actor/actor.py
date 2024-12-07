import fury.primitive as fp
import numpy as np
from fury.v2.actor.materials import _create_mesh_material
from fury.v2.actor.geometry import buffer_to_geometry, create_mesh


def sphere(
    centers,
    colors,
    *,
    radii=1.0,
    phi=16,
    theta=16,
    vertices=None,
    faces=None,
    opacity=1,
    material='phong',
    enable_picking=True
):


    scales = radii
    directions = (1, 0, 0)


    if faces is None and vertices is None:
        vertices, faces = fp.prim_sphere(phi=phi, theta=theta)

    res = fp.repeat_primitive(
        vertices,
        faces,
        directions=directions,
        centers=centers,
        colors=colors,
        scales=scales,
    )
    big_verts, big_faces, big_colors, _ = res
    print(big_colors)

    prim_count = len(centers)

    geo = buffer_to_geometry(
        indices=big_faces.astype('int32'),
        positions=big_verts.astype('float32'),
        texcoords=big_verts.astype('float32'),
        colors=np.array(big_colors, dtype='float32')/255.,
    )

    mat = _create_mesh_material(material=material, enable_picking=enable_picking)
    obj = create_mesh(geometry=geo, material=mat)
    obj.local.position = centers[0]
    obj.prim_count = prim_count
    return obj

