from numpy import pi
import pygfx as gfx


def lines(
    positions,
    colors=(1, 1, 1, 1),
    thickness=5.0,
    opacity=1.0,
    color_mode="auto",
    material="base",
    enable_picking=True,
):
    geo = gfx.Geometry(positions=positions, colors=colors)
    mat = create_line_material(material, thickness, opacity, color_mode, enable_picking)
    obj = gfx.Line(geo, mat)
    return obj


def sphere(
    radius=1,
    width_segments=32,
    height_segments=16,
    phi_start=0,
    phi_length=pi * 2,
    theta_start=0,
    theta_length=pi,
    material="phong",
    color=(1, 1, 1, 1),
    position=(0, 0, 0),
    enable_picking=True,
):
    geo = gfx.sphere_geometry(
        radius,
        width_segments,
        height_segments,
        phi_start,
        phi_length,
        theta_start,
        theta_length,
    )

    mat = create_mesh_material(material, color, enable_picking)
    obj = gfx.Mesh(geo, mat)
    obj.local.position = position

    return obj


def create_mesh_material(material, color=(1, 1, 1, 1), enable_picking=True):
    if material == "phong":
        return gfx.MeshPhongMaterial(color=color, pick_write=enable_picking)


def create_line_material(
    material, thickness=1.0, opacity=1.0, color_mode="auto", enable_picking=True
):
    if material == "base":
        return gfx.LineMaterial(
            thickness=thickness,
            opacity=opacity,
            color_mode=color_mode,
            pick_write=enable_picking,
        )


def points(radius, point_positions, colors, position=(0, 0, 0)):
    group = gfx.Group()

    for i in range(len(point_positions)):
        group.add(sphere(radius=radius, color=colors[i], position=point_positions[i]))

    group.local.position = position
    return group
