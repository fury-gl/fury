from fury import material


def test_create_mesh_material():
    color = (1, 0, 0)
    mat = material._create_mesh_material(
        material="phong", color=color, opacity=0.5, mode="auto"
    )
    assert type(mat) == material.gfx.MeshPhongMaterial
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0, 0.5)
    mat = material._create_mesh_material(
        material="phong", color=color, opacity=0.5, mode="auto"
    )
    assert type(mat) == material.gfx.MeshPhongMaterial
    assert mat.color == (1, 0, 0, 0.25)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_mesh_material(
        material="phong", color=color, opacity=0.5, mode="vertex"
    )
    assert type(mat) == material.gfx.MeshPhongMaterial
    assert mat.color == (1, 1, 1)
    assert mat.color_mode == "vertex"

    color = (1, 0, 0)
    mat = material._create_mesh_material(
        material="basic", color=color, mode="vertex", enable_picking=False
    )
    assert type(mat) == material.gfx.MeshBasicMaterial
    assert mat.color == (1, 1, 1)
    assert mat.color_mode == "vertex"
