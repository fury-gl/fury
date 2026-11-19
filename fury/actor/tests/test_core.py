import numpy as np
import pytest

from fury import actor, geometry, material, window
from fury.actor import (
    Actor,
    Image as ActorImage,
    Text,
    actor_from_primitive,
    create_axes_helper,
)
from fury.actor.tests._helpers import (
    CENTERS,
    PHYSICAL_ONLY_PARAMS,
    assert_rejects_pbr_params_on_phong,
    assert_supports_pbr,
    render_manager,
    render_snapshot,
    snapshot_mask,
    validate_actors,
)
from fury.lib import DirectionalLight, Texture
from fury.material import PBRMaterial
from fury.primitive import prim_square


def test_actor_rotate_sets_quaternion_on_local_rotation(sphere_actor):
    sphere_actor.rotate((90, 0, 0))

    assert sphere_actor.local.rotation is not None
    np.testing.assert_array_almost_equal(
        sphere_actor.local.rotation,
        Actor._euler_to_quaternion(np.radians([90, 0, 0])),
    )


@pytest.mark.parametrize(
    "invalid_rotation",
    [
        (0, 0),  # too short
        (0, 0, 0, 0),  # too long
    ],
)
def test_actor_rotate_invalid_input_raises(sphere_actor, invalid_rotation):
    with pytest.raises(ValueError, match="Rotation must contain three angles"):
        sphere_actor.rotate(invalid_rotation)


def test_actor_translate_updates_local_translation_vector(sphere_actor):
    sphere_actor.translate((1, 2, 3))

    np.testing.assert_array_equal(
        sphere_actor.local.position, np.array([1, 2, 3], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "scale_input,expected",
    [
        (2, np.array([2, 2, 2], dtype=np.float32)),
        ((1, 2, 3), np.array([1, 2, 3], dtype=np.float32)),
    ],
)
def test_actor_scale_accepts_scalar_and_vector_inputs(
    sphere_actor, scale_input, expected
):
    sphere_actor.scale(scale_input)

    np.testing.assert_array_equal(sphere_actor.local.scale, expected)


@pytest.mark.parametrize(
    "invalid_scale",
    [
        (1, 2),
        (1, 2, 3, 4),
    ],
)
def test_actor_scale_invalid_shape(sphere_actor, invalid_scale):
    with pytest.raises(ValueError, match="Scale must contain three values"):
        sphere_actor.scale(invalid_scale)


def test_actor_transform_accepts_4x4_matrix(sphere_actor):
    matrix = np.eye(4, dtype=np.float32)

    sphere_actor.transform(matrix)

    np.testing.assert_array_equal(sphere_actor.local.matrix, matrix)


@pytest.mark.parametrize(
    "invalid_matrix",
    [
        np.eye(3),
        np.ones((4, 3)),
        np.zeros((5, 5)),
    ],
)
def test_actor_transform_invalid_matrix_shape(sphere_actor, invalid_matrix):
    with pytest.raises(ValueError, match="Transformation matrix must be of shape"):
        sphere_actor.transform(invalid_matrix)


@pytest.mark.parametrize(
    "opacity",
    [0, 0.5, 1],
)
def test_actor_opacity_sets_material(sphere_actor, opacity):
    sphere_actor.opacity = opacity
    assert sphere_actor.material.opacity == opacity


@pytest.mark.parametrize(
    "invalid_opacity",
    [
        -0.5,  # negative
        1.5,  # greater than 1
    ],
)
def test_actor_opacity_invalid_values_raise(sphere_actor, invalid_opacity):
    with pytest.raises(ValueError, match="Opacity must be"):
        sphere_actor.opacity = invalid_opacity


def test_create_mesh():
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype("float32")
    geo = geometry.buffer_to_geometry(positions)
    mat = material._create_mesh_material(
        material="phong", color=(1, 0, 0), opacity=0.5, mode="auto"
    )
    mesh = actor.create_mesh(geometry=geo, material=mat)
    assert mesh.geometry == geo
    assert mesh.material == mat


def test_create_point():
    vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype("float32")
    geo = geometry.buffer_to_geometry(vertices)
    mat = material._create_points_material(
        material="basic", color=(1, 0, 0), opacity=0.5, mode="auto"
    )
    point = actor.create_point(geometry=geo, material=mat)
    assert point.geometry == geo
    assert point.material == mat


def test_create_text():
    text = "FURY"
    mat = material._create_text_material(color=(1, 0, 0), opacity=0.5)
    text_obj = actor.create_text(text=text, material=mat)
    assert text_obj.material == mat
    assert isinstance(text_obj, Text)


def test_create_image():
    img_data = np.random.rand(128, 128)
    mat = material._create_image_material()
    image_obj = actor.create_image(image_input=img_data, material=mat)
    assert image_obj.material == mat
    assert isinstance(image_obj, ActorImage)
    assert image_obj.geometry.grid.dim == 2


def test_line():
    lines_points = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 1, 1], [2, 2, 2]]])
    colors = np.array([[[1, 0, 0]], [[0, 1, 0]]])
    validate_actors(lines=lines_points, colors=colors, actor_type="line", prim_count=2)

    line = np.array([[0, 0, 0], [1, 1, 1]])
    colors = None
    validate_actors(lines=line, colors=colors, actor_type="line", prim_count=2)

    line = np.array([[0, 0, 0], [1, 1, 1]])
    actor.line(line, colors=colors)
    actor.line(line)
    actor.line(line, colors=colors)
    actor.line(line, colors=colors, material="basic")
    actor.line(line, colors=line, material="basic")


def test_arrow():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="arrow")


def test_axes():
    scene = window.Scene()
    axes_actor = actor.axes()
    scene.add(axes_actor)

    assert axes_actor.prim_count == 3

    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    assert report.objects >= 1
    mean_r, mean_g, mean_b, _mean_a = np.mean(arr.reshape(-1, arr.shape[2]), axis=0)
    assert np.isclose(mean_r, mean_g, atol=0.02)
    assert 0 < mean_r < 255
    assert 0 < mean_g < 255
    assert 0 < mean_b < 255

    scene.remove(axes_actor)


def test_actor_from_primitive_position(sphere_prim):
    vertices, faces = sphere_prim
    centers = np.array([[5, 10, 15]], dtype=np.float32)
    colors = np.array([[1, 0, 0]])

    obj = actor_from_primitive(
        vertices, faces, centers, colors=colors, repeat_primitive=True
    )
    np.testing.assert_array_equal(
        obj.local.position, np.array([0, 0, 0], dtype=np.float32)
    )
    mean_vertex = np.round(np.mean(obj.geometry.positions.data, axis=0))
    np.testing.assert_array_almost_equal(mean_vertex, centers[0])

    obj = actor_from_primitive(
        vertices, faces, centers, colors=colors, repeat_primitive=False
    )
    np.testing.assert_array_equal(
        obj.local.position, np.array([5, 10, 15], dtype=np.float32)
    )


def test_actor_from_primitive_transparency(sphere_prim):
    vertices, faces = sphere_prim
    centers = np.array([[0, 0, 0]], dtype=np.float32)
    rgb = np.array([[1, 0, 0]])
    rgba_opaque = np.array([[1, 0, 0, 1.0]])
    rgba_transparent = np.array([[1, 0, 1, 0.3]])

    obj = actor_from_primitive(vertices, faces, centers, colors=rgb)
    assert obj.material.alpha_mode == "auto"
    assert obj.material.depth_write is True

    obj = actor_from_primitive(vertices, faces, centers, colors=rgb, opacity=1.0)
    assert obj.material.alpha_mode == "auto"
    assert obj.material.depth_write is True

    obj = actor_from_primitive(vertices, faces, centers, colors=rgba_opaque)
    assert obj.material.alpha_mode == "auto"
    assert obj.material.depth_write is True

    obj = actor_from_primitive(vertices, faces, centers, colors=rgb, opacity=0.5)
    assert obj.material.alpha_mode == "weighted_blend"
    assert obj.material.depth_write is False

    obj = actor_from_primitive(vertices, faces, centers, colors=rgba_transparent)
    assert obj.material.alpha_mode == "weighted_blend"
    assert obj.material.depth_write is False

    obj = actor_from_primitive(vertices, faces, centers, colors=rgb, opacity=0.4)
    np.testing.assert_allclose(obj.geometry.colors.data[:, 3], 0.4, atol=0.01)

    rgba_half = np.array([[1, 0, 0, 0.5]])
    obj = actor_from_primitive(vertices, faces, centers, colors=rgba_half, opacity=0.5)
    np.testing.assert_allclose(obj.geometry.colors.data[:, 3], 0.25, atol=0.01)
    assert obj.material.alpha_mode == "weighted_blend"
    assert obj.material.depth_write is False


def test_actor_from_primitive_transparency_visual(sphere_prim):
    vertices, faces = sphere_prim
    centers = np.array([[0, 0, 0]], dtype=np.float32)
    colors = np.array([[1, 0, 0]])

    scene = window.Scene(background=(0, 1, 0))

    opaque = actor_from_primitive(vertices, faces, centers, colors=colors)
    scene.add(opaque)
    img_array_op = window.snapshot(scene=scene, fname=None, return_array=True)
    mid = img_array_op[img_array_op.shape[0] // 2, img_array_op.shape[1] // 2]
    assert mid[0] > mid[1] and mid[0] > mid[2]
    scene.remove(opaque)

    transparent = actor_from_primitive(
        vertices, faces, centers, colors=colors, opacity=0.5
    )
    scene.add(transparent)
    img_array_tr = window.snapshot(scene=scene, fname=None, return_array=True)
    mean_r_tr, mean_g_tr, mean_b_tr, _ = np.mean(
        img_array_tr.reshape(-1, img_array_tr.shape[2]), axis=0
    )
    assert mean_r_tr > 0

    mean_r_op, mean_g_op, mean_b_op, _ = np.mean(
        img_array_op.reshape(-1, img_array_op.shape[2]), axis=0
    )
    assert mean_r_op > 0
    assert mean_g_tr > mean_g_op
    scene.remove(transparent)


def test_arrow_per_instance_geometry():
    """Test arrow actor with per-instance geometry parameters."""
    centers = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Per-instance height, tip_length, tip_radius, shaft_radius
    heights = np.array([1.0, 1.5, 2.0])
    tip_lengths = np.array([0.3, 0.4, 0.5])
    tip_radii = np.array([0.08, 0.1, 0.12])
    shaft_radii = np.array([0.02, 0.03, 0.04])
    arrow_actor = actor.arrow(
        centers=centers,
        colors=colors,
        height=heights,
        tip_length=tip_lengths,
        tip_radius=tip_radii,
        shaft_radius=shaft_radii,
    )
    assert arrow_actor.prim_count == 3

    # Verify geometry varies: per-instance vertex height extents differ
    verts = arrow_actor.geometry.positions.view
    n_verts_per = len(verts) // 3
    y_extents = []
    for i in range(3):
        chunk = verts[i * n_verts_per : (i + 1) * n_verts_per] - centers[i]
        y_extents.append(chunk[:, 0].max() - chunk[:, 0].min())
    # Larger heights should yield larger extents
    assert y_extents[0] < y_extents[1] < y_extents[2]

    # Scalar (backward compat)
    arrow_actor_scalar = actor.arrow(centers=centers, colors=colors, height=1.0)
    assert arrow_actor_scalar.prim_count == 3

    # Wrong-size array raises ValueError
    with pytest.raises(ValueError, match="height"):
        actor.arrow(
            centers=centers,
            colors=colors,
            height=np.array([1.0, 1.5]),
        )

    with pytest.raises(ValueError, match="tip_length"):
        actor.arrow(
            centers=centers,
            colors=colors,
            tip_length=np.array([0.3, 0.4]),
        )

    with pytest.raises(ValueError, match="tip_radius"):
        actor.arrow(
            centers=centers,
            colors=colors,
            tip_radius=np.array([0.08, 0.1]),
        )

    with pytest.raises(ValueError, match="shaft_radius"):
        actor.arrow(
            centers=centers,
            colors=colors,
            shaft_radius=np.array([0.02, 0.03]),
        )


def test_create_axes_helper_structure():
    helper = create_axes_helper()

    expected_keys = {
        "group",
        "center_disk",
        "disks",
        "labels",
        "lines",
        "line_points",
        "axis_vectors",
    }

    assert set(helper.keys()) == expected_keys
    assert len(helper["disks"]) == 6
    assert len(helper["labels"]) == 6
    assert len(helper["lines"]) == 6


def test_create_axes_helper_axis_vector_order():
    helper = create_axes_helper()
    axis_vectors = np.stack(helper["axis_vectors"])
    expected = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(axis_vectors, expected)


def test_actor_accepts_255_colors():
    """Colors in [0, 255] produce same result as [0, 1]."""
    centers = np.array([[0, 0, 0]])
    a1 = actor.arrow(centers=centers, colors=(255, 0, 0))
    a2 = actor.arrow(centers=centers, colors=(1.0, 0.0, 0.0))
    np.testing.assert_array_almost_equal(
        a1.geometry.colors.view, a2.geometry.colors.view
    )


def test_actor_accepts_hex_colors():
    """Hex color strings produce same result as [0, 1]."""
    centers = np.array([[0, 0, 0]])
    a1 = actor.arrow(centers=centers, colors="#FF0000")
    a2 = actor.arrow(centers=centers, colors=(1.0, 0.0, 0.0))
    np.testing.assert_array_almost_equal(
        a1.geometry.colors.view, a2.geometry.colors.view
    )


def test_actor_accepts_255_array_colors():
    """Array of [0, 255] colors works for multiple centers."""
    centers = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    colors_255 = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    a = actor.arrow(centers=centers, colors=colors_255)
    assert a.prim_count == 3


@pytest.mark.parametrize("mesh_material", ["standard", "physical"])
def test_arrow_supports_pbr(mesh_material):
    assert_supports_pbr("arrow", mesh_material)


def test_arrow_rejects_pbr_params_on_phong():
    assert_rejects_pbr_params_on_phong("arrow")


@pytest.mark.parametrize("mesh_material", ["standard", "physical"])
def test_actor_from_primitive_forwards_material_params(mesh_material):
    vertices, faces = prim_square()
    obj = actor_from_primitive(
        vertices,
        faces,
        centers=np.array([[0.0, 0.0, 0.0]]),
        material=mesh_material,
        material_params={"metalness": 0.8, "roughness": 0.2},
    )
    assert obj.material.metalness == pytest.approx(0.8)
    assert obj.material.roughness == pytest.approx(0.2)


@pytest.mark.parametrize("param, value", sorted(PHYSICAL_ONLY_PARAMS.items()))
def test_actor_from_primitive_forwards_physical_only_params(param, value):
    vertices, faces = prim_square()
    obj = actor_from_primitive(
        vertices,
        faces,
        centers=np.array([[0.0, 0.0, 0.0]]),
        material="physical",
        material_params={param: value},
    )
    assert getattr(obj.material, param) == pytest.approx(value)


def test_actor_from_primitive_rejects_physical_params_on_standard():
    vertices, faces = prim_square()
    with pytest.raises(ValueError, match="Use material='physical'"):
        actor_from_primitive(
            vertices,
            faces,
            centers=np.array([[0.0, 0.0, 0.0]]),
            material="standard",
            material_params={"clearcoat": 1.0},
        )


def test_manifest_pbr_converts_existing_mesh():
    box = actor.box(CENTERS, colors=(1, 0, 0))
    flat_shading = box.material.flat_shading
    pick_write = box.material.pick_write
    alpha_mode = box.material.alpha_mode

    mat = actor.manifest_pbr(
        box, metalness=0.8, roughness=0.3, material_params={"sheen": 0.5}
    )

    assert mat is box.material
    assert type(mat) is PBRMaterial
    assert mat.metalness == pytest.approx(0.8)
    assert mat.roughness == pytest.approx(0.3)
    assert mat.sheen == pytest.approx(0.5)
    assert mat.flat_shading == flat_shading
    assert mat.pick_write == pick_write
    assert mat.alpha_mode == alpha_mode
    tangents = box.geometry.tangents.data
    assert tangents.shape == (len(box.geometry.positions.data), 4)
    assert np.isfinite(tangents).all()


def test_manifest_pbr_updates_pbr_material_in_place():
    sphere = actor.sphere(CENTERS, impostor=False, material="physical")
    before = sphere.material

    assert actor.manifest_pbr(sphere, clearcoat=1.0) is before
    assert before.clearcoat == 1.0


def test_manifest_pbr_factory_builds_one_mesh_per_center(caplog):
    centers = np.array([[-5, 0, 0], [0, 0, 0], [5, 0, 0]], dtype=np.float32)
    colors = np.eye(3)
    metalness = (0.0, 0.5, 1.0)

    group = actor.manifest_pbr(
        actor.sphere,
        centers=centers,
        roughness=0.2,
        material_params=[{"metalness": m} for m in metalness],
        colors=colors,
        radii=1,
    )

    assert isinstance(group, actor.Group)
    assert len(group.children) == 3
    assert len({id(child.material) for child in group.children}) == 3
    for i, child in enumerate(group.children):
        assert type(child.material) is PBRMaterial
        assert child.material.metalness == pytest.approx(metalness[i])
        assert child.material.roughness == pytest.approx(0.2)
        assert child.prim_count == 1
        child_colors = child.geometry.colors.data
        np.testing.assert_allclose(
            child_colors[:, :3], np.tile(colors[i], (len(child_colors), 1))
        )
        positions = child.geometry.positions.data
        midpoint = (positions.min(axis=0) + positions.max(axis=0)) / 2
        np.testing.assert_allclose(midpoint, centers[i], atol=1e-4)
    assert "impostor" not in caplog.text


def test_manifest_pbr_factory_shared_dict_gives_independent_materials():
    group = actor.manifest_pbr(
        actor.sphere,
        centers=np.array([[0, 0, 0], [5, 0, 0]]),
        material_params={"clearcoat": 1.0},
    )
    first, second = group.children

    first.material.clearcoat = 0.0

    assert second.material.clearcoat == 1.0


def test_manifest_pbr_anisotropy_renders_on_primitive_spheres():
    images = []
    for strength in (0.0, 1.0):
        group = actor.manifest_pbr(
            actor.sphere,
            centers=CENTERS,
            metalness=1.0,
            roughness=0.4,
            anisotropy=strength,
            radii=3,
            phi=48,
            theta=48,
        )
        sun = DirectionalLight(intensity=3)
        sun.local.position = (5, 5, 10)
        with render_manager(group, sun, width=8) as show_m:
            images.append(render_snapshot(show_m))

    assert snapshot_mask(images[0]).any()
    assert snapshot_mask(images[1]).any()
    diff = np.abs(images[0][..., :3].astype(float) - images[1][..., :3].astype(float))
    assert diff.mean() > 0.1


def _crossed_boxes(centers, material, material_params=None):
    """
    Build boxes whose first face points into the next center's vertices.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Box centers.
    material : str
        Material type forwarded to ``actor.box``.
    material_params : dict or None
        Material properties forwarded to ``actor.box``.

    Returns
    -------
    Mesh
        Box mesh that cannot be split into independent per-center meshes.
    """
    boxes = actor.box(centers, material=material, material_params=material_params)
    faces = boxes.geometry.indices.data.reshape(-1, 3)
    faces[0] = faces[-1]
    return boxes


@pytest.mark.parametrize(
    "call, error, match",
    [
        (
            lambda: actor.manifest_pbr(
                actor.sphere,
                centers=CENTERS,
                material_params=[{}, {}],
            ),
            ValueError,
            "one dict per center",
        ),
        (
            lambda: actor.manifest_pbr(
                actor.sphere, centers=CENTERS, material_params={"shininess": 1}
            ),
            ValueError,
            "Unknown PBR material properties",
        ),
        (
            lambda: actor.manifest_pbr(
                actor.sphere, centers=CENTERS, material_params={"uniform_buffer": 1}
            ),
            ValueError,
            "Unknown PBR material properties",
        ),
        (
            lambda: actor.manifest_pbr(actor.sphere, centers=np.zeros((0, 3))),
            ValueError,
            "N >= 1",
        ),
        (
            lambda: actor.manifest_pbr(actor.sphere, centers=np.zeros((2, 2))),
            ValueError,
            "shape \\(N, 3\\)",
        ),
        (
            lambda: actor.manifest_pbr(
                _crossed_boxes, centers=np.array([[0, 0, 0], [5, 0, 0]])
            ),
            ValueError,
            "equal per-center meshes",
        ),
        (
            lambda: actor.manifest_pbr(actor.sphere(CENTERS)),
            TypeError,
            "cannot convert",
        ),
        (
            lambda: actor.manifest_pbr(actor.point, centers=CENTERS),
            TypeError,
            "does not accept material and material_params",
        ),
        (
            lambda: actor.manifest_pbr(actor.sphere, centers=CENTERS, material="phong"),
            TypeError,
            "material is set by manifest_pbr",
        ),
        (
            lambda: actor.manifest_pbr(actor.box(CENTERS), centers=CENTERS),
            TypeError,
            "only accepted when actor is an actor factory",
        ),
        (
            lambda: actor.manifest_pbr(42),
            TypeError,
            "must be a mesh actor or an actor factory",
        ),
    ],
)
def test_manifest_pbr_rejects_invalid_input(call, error, match):
    with pytest.raises(error, match=match):
        call()


def test_manifest_pbr_rejected_update_leaves_material_unchanged():
    sphere = actor.sphere(CENTERS, impostor=False, material="physical")

    with pytest.raises(ValueError, match="uniform_buffer"):
        actor.manifest_pbr(sphere, clearcoat=1.0, material_params={"uniform_buffer": 1})

    assert sphere.material.clearcoat == 0.0


@pytest.mark.parametrize("material", ["standard", "physical"])
def test_manifest_pbr_keeps_unspecified_pbr_properties(material):
    sphere = actor.sphere(
        CENTERS,
        impostor=False,
        material=material,
        material_params={"metalness": 1.0, "roughness": 0.1, "emissive": (1, 0, 0)},
    )

    mat = actor.manifest_pbr(sphere, clearcoat=1.0)

    assert type(mat) is PBRMaterial
    assert mat.metalness == pytest.approx(1.0)
    assert mat.roughness == pytest.approx(0.1)
    assert tuple(mat.emissive)[:3] == pytest.approx((1, 0, 0))
    assert mat.clearcoat == 1.0


def test_manifest_pbr_keeps_base_properties_when_converting_phong():
    box = actor.box(CENTERS, colors=(1, 0, 0), opacity=0.5)
    color = tuple(box.material.color)
    opacity = box.material.opacity

    mat = actor.manifest_pbr(box)

    assert tuple(mat.color) == pytest.approx(color)
    assert mat.opacity == pytest.approx(opacity)
    assert mat.roughness == pytest.approx(0.5)


def test_manifest_pbr_layer_can_be_switched_off_again():
    sphere = actor.sphere(CENTERS, impostor=False, material="physical")

    actor.manifest_pbr(sphere, clearcoat=1.0, anisotropy=1.0)
    actor.manifest_pbr(sphere, clearcoat=0.0, anisotropy=0.0)

    assert sphere.material.clearcoat == 0.0
    assert sphere.material.anisotropy == 0.0


def test_manifest_pbr_skips_generated_tangents_with_normal_map():
    normal_map = Texture(np.full((2, 2, 3), 128, dtype=np.uint8), dim=2)
    plain = actor.sphere(CENTERS, impostor=False)
    mapped = actor.sphere(CENTERS, impostor=False)

    actor.manifest_pbr(plain, anisotropy=1.0)
    actor.manifest_pbr(
        mapped, anisotropy=1.0, material_params={"normal_map": normal_map}
    )

    assert plain.geometry.tangents is not None
    assert getattr(mapped.geometry, "tangents", None) is None


def test_manifest_pbr_factory_skips_tangents_only_for_normal_mapped_children():
    normal_map = Texture(np.full((2, 2, 3), 128, dtype=np.uint8), dim=2)

    group = actor.manifest_pbr(
        actor.sphere,
        centers=np.array([[0, 0, 0], [5, 0, 0]]),
        material_params=[{"normal_map": normal_map}, {}],
    )
    mapped, plain = group.children

    assert getattr(mapped.geometry, "tangents", None) is None
    assert plain.geometry.tangents.data.shape == (
        len(plain.geometry.positions.data),
        4,
    )
