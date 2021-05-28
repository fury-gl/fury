from fury import actor, material, window
from fury.optpkg import optional_package
import fury.testing as ft
from scipy.spatial import Delaunay


import math
import numpy as np
import numpy.testing as npt
import random
import pytest


dipy, have_dipy, _ = optional_package('dipy')
VTK_9_PLUS = window.vtk.vtkVersion.GetVTKMajorVersion() >= 9


def _generate_surface():
    size = 11
    vertices = list()
    for i in range(-size, size):
        for j in range(-size, size):
            fact1 = - math.sin(i) * math.cos(j)
            fact2 = - math.exp(abs(1 - math.sqrt(i ** 2 + j ** 2) / math.pi))
            z_coord = -abs(fact1 * fact2)
            vertices.append([i, j, z_coord])
    c_arr = np.random.rand(len(vertices), 3)
    random.shuffle(vertices)
    vertices = np.array(vertices)
    tri = Delaunay(vertices[:, [0, 1]])
    faces = tri.simplices
    c_loop = [None, c_arr]
    f_loop = [None, faces]
    s_loop = [None, "butterfly", "loop"]
    for smooth_type in s_loop:
        for face in f_loop:
            for color in c_loop:
                surface_actor = actor.surface(vertices, faces=face,
                                              colors=color, smooth=smooth_type)
    return surface_actor


@pytest.mark.skipif(not VTK_9_PLUS, reason="Requires VTK >= 9.0.0")
def test_manifest_pbr(interactive=False):
    scene = window.Scene()  # Setup scene

    # Setup surface
    surface_actor = _generate_surface()
    material.manifest_pbr(surface_actor)
    scene.add(surface_actor)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)

    scene.clear()  # Reset scene

    # Contour from roi setup
    data = np.zeros((50, 50, 50))
    data[20:30, 25, 25] = 1.
    data[25, 20:30, 25] = 1.
    affine = np.eye(4)
    surface = actor.contour_from_roi(data, affine, color=np.array([1, 0, 1]))
    material.manifest_pbr(surface)
    scene.add(surface)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)

    scene.clear()  # Reset scene

    # Streamtube setup
    data1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    data2 = data1 + np.array([0.5, 0., 0.])
    data = [data1, data2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    tubes = actor.streamtube(data, colors, linewidth=.1)
    material.manifest_pbr(tubes)
    scene.add(tubes)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 2)

    scene.clear()  # Reset scene

    # Axes setup
    axes = actor.axes()
    material.manifest_pbr(axes)
    scene.add(axes)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)

    scene.clear()  # Reset scene

    # ODF slicer setup
    if have_dipy:
        from dipy.data import get_sphere
        from tempfile import mkstemp
        sphere = get_sphere('symmetric362')
        shape = (11, 11, 11, sphere.vertices.shape[0])
        fid, fname = mkstemp(suffix='_odf_slicer.mmap')
        odfs = np.memmap(fname, dtype=np.float64, mode='w+', shape=shape)
        odfs[:] = 1
        affine = np.eye(4)
        mask = np.ones(odfs.shape[:3])
        mask[:4, :4, :4] = 0
        odfs[..., 0] = 1
        odf_actor = actor.odf_slicer(odfs, affine, mask=mask, sphere=sphere,
                                     scale=.25, colormap='blues')
        material.manifest_pbr(odf_actor)
        k = 5
        I, J, _ = odfs.shape[:3]
        odf_actor.display_extent(0, I, 0, J, k, k)
        odf_actor.GetProperty().SetOpacity(1.0)
        scene.add(odf_actor)
        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 11 * 11)

    scene.clear()  # Reset scene

    # Tensor slicer setup
    if have_dipy:
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')
        evals = np.array([1.4, .35, .35]) * 10 ** (-3)
        evecs = np.eye(3)
        mevals = np.zeros((3, 2, 4, 3))
        mevecs = np.zeros((3, 2, 4, 3, 3))
        mevals[..., :] = evals
        mevecs[..., :, :] = evecs
        affine = np.eye(4)
        scene = window.Scene()
        tensor_actor = actor.tensor_slicer(mevals, mevecs, affine=affine,
                                           sphere=sphere, scale=.3)
        material.manifest_pbr(tensor_actor)
        _, J, K = mevals.shape[:3]
        tensor_actor.display_extent(0, 1, 0, J, 0, K)
        scene.add(tensor_actor)
        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 4)
        # TODO: Rotate to test
        # npt.assert_equal(report.objects, 4 * 2 * 2)

    scene.clear()  # Reset scene

    # Point setup
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    opacity = 0.5
    points_actor = actor.point(points, colors, opacity=opacity)
    material.manifest_pbr(points_actor)
    scene.add(points_actor)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)

    scene.clear()  # Reset scene

    # Sphere setup
    xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.99]])
    opacity = 0.5
    sphere_actor = actor.sphere(centers=xyzr[:, :3], colors=colors[:],
                                radii=xyzr[:, 3], opacity=opacity)
    material.manifest_pbr(sphere_actor)
    scene.add(sphere_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)

    scene.clear()  # Reset scene

    # Advanced geometry actors setup (Arrow, cone, cylinder)
    xyz = np.array([[0, 0, 0], [50, 0, 0], [100, 0, 0]])
    dirs = np.array([[0, 1, 0], [1, 0, 0], [0, 0.5, 0.5]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [1, 1, 0, 1]])
    heights = np.array([5, 7, 10])
    actor_list = [[actor.cone, {'directions': dirs, 'resolution': 8}],
                  [actor.arrow, {'directions': dirs, 'resolution': 9}],
                  [actor.cylinder, {'directions': dirs}]]
    for act_func, extra_args in actor_list:
        aga_actor = act_func(centers=xyz, colors=colors[:], heights=heights,
                             **extra_args)
        material.manifest_pbr(aga_actor)
        scene.add(aga_actor)
        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 3)
        scene.clear()

    # Basic geometry actors (Box, cube, frustum, octagonalprism, rectangle,
    # square)
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [1, 2, (1, 1, 1), [3, 2, 1], np.array([1, 2, 3]),
                  np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])]
    actor_list = [[actor.box, {}], [actor.cube, {}], [actor.frustum, {}],
                  [actor.octagonalprism, {}], [actor.rectangle, {}],
                  [actor.square, {}]]
    for act_func, extra_args in actor_list:
        for scale in scale_list:
            scene = window.Scene()
            bga_actor = act_func(centers=centers, directions=directions,
                                 colors=colors, scales=scale, **extra_args)
            material.manifest_pbr(bga_actor)
            scene.add(bga_actor)
            arr = window.snapshot(scene)
            report = window.analyze_snapshot(arr)
            msg = 'Failed with {}, scale={}'.format(act_func.__name__, scale)
            npt.assert_equal(report.objects, 3, err_msg=msg)
            scene.clear()

    # Cone setup using vertices
    centers = np.array([[0, 0, 0], [20, 0, 0], [40, 0, 0]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.99]])
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                         [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    faces = np.array([[0, 1, 3], [0, 1, 2]])
    cone_actor = actor.cone(centers=centers, directions=directions,
                            colors=colors[:], vertices=vertices, faces=faces)
    material.manifest_pbr(cone_actor)
    scene.add(cone_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)

    scene.clear()  # Reset scene

    # Superquadric setup
    centers = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = np.random.rand(3, 3)
    scales = [1, 2, 3]
    roundness = np.array([[1, 1], [1, 2], [2, 1]])
    sq_actor = actor.superquadric(centers, roundness=roundness,
                                  directions=directions,
                                  colors=colors.astype(np.uint8),
                                  scales=scales)
    material.manifest_pbr(sq_actor)
    scene.add(sq_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)

    scene.clear()  # Reset scene

    # Label setup
    text_actor = actor.label("Hello")
    material.manifest_pbr(text_actor)
    scene.add(text_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 5)

    # NOTE: From this point on, these actors don't have full support for PBR
    # interpolation. This is, the test passes but there is no evidence of the
    # desired effect.

    """
    # Line setup
    data1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    data2 = data1 + np.array([0.5, 0., 0.])
    data = [data1, data2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    lines = actor.line(data, colors, linewidth=5)
    material.manifest_pbr(lines)
    scene.add(lines)
    """

    """
    # Peak slicer setup
    _peak_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='f4')
    # peak_dirs.shape = (1, 1, 1) + peak_dirs.shape
    peak_dirs = np.zeros((11, 11, 11, 3, 3))
    peak_dirs[:, :, :] = _peak_dirs
    peak_actor = actor.peak_slicer(peak_dirs)
    material.manifest_pbr(peak_actor)
    scene.add(peak_actor)
    """

    """
    # Dots setup
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    dots_actor = actor.dots(points, color=(0, 255, 0))
    material.manifest_pbr(dots_actor)
    scene.add(dots_actor)
    """

    """
    # Texture setup
    arr = (255 * np.ones((512, 212, 4))).astype('uint8')
    arr[20:40, 20:40, :] = np.array([255, 0, 0, 255], dtype='uint8')
    tp2 = actor.texture(arr)
    material.manifest_pbr(tp2)
    scene.add(tp2)
    """

    """
    # Texture on sphere setup
    arr = 255 * np.ones((810, 1620, 3), dtype='uint8')
    rows, cols, _ = arr.shape
    rs = rows // 2
    cs = cols // 2
    w = 150 // 2
    arr[rs - w: rs + w, cs - 10 * w: cs + 10 * w] = np.array([255, 127, 0])
    tsa = actor.texture_on_sphere(arr)
    material.manifest_pbr(tsa)
    scene.add(tsa)
    """

    """
    # SDF setup
    centers = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]]) * 11
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    scales = [1, 2, 3]
    primitive = ['sphere', 'ellipsoid', 'torus']

    sdf_actor = actor.sdf(centers, directions=directions, colors=colors,
                          primitives=primitive, scales=scales)
    material.manifest_pbr(sdf_actor)
    scene.add(sdf_actor)
    """

    # NOTE: For these last set of actors, there is not support for PBR
    # interpolation at all.

    """
    # Setup slicer
    data = (255 * np.random.rand(50, 50, 50))
    affine = np.eye(4)
    slicer = actor.slicer(data, affine, value_range=[data.min(), data.max()])
    slicer.display(None, None, 25)
    material.manifest_pbr(slicer)
    scene.add(slicer)
    """

    """
    # Contour from label setup
    data = np.zeros((50, 50, 50))
    data[5:15, 1:10, 25] = 1.
    data[25:35, 1:10, 25] = 2.
    data[40:49, 1:10, 25] = 3.
    color = np.array([[255, 0, 0, 0.6],
                      [0, 255, 0, 0.5],
                      [0, 0, 255, 1.0]])
    surface = actor.contour_from_label(data, color=color)
    material.manifest_pbr(surface)
    scene.add(surface)
    """

    """
    # Scalar bar setup
    lut = actor.colormap_lookup_table(
        scale_range=(0., 100.), hue_range=(0., 0.1), saturation_range=(1, 1),
        value_range=(1., 1))
    sb_actor = actor.scalar_bar(lut, ' ')
    material.manifest_pbr(sb_actor)
    scene.add(sb_actor)
    """

    """
    # Billboard setup
    centers = np.array([[0, 0, 0], [5, -5, 5], [-7, 7, -7], [10, 10, 10],
                        [10.5, 11.5, 11.5], [12, -12, -12], [-17, 17, 17],
                        [-22, -22, 22]])
    colors = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1],
                       [1, 0, 0], [0, 1, 0], [0, 1, 1]])
    scales = [6, .4, 1.2, 1, .2, .7, 3, 2]
    """
    fake_sphere = \
        """
        float len = length(point);
        float radius = 1.;
        if(len > radius)
            discard;
        vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
        vec3 direction = normalize(vec3(1., 1., 1.));
        float df_1 = max(0, dot(direction, normalizedPoint));
        float sf_1 = pow(df_1, 24);
        fragOutput0 = vec4(max(df_1 * color, sf_1 * vec3(1)), 1);
        """
    """
    billboard_actor = actor.billboard(centers, colors=colors, scales=scales,
                                      fs_impl=fake_sphere)
    material.manifest_pbr(billboard_actor)
    scene.add(billboard_actor)
    """

    """
    # Text3D setup
    msg = 'I \nlove\n FURY'
    txt_actor = actor.text_3d(msg)
    material.manifest_pbr(txt_actor)
    scene.add(txt_actor)
    """

    """
    # Figure setup
    arr = (255 * np.ones((512, 212, 4))).astype('uint8')
    arr[20:40, 20:40, 3] = 0
    tp = actor.figure(arr)
    material.manifest_pbr(tp)
    scene.add(tp)
    """

    if interactive:
        window.show(scene)


def test_manifest_standard(interactive=False):
    scene = window.Scene()  # Setup scene

    # Setup surface
    surface_actor = _generate_surface()
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(surface_actor)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)

    scene.clear()  # Reset scene

    # Contour from roi setup
    data = np.zeros((50, 50, 50))
    data[20:30, 25, 25] = 1.
    data[25, 20:30, 25] = 1.
    affine = np.eye(4)
    surface = actor.contour_from_roi(data, affine, color=np.array([1, 0, 1]))
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(surface)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)

    scene.clear()  # Reset scene

    # Contour from label setup
    data = np.zeros((50, 50, 50))
    data[5:15, 1:10, 25] = 1.
    data[25:35, 1:10, 25] = 2.
    data[40:49, 1:10, 25] = 3.
    color = np.array([[255, 0, 0],
                      [0, 255, 0],
                      [0, 0, 255]])
    surface = actor.contour_from_label(data, color=color)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(surface)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)

    scene.clear()  # Reset scene

    # Streamtube setup
    data1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    data2 = data1 + np.array([0.5, 0., 0.])
    data = [data1, data2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    tubes = actor.streamtube(data, colors, linewidth=.1)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(tubes)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 2)

    scene.clear()  # Reset scene

    # ODF slicer setup
    if have_dipy:
        from dipy.data import get_sphere
        from tempfile import mkstemp
        sphere = get_sphere('symmetric362')
        shape = (11, 11, 11, sphere.vertices.shape[0])
        fid, fname = mkstemp(suffix='_odf_slicer.mmap')
        odfs = np.memmap(fname, dtype=np.float64, mode='w+', shape=shape)
        odfs[:] = 1
        affine = np.eye(4)
        mask = np.ones(odfs.shape[:3])
        mask[:4, :4, :4] = 0
        odfs[..., 0] = 1
        odf_actor = actor.odf_slicer(odfs, affine, mask=mask, sphere=sphere,
                                     scale=.25, colormap='blues')
        material.manifest_standard(surface_actor, ambient_level=.3,
                                   diffuse_level=.25)
        k = 5
        I, J, _ = odfs.shape[:3]
        odf_actor.display_extent(0, I, 0, J, k, k)
        odf_actor.GetProperty().SetOpacity(1.0)
        scene.add(odf_actor)
        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 11 * 11)

    scene.clear()  # Reset scene

    # Tensor slicer setup
    if have_dipy:
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')
        evals = np.array([1.4, .35, .35]) * 10 ** (-3)
        evecs = np.eye(3)
        mevals = np.zeros((3, 2, 4, 3))
        mevecs = np.zeros((3, 2, 4, 3, 3))
        mevals[..., :] = evals
        mevecs[..., :, :] = evecs
        affine = np.eye(4)
        scene = window.Scene()
        tensor_actor = actor.tensor_slicer(mevals, mevecs, affine=affine,
                                           sphere=sphere, scale=.3)
        material.manifest_standard(surface_actor, ambient_level=.3,
                                   diffuse_level=.25)
        _, J, K = mevals.shape[:3]
        tensor_actor.display_extent(0, 1, 0, J, 0, K)
        scene.add(tensor_actor)
        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 4)

    scene.clear()  # Reset scene

    # Point setup
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    opacity = 0.5
    points_actor = actor.point(points, colors, opacity=opacity)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(points_actor)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)

    scene.clear()  # Reset scene

    # Sphere setup
    xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.99]])
    opacity = 0.5
    sphere_actor = actor.sphere(centers=xyzr[:, :3], colors=colors[:],
                                radii=xyzr[:, 3], opacity=opacity)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(sphere_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)

    scene.clear()  # Reset scene

    # Advanced geometry actors setup (Arrow, cone, cylinder)
    xyz = np.array([[0, 0, 0], [50, 0, 0], [100, 0, 0]])
    dirs = np.array([[0, 1, 0], [1, 0, 0], [0, 0.5, 0.5]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [1, 1, 0, 1]])
    heights = np.array([5, 7, 10])
    actor_list = [[actor.cone, {'directions': dirs, 'resolution': 8}],
                  [actor.arrow, {'directions': dirs, 'resolution': 9}],
                  [actor.cylinder, {'directions': dirs}]]
    for act_func, extra_args in actor_list:
        aga_actor = act_func(centers=xyz, colors=colors[:], heights=heights,
                             **extra_args)
        material.manifest_standard(surface_actor, ambient_level=.3,
                                   diffuse_level=.25)
        scene.add(aga_actor)
        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 3)
        scene.clear()

    # Basic geometry actors (Box, cube, frustum, octagonalprism, rectangle,
    # square)
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [1, 2, (1, 1, 1), [3, 2, 1], np.array([1, 2, 3]),
                  np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])]
    actor_list = [[actor.box, {}], [actor.cube, {}], [actor.frustum, {}],
                  [actor.octagonalprism, {}], [actor.rectangle, {}],
                  [actor.square, {}]]
    for act_func, extra_args in actor_list:
        for scale in scale_list:
            scene = window.Scene()
            bga_actor = act_func(centers=centers, directions=directions,
                                 colors=colors, scales=scale, **extra_args)
            material.manifest_standard(surface_actor, ambient_level=.3,
                                       diffuse_level=.25)
            scene.add(bga_actor)
            arr = window.snapshot(scene)
            report = window.analyze_snapshot(arr)
            msg = 'Failed with {}, scale={}'.format(act_func.__name__, scale)
            npt.assert_equal(report.objects, 3, err_msg=msg)
            scene.clear()

    # Cone setup using vertices
    centers = np.array([[0, 0, 0], [20, 0, 0], [40, 0, 0]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.99]])
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                         [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    faces = np.array([[0, 1, 3], [0, 1, 2]])
    cone_actor = actor.cone(centers=centers, directions=directions,
                            colors=colors[:], vertices=vertices, faces=faces)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(cone_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)

    scene.clear()  # Reset scene

    # Superquadric setup
    centers = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = np.random.rand(3, 3)
    scales = [1, 2, 3]
    roundness = np.array([[1, 1], [1, 2], [2, 1]])
    sq_actor = actor.superquadric(centers, roundness=roundness,
                                  directions=directions,
                                  colors=colors.astype(np.uint8),
                                  scales=scales)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(sq_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    ft.assert_greater_equal(report.objects, 3)

    scene.clear()  # Reset scene

    # Label setup
    text_actor = actor.label("Hello")
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(text_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 5)

    scene.clear()  # Reset scene

    # Texture setup
    arr = (255 * np.ones((512, 212, 4))).astype('uint8')
    arr[20:40, 20:40, :] = np.array([255, 0, 0, 255], dtype='uint8')
    tp2 = actor.texture(arr)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(tp2)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)

    scene.clear()  # Reset scene

    # Texture on sphere setup
    arr = 255 * np.ones((810, 1620, 3), dtype='uint8')
    rows, cols, _ = arr.shape
    rs = rows // 2
    cs = cols // 2
    w = 150 // 2
    arr[rs - w: rs + w, cs - 10 * w: cs + 10 * w] = np.array([255, 127, 0])
    tsa = actor.texture_on_sphere(arr)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(tsa)
    scene.reset_camera()
    scene.reset_clipping_range()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)

    # NOTE: From this point on, these actors don't have full support for PBR
    # interpolation. This is, the test passes but there is no evidence of the
    # desired effect.

    """
    # Setup slicer
    data = (255 * np.random.rand(50, 50, 50))
    affine = np.eye(4)
    slicer = actor.slicer(data, affine, value_range=[data.min(), data.max()])
    slicer.display(None, None, 25)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(slicer)
    """

    """
    # Line setup
    data1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    data2 = data1 + np.array([0.5, 0., 0.])
    data = [data1, data2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    lines = actor.line(data, colors, linewidth=5)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(lines)
    """

    """
    # Scalar bar setup
    lut = actor.colormap_lookup_table(
        scale_range=(0., 100.), hue_range=(0., 0.1), saturation_range=(1, 1),
        value_range=(1., 1))
    sb_actor = actor.scalar_bar(lut, ' ')
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(sb_actor)
    """

    """
    # Axes setup
    axes = actor.axes()
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(axes)
    """

    """
    # Peak slicer setup
    _peak_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='f4')
    # peak_dirs.shape = (1, 1, 1) + peak_dirs.shape
    peak_dirs = np.zeros((11, 11, 11, 3, 3))
    peak_dirs[:, :, :] = _peak_dirs
    peak_actor = actor.peak_slicer(peak_dirs)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(peak_actor)
    """

    """
    # Dots setup
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    dots_actor = actor.dots(points, color=(0, 255, 0))
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(dots_actor)
    """

    """
    # Text3D setup
    msg = 'I \nlove\n FURY'
    txt_actor = actor.text_3d(msg)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(txt_actor)
    """

    """
    # Figure setup
    arr = (255 * np.ones((512, 212, 4))).astype('uint8')
    arr[20:40, 20:40, 3] = 0
    tp = actor.figure(arr)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(tp)
    """

    """
    # SDF setup
    centers = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]]) * 11
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    scales = [1, 2, 3]
    primitive = ['sphere', 'ellipsoid', 'torus']

    sdf_actor = actor.sdf(centers, directions=directions, colors=colors,
                          primitives=primitive, scales=scales)
    material.manifest_standard(surface_actor, ambient_level=.3,
                               diffuse_level=.25)
    scene.add(sdf_actor)
    """

    # NOTE: For these last set of actors, there is not support for PBR
    # interpolation at all.

    """
    # Billboard setup
    centers = np.array([[0, 0, 0], [5, -5, 5], [-7, 7, -7], [10, 10, 10],
                        [10.5, 11.5, 11.5], [12, -12, -12], [-17, 17, 17],
                        [-22, -22, 22]])
    colors = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1],
                       [1, 0, 0], [0, 1, 0], [0, 1, 1]])
    scales = [6, .4, 1.2, 1, .2, .7, 3, 2]
    """
    fake_sphere = \
        """
        float len = length(point);
        float radius = 1.;
        if (len > radius)
            discard;
        vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
        vec3 direction = normalize(vec3(1., 1., 1.));
        float df_1 = max(0, dot(direction, normalizedPoint));
        float sf_1 = pow(df_1, 24);
        fragOutput0 = vec4(max(df_1 * color, sf_1 * vec3(1)), 1);
        """
    """
    billboard_actor = actor.billboard(centers, colors=colors, scales=scales,
                                      fs_impl=fake_sphere)
    material.manifest_pbr(billboard_actor)
    scene.add(billboard_actor)
    """

    if interactive:
        window.show(scene)
