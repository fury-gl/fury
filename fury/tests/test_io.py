import os
from os.path import join as pjoin
from tempfile import TemporaryDirectory as InTemporaryDirectory

# from PIL import Image
import numpy as np
import numpy.testing as npt

# import pytest
import polyxios as px

# from fury.decorators import skip_osx
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import (
    get_extension,
    load_cube_map_texture,
    load_image,
    load_image_as_wgpu_texture_view,
    load_image_texture,
    load_network,
    # load_polydata,
    # load_sprite_sheet,
    # load_text,
    read_lines,
    read_mesh,
    read_points,
    save_image,
    save_network,
)

# save_polydata,
from fury.lib import Texture, wgpu
from fury.testing import assert_greater
from fury.window import ShowManager


def test_get_extension():
    npt.assert_equal(get_extension("image.png"), "png")
    npt.assert_equal(get_extension("archive.tar.gz"), "gz")
    npt.assert_equal(get_extension("/path/to/folder/file.jpg"), "jpg")
    npt.assert_equal(get_extension(".hidden"), "")
    npt.assert_equal(get_extension("noextension"), "")


def test_load_cube_map_texture():
    fetch_viz_cubemaps()
    texture_files = read_viz_cubemap("skybox")
    texture = load_cube_map_texture(texture_files)

    npt.assert_equal(type(texture), Texture)


def test_load_image_texture():
    fetch_viz_cubemaps()
    texture_files = read_viz_cubemap("skybox")
    texture = load_image_texture(texture_files[0])

    npt.assert_equal(type(texture), Texture)


def test_load_image_as_wgpu_texture_view_uses_show_manager_device():
    with InTemporaryDirectory() as tdir:
        data = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
        fname_path = pjoin(tdir, "temp.png")
        save_image(data, fname_path, compression_quality=100)

        show_m = ShowManager(window_type="offscreen")
        try:
            texture_view = load_image_as_wgpu_texture_view(fname_path, show_m.device)
        finally:
            show_m.close()

    assert isinstance(texture_view, wgpu.GPUTextureView)
    npt.assert_array_equal(texture_view.texture.size, (8, 8, 1))
    assert texture_view.texture._device is show_m.device


# def test_save_and_load_polydata():
#     l_ext = ["vtk", "fib", "ply", "xml"]
#     fname = "temp-io"

#     for ext in l_ext:
#         with InTemporaryDirectory() as odir:
#             data = np.random.randint(0, 255, size=(50, 3))

#             pd = PolyData()
#             pd.SetPoints(numpy_to_vtk_points(data))

#             fname_path = pjoin(odir, "{0}.{1}".format(fname, ext))
#             save_polydata(pd, fname_path)

#             npt.assert_equal(os.path.isfile(fname_path), True)
#             assert_greater(os.stat(fname_path).st_size, 0)

#             out_pd = load_polydata(fname_path)
#             out_data = numpy_support.vtk_to_numpy(out_pd.GetPoints().GetData())

#             npt.assert_array_equal(data, out_data)

#     npt.assert_raises(
#         IOError,
#         save_polydata,
#         PolyData(),
#         "test.vti",
#         binary=False,
#         color_array_name=None,
#     )
#     npt.assert_raises(
#         IOError,
#         save_polydata,
#         PolyData(),
#         "test.obj",
#         binary=False,
#         color_array_name=None,
#     )
#     npt.assert_raises(IOError, load_polydata, "test.vti")
#     npt.assert_raises(FileNotFoundError, load_polydata, "does-not-exist.obj")


# def test_save_and_load_options():
#     l_ext = ["ply", "vtk"]
#     l_options = [
#         {
#             "color_array_name": "horizon",
#         },
#         {
#             "binary": True,
#         },
#     ]
#     fname = "temp-io"

#     for ext, option in zip(l_ext, l_options):
#         with InTemporaryDirectory() as odir:
#             data = np.random.randint(0, 255, size=(50, 3))

#             pd = PolyData()
#             pd.SetPoints(numpy_to_vtk_points(data))

#             fname_path = pjoin(odir, "{0}.{1}".format(fname, ext))
#             save_polydata(pd, fname_path, **option)

#             npt.assert_equal(os.path.isfile(fname_path), True)
#             assert_greater(os.stat(fname_path).st_size, 0)

#             out_pd = load_polydata(fname_path)
#             out_data = numpy_support.vtk_to_numpy(out_pd.GetPoints().GetData())

#             npt.assert_array_equal(data, out_data)

#     l_ext = ["vtk", "vtp", "ply", "stl", "mni.obj"]
#     l_options = [
#         {},
#         {
#             "binary": False,
#         },
#     ]
#     for ext, option in zip(l_ext, l_options):
#         with InTemporaryDirectory() as odir:
#             data = np.random.randint(0, 255, size=(50, 3))

#             pd = PolyData()
#             pd.SetPoints(numpy_to_vtk_points(data))

#             fname_path = pjoin(odir, "{0}.{1}".format(fname, ext))
#             save_polydata(pd, fname_path, **option)

#             npt.assert_equal(os.path.isfile(fname_path), True)
#             assert_greater(os.stat(fname_path).st_size, 0)


def test_save_and_load_network():
    formats = ["gexf", "gml", "xnet"]
    fname = "temp-network"

    nodes_xyz = np.random.rand(10, 3).astype(np.float32)
    edges_indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    colors = np.random.rand(10, 4).astype(np.float32)
    network_data = (nodes_xyz, edges_indices, colors)

    for fmt in formats:
        with InTemporaryDirectory() as odir:
            fname_path = pjoin(odir, f"{fname}.{fmt}")

            save_network(network_data, fname_path)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

            out_data = load_network(fname_path)

            npt.assert_array_almost_equal(network_data[0], out_data[0], decimal=4)
            npt.assert_array_equal(network_data[1], out_data[1])
            npt.assert_array_almost_equal(network_data[2], out_data[2], decimal=2)

    # Test invalid format
    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, f"{fname}.invalid")
        npt.assert_raises(
            ValueError,
            save_network,
            network_data,
            fname_path,
            format="invalid",
        )
        with open(fname_path, "w") as f:
            f.write("invalid data")
        npt.assert_raises(
            ValueError,
            load_network,
            fname_path,
            format="invalid",
        )


def test_read_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)

    poly = px.make_polydata(
        vertices,
        [("triangle", faces)],
        vertex_attrs={"colors": colors},
    )

    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, "temp-mesh.vtk")
        px.write(poly, fname_path)

        out_vertices, out_faces, out_colors = read_mesh(fname_path)

    npt.assert_equal(out_vertices.dtype, np.float32)
    npt.assert_equal(out_faces.dtype, np.int32)
    npt.assert_array_almost_equal(out_vertices, vertices)
    npt.assert_array_equal(out_faces, faces)
    npt.assert_array_almost_equal(out_colors, colors)


def test_read_points():
    vertices = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    connectivity = np.array([[0], [1], [2], [3]], dtype=np.int32)

    poly = px.make_polydata(vertices, [("vertex", connectivity)])

    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, "temp-points.vtk")
        px.write(poly, fname_path)

        out_points, out_colors = read_points(fname_path)

    npt.assert_equal(out_points.dtype, np.float32)
    npt.assert_equal(out_points.shape, (4, 3))
    npt.assert_array_almost_equal(out_points, vertices)
    npt.assert_equal(out_colors, None)


def test_read_lines():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    line = np.array([[0, 1, 2, 3]], dtype=np.int32)

    poly = px.make_polydata(vertices, [("poly_line", line)])

    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, "temp-lines.vtk")
        px.write(poly, fname_path)

        out_lines, out_colors = read_lines(fname_path)

    npt.assert_equal(len(out_lines), 1)
    npt.assert_equal(out_lines[0].dtype, np.float32)
    npt.assert_array_almost_equal(out_lines[0], vertices)
    npt.assert_equal(out_colors, None)


def test_read_lines_without_line_elements():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    poly = px.make_polydata(vertices, [("triangle", faces)])

    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, "temp-no-lines.vtk")
        px.write(poly, fname_path)

        out_lines, out_colors = read_lines(fname_path)

    npt.assert_equal(out_lines, [])
    npt.assert_equal(out_colors, None)


def test_save_load_image():
    l_ext = ["png", "jpeg", "jpg", "bmp", "tiff"]
    fury_logo_link = (
        "https://raw.githubusercontent.com/fury-gl/"
        "fury-communication-assets/main/fury-logo.png"
    )

    invalid_link = "https://picsum.photos/200"
    fname = "temp-io"

    for ext in l_ext:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
            url_image = load_image(fury_logo_link)

            url_fname_path = pjoin(odir, f"fury_logo.{ext}")
            fname_path = pjoin(odir, f"{fname}.{ext}")

            save_image(data, fname_path, compression_quality=100)
            save_image(url_image, url_fname_path, compression_quality=100)

            npt.assert_equal(os.path.isfile(fname_path), True)
            npt.assert_equal(os.path.isfile(url_fname_path), True)

            assert_greater(os.stat(fname_path).st_size, 0)
            assert_greater(os.stat(url_fname_path).st_size, 0)

            out_image = load_image(fname_path)
            if ext not in ["jpeg", "jpg", "tiff"]:
                npt.assert_array_equal(data[..., 0], out_image[..., 0])
            else:
                npt.assert_array_almost_equal(
                    data[..., 0], out_image[..., 0], decimal=0
                )

    npt.assert_raises(
        IOError,
        load_image,
        invalid_link,
    )
    npt.assert_raises(
        IOError,
        load_image,
        "test.vtk",
    )
    npt.assert_raises(
        IOError,
        save_image,
        np.random.randint(0, 255, size=(50, 3)),
        "test.vtk",
        compression_quality=75,
        compression_type="deflation",
        dpi=(72, 72),
    )
    npt.assert_raises(
        IOError,
        save_image,
        np.random.randint(0, 255, size=(50, 3)),
        "test.vtk",
        compression_quality=75,
        compression_type="deflation",
        dpi=(72, 72),
    )
    npt.assert_raises(
        IOError,
        save_image,
        np.random.randint(0, 255, size=(50, 3, 1, 1)),
        "test.png",
        compression_quality=75,
        compression_type="deflation",
        dpi=(72, 72),
    )

    # Test valid TIFF compression types produce files
    for ct in [None, "lzw", "deflation"]:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
            fname_path = pjoin(odir, f"{fname}.tif")

            save_image(data, fname_path, compression_type=ct)
            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

    # Test that compression actually changes file output
    with InTemporaryDirectory() as odir:
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        path_none = pjoin(odir, "none.tiff")
        path_lzw = pjoin(odir, "lzw.tiff")
        path_deflate = pjoin(odir, "deflate.tiff")

        save_image(data, path_none, compression_type=None)
        save_image(data, path_lzw, compression_type="lzw")
        save_image(data, path_deflate, compression_type="deflation")

        size_none = os.stat(path_none).st_size
        size_lzw = os.stat(path_lzw).st_size
        size_deflate = os.stat(path_deflate).st_size

        # Compressed files should be smaller than uncompressed for uniform data
        assert size_lzw < size_none
        assert size_deflate < size_none

    # Test invalid compression type raises OSError
    npt.assert_raises(
        OSError,
        save_image,
        np.random.randint(0, 255, size=(50, 3), dtype=np.uint8),
        "test.tiff",
        compression_type="invalid",
    )


# @pytest.mark.skipif(
#     skip_osx,
#     reason="This test does not work on OSX due to "
#     "libpng version conflict. Need to be "
#     "introspected on Travis",
# )
# def test_pillow():
#     with InTemporaryDirectory() as odir:
#         data = (255 * np.random.rand(400, 255, 4)).astype(np.uint8)
#         fname_path = pjoin(odir, "test.png")

#         for opt1, opt2 in [
# (True, True), (False, True), (True, False), (False, False)]:
#             if not opt1:
#                 with pytest.warns(UserWarning):
#                     save_image(data, fname_path, use_pillow=opt1)
#             else:
#                 save_image(data, fname_path, use_pillow=opt1)
#             data2 = load_image(fname_path, use_pillow=opt2)
#             npt.assert_array_almost_equal(data, data2)
#             npt.assert_equal(data.dtype, data2.dtype)

#         dpi_tolerance = 0.01

#         save_image(data, fname_path, use_pillow=True)
#         img_dpi = Image.open(fname_path).info.get("dpi")
#         assert abs(72 - img_dpi[0]) < dpi_tolerance
#         assert abs(72 - img_dpi[1]) < dpi_tolerance

#         save_image(data, fname_path, use_pillow=True, dpi=300)
#         img_dpi = Image.open(fname_path).info.get("dpi")
#         assert abs(300 - img_dpi[0]) < dpi_tolerance
#         assert abs(300 - img_dpi[1]) < dpi_tolerance

#         save_image(data, fname_path, use_pillow=True, dpi=(45, 45))
#         img_dpi = Image.open(fname_path).info.get("dpi")
#         assert abs(45 - img_dpi[0]) < dpi_tolerance
#         assert abs(45 - img_dpi[1]) < dpi_tolerance

#         save_image(data, fname_path, use_pillow=True, dpi=(300, 72))
#         img_dpi = Image.open(fname_path).info.get("dpi")
#         assert abs(300 - img_dpi[0]) < dpi_tolerance
#         assert abs(72 - img_dpi[1]) < dpi_tolerance


# def test_load_sprite_sheet():
#     sprite_URL = (
#         "https://raw.githubusercontent.com/"
#         "fury-gl/fury-data/master/unittests/fury_sprite.png"
#     )

#     with InTemporaryDirectory() as tdir:
#         sprites = load_sprite_sheet(sprite_URL, 5, 5)

#         for idx, sprite in enumerate(list(sprites.values())):
#             img_name = f"{idx}.png"
#             save_image(sprite, os.path.join(tdir, img_name))

#         sprite_count = len(os.listdir(tdir))
#         npt.assert_equal(sprite_count, 25)

#         vtktype_sprites = load_sprite_sheet(sprite_URL, 5, 5, as_vtktype=True)

#         for vtk_sprite in list(vtktype_sprites.values()):
#             npt.assert_equal(isinstance(vtk_sprite, ImageData), True)


# def test_load_text():
#     with InTemporaryDirectory() as tdir:
#         test_file_name = "test.txt"

#         # Test file does not exist
#         npt.assert_raises(IOError, load_text, test_file_name)

#         # Saving file with content
#         test_file_contents = "This is some test text."
#         test_fname = os.path.join(tdir, test_file_name)
#         test_file = open(test_fname, "w")
#         test_file.write(test_file_contents)
#         test_file.close()

#         npt.assert_string_equal(load_text(test_fname), test_file_contents)
