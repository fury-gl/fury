import os
from os.path import join as pjoin
import itertools
from tempfile import TemporaryDirectory as InTemporaryDirectory
import numpy as np
import numpy.testing as npt
import pytest
from PIL import Image, ImageDraw
from fury import window, actor, ui

from fury.decorators import skip_osx
from fury.io import load_polydata, save_polydata, load_image, save_image
from fury.utils import vtk, numpy_support, numpy_to_vtk_points
from fury.testing import assert_greater


def test_save_and_load_polydata():
    l_ext = ["vtk", "fib", "ply", "xml"]
    fname = "temp-io"

    for ext in l_ext:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3))

            pd = vtk.vtkPolyData()
            pd.SetPoints(numpy_to_vtk_points(data))

            fname_path = pjoin(odir, "{0}.{1}".format(fname, ext))
            save_polydata(pd, fname_path)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

            out_pd = load_polydata(fname_path)
            out_data = numpy_support.vtk_to_numpy(out_pd.GetPoints().GetData())

            npt.assert_array_equal(data, out_data)

    npt.assert_raises(IOError, save_polydata, vtk.vtkPolyData(), "test.vti")
    npt.assert_raises(IOError, save_polydata, vtk.vtkPolyData(), "test.obj")
    npt.assert_raises(IOError, load_polydata, "test.vti")


def test_save_and_load_options():
    l_ext = ["ply", "vtk"]
    l_options = [{'color_array_name': 'horizon', }, {'binary': True, }]
    fname = "temp-io"

    for ext, option in zip(l_ext, l_options):
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3))

            pd = vtk.vtkPolyData()
            pd.SetPoints(numpy_to_vtk_points(data))

            fname_path = pjoin(odir, "{0}.{1}".format(fname, ext))
            save_polydata(pd, fname_path, **option)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

            out_pd = load_polydata(fname_path)
            out_data = numpy_support.vtk_to_numpy(out_pd.GetPoints().GetData())

            npt.assert_array_equal(data, out_data)

    l_ext = ["vtk", "vtp", "ply", "stl", "mni.obj"]
    l_options = [{}, {'binary': False, }]
    for ext, option in zip(l_ext, l_options):
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3))

            pd = vtk.vtkPolyData()
            pd.SetPoints(numpy_to_vtk_points(data))

            fname_path = pjoin(odir, "{0}.{1}".format(fname, ext))
            save_polydata(pd, fname_path, **option)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)


def test_save_load_image():
    l_ext = ["png", "jpeg", "jpg", "bmp", "tiff"]
    fury_logo_link = 'https://raw.githubusercontent.com/fury-gl/'\
                     'fury-communication-assets/main/fury-logo.png'
         
    invalid_link = 'https://picsum.photos/200'
    fname = "temp-io"

    for ext in l_ext:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
            url_image = load_image(fury_logo_link)

            url_fname_path = pjoin(odir, f'fury_logo.{ext}')
            fname_path = pjoin(odir, "{0}.{1}".format(fname, ext))

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

                npt.assert_array_almost_equal(data[..., 0], out_image[..., 0],
                                              decimal=0)

    npt.assert_raises(IOError, load_image, invalid_link)
    npt.assert_raises(IOError, load_image, "test.vtk")
    npt.assert_raises(IOError, load_image, "test.vtk", use_pillow=False)
    npt.assert_raises(IOError, save_image,
                      np.random.randint(0, 255, size=(50, 3)),
                      "test.vtk")
    npt.assert_raises(IOError, save_image,
                      np.random.randint(0, 255, size=(50, 3)),
                      "test.vtk", use_pillow=False)
    npt.assert_raises(IOError, save_image,
                      np.random.randint(0, 255, size=(50, 3, 1, 1)),
                      "test.png")

    compression_type = [None, "lzw"]

    for ct in compression_type:
        with InTemporaryDirectory() as odir:
            try:
                data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
                fname_path = pjoin(odir, "{0}.tif".format(fname))

                save_image(data, fname_path, compression_type=ct,
                           use_pillow=False)
                npt.assert_equal(os.path.isfile(fname_path), True)
                assert_greater(os.stat(fname_path).st_size, 0)
            except OSError:
                continue


@pytest.mark.skipif(skip_osx, reason="This test does not work on OSX due to "
                                     "libpng version conflict. Need to be "
                                     "introspected on Travis")
def test_pillow():

    with InTemporaryDirectory() as odir:
        data = (255 * np.random.rand(400, 255, 4)).astype(np.uint8)
        fname_path = pjoin(odir, "test.png")

        for opt1, opt2 in [(True, True), (False, True), (True, False),
                           (False, False)]:

            save_image(data, fname_path, use_pillow=opt1)
            data2 = load_image(fname_path, use_pillow=opt2)
            npt.assert_array_almost_equal(data, data2)
            npt.assert_equal(data.dtype, data2.dtype)

def test_frames_to_gif():
    xyz = 10 * np.random.rand(100, 3)
    colors = np.random.rand(100, 4)
    radii = np.random.rand(100) + 0.5
    images = []
    def execute_for_count_value(count_value):
        scene = window.Scene()
        sphere_actor = actor.sphere(centers=xyz,
                                    colors=colors,
                                    radii=radii)
        scene.add(sphere_actor)
        showm = window.ShowManager( scene,
                                    size=(900, 768),
                                    reset_camera=False,
                                    order_transparent=True)

        showm.initialize()

    tb = ui.TextBlock2D(bold=True)
    counter = itertools.count()

    def timer_callback(_obj, _event):
        cnt = next(counter)
        showm.scene.azimuth(0.05 * cnt)
        sphere_actor.GetProperty().SetOpacity(cnt/100.)
        showm.render()

        if cnt == count_value:
            showm.exit()

    scene.add(tb)

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)

    showm.start()

    # Save the image for each counter value
    window.record(scene,
                    out_path= str(count_value) + "img.png",
                    size=(900, 768))

    images.append(Image.open(str(count_value)+"img.png"))

    for t in range(1, 101, 10):
        execute_for_count_value(t)

    # Using `save` from PIL to convert the Series of Frames into a GIF Image.
    images[0].save('frames_to_gif.gif',
                    save_all=True,
                    append_images=images[1:],
                    optimize=False,
                    duration=0.005,
                    loop=0)

    gif = Image.open('Frames_to_gif.gif')
    try:
        gif.seek(1)
    except EOFError:
        isanimated = False
    else:
        isanimated = True

    npt.assert_equal(isanimated,True,
                    err_msg="Gif not created",
                    verbose=True)
