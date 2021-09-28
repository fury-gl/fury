import os
from os.path import join as pjoin
from tempfile import TemporaryDirectory as InTemporaryDirectory
import numpy as np
import numpy.testing as npt
import pytest

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
    fname = "temp-io"

    for ext in l_ext:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
            fname_path = pjoin(odir, "{0}.{1}".format(fname, ext))

            save_image(data, fname_path, compression_quality=100)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

            out_image = load_image(fname_path)
            if ext not in ["jpeg", "jpg", "tiff"]:
                npt.assert_array_equal(data[..., 0], out_image[..., 0])
            else:

                npt.assert_array_almost_equal(data[..., 0], out_image[..., 0],
                                              decimal=0)

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
