import os
from os.path import join as pjoin
import numpy as np
import numpy.testing as npt
import vtk
from fury.io import load_polydata, save_polydata, load_image, save_image
from fury.utils import vtk, numpy_support, numpy_to_vtk_points
from fury.tmpdirs import InTemporaryDirectory
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

    l_ext = ["stl", "obj"]
    l_options = [{}, {'is_mni_obj': True, }]
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
    l_ext = ["png", "jpeg", "jpg", "bmp", "tiff", "tif"]
    fname = "temp-io"

    for ext in l_ext:
        with InTemporaryDirectory() as odir:

            data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
            fname_path = pjoin(odir, "{0}.{1}".format(fname, ext))

            save_image(data, fname_path)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

            out_image = load_image(fname_path)

            # import ipdb;ipdb.set_trace()
            if ext not in ["jpeg", "jpg", "tiff", "tif"]:
                npt.assert_array_equal(data, out_image[..., 0])
            else:
                npt.assert_array_almost_equal(data, out_image[..., 0],
                                              decimal=0)

    npt.assert_raises(IOError, load_image, "test.vtk")
    npt.assert_raises(IOError, save_image, np.random.randint(0, 255,
                                                             size=(50, 3)),
                      "test.vtk")
    npt.assert_raises(IOError, save_image,
                      np.random.randint(0, 255, size=(50, 3, 1, 1)),
                      "test.png")

    compression_type = [None, "lzw"]
    for ct in compression_type:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
            fname_path = pjoin(odir, "{0}.tif".format(fname))

            save_image(data, fname_path, compression_type=ct)
            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)


if __name__ == "__main__":
    npt.run_module_suite()
