import os
from os.path import join as pjoin
from tempfile import TemporaryDirectory as InTemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest
from PIL import Image

from fury.decorators import skip_osx
from fury.io import (
    load_cubemap_texture,
    load_image,
    load_polydata,
    load_sprite_sheet,
    load_text,
    save_image,
    save_polydata,
)
from fury.lib import ImageData, PolyData, numpy_support
from fury.testing import assert_greater
from fury.utils import numpy_to_vtk_points


def test_save_and_load_polydata():
    l_ext = ['vtk', 'fib', 'ply', 'xml']
    fname = 'temp-io'

    for ext in l_ext:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3))

            pd = PolyData()
            pd.SetPoints(numpy_to_vtk_points(data))

            fname_path = pjoin(odir, '{0}.{1}'.format(fname, ext))
            save_polydata(pd, fname_path)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

            out_pd = load_polydata(fname_path)
            out_data = numpy_support.vtk_to_numpy(out_pd.GetPoints().GetData())

            npt.assert_array_equal(data, out_data)

    npt.assert_raises(IOError, save_polydata, PolyData(), 'test.vti')
    npt.assert_raises(IOError, save_polydata, PolyData(), 'test.obj')
    npt.assert_raises(IOError, load_polydata, 'test.vti')
    npt.assert_raises(FileNotFoundError, load_polydata, 'does-not-exist.obj')


def test_save_and_load_options():
    l_ext = ['ply', 'vtk']
    l_options = [
        {
            'color_array_name': 'horizon',
        },
        {
            'binary': True,
        },
    ]
    fname = 'temp-io'

    for ext, option in zip(l_ext, l_options):
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3))

            pd = PolyData()
            pd.SetPoints(numpy_to_vtk_points(data))

            fname_path = pjoin(odir, '{0}.{1}'.format(fname, ext))
            save_polydata(pd, fname_path, **option)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

            out_pd = load_polydata(fname_path)
            out_data = numpy_support.vtk_to_numpy(out_pd.GetPoints().GetData())

            npt.assert_array_equal(data, out_data)

    l_ext = ['vtk', 'vtp', 'ply', 'stl', 'mni.obj']
    l_options = [
        {},
        {
            'binary': False,
        },
    ]
    for ext, option in zip(l_ext, l_options):
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3))

            pd = PolyData()
            pd.SetPoints(numpy_to_vtk_points(data))

            fname_path = pjoin(odir, '{0}.{1}'.format(fname, ext))
            save_polydata(pd, fname_path, **option)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)


def test_save_load_image():
    l_ext = ['png', 'jpeg', 'jpg', 'bmp', 'tiff']
    fury_logo_link = (
        'https://raw.githubusercontent.com/fury-gl/'
        'fury-communication-assets/main/fury-logo.png'
    )

    invalid_link = 'https://picsum.photos/200'
    fname = 'temp-io'

    for ext in l_ext:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
            url_image = load_image(fury_logo_link)

            url_fname_path = pjoin(odir, f'fury_logo.{ext}')
            fname_path = pjoin(odir, '{0}.{1}'.format(fname, ext))

            save_image(data, fname_path, compression_quality=100)
            save_image(url_image, url_fname_path, compression_quality=100)

            npt.assert_equal(os.path.isfile(fname_path), True)
            npt.assert_equal(os.path.isfile(url_fname_path), True)

            assert_greater(os.stat(fname_path).st_size, 0)
            assert_greater(os.stat(url_fname_path).st_size, 0)

            out_image = load_image(fname_path)
            if ext not in ['jpeg', 'jpg', 'tiff']:
                npt.assert_array_equal(data[..., 0], out_image[..., 0])
            else:

                npt.assert_array_almost_equal(
                    data[..., 0], out_image[..., 0], decimal=0
                )

    npt.assert_raises(IOError, load_image, invalid_link)
    npt.assert_raises(IOError, load_image, 'test.vtk')
    npt.assert_raises(IOError, load_image, 'test.vtk', use_pillow=False)
    npt.assert_raises(
        IOError, save_image, np.random.randint(0, 255, size=(50, 3)), 'test.vtk'
    )
    npt.assert_raises(
        IOError,
        save_image,
        np.random.randint(0, 255, size=(50, 3)),
        'test.vtk',
        use_pillow=False,
    )
    npt.assert_raises(
        IOError, save_image, np.random.randint(0, 255, size=(50, 3, 1, 1)), 'test.png'
    )

    compression_type = [None, 'bits', 'random']

    for ct in compression_type:
        with InTemporaryDirectory() as odir:
            try:
                data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
                fname_path = pjoin(odir, '{0}.tif'.format(fname))

                save_image(data, fname_path, compression_type=ct, use_pillow=False)
                npt.assert_equal(os.path.isfile(fname_path), True)
                assert_greater(os.stat(fname_path).st_size, 0)
            except OSError:
                continue


@pytest.mark.skipif(
    skip_osx,
    reason='This test does not work on OSX due to '
    'libpng version conflict. Need to be '
    'introspected on Travis',
)
def test_pillow():

    with InTemporaryDirectory() as odir:
        data = (255 * np.random.rand(400, 255, 4)).astype(np.uint8)
        fname_path = pjoin(odir, 'test.png')

        for opt1, opt2 in [(True, True), (False, True), (True, False), (False, False)]:
            if not opt1:
                with pytest.warns(UserWarning):
                    save_image(data, fname_path, use_pillow=opt1)
            else:
                save_image(data, fname_path, use_pillow=opt1)
            data2 = load_image(fname_path, use_pillow=opt2)
            npt.assert_array_almost_equal(data, data2)
            npt.assert_equal(data.dtype, data2.dtype)

        dpi_tolerance = 0.01

        save_image(data, fname_path, use_pillow=True)
        img_dpi = Image.open(fname_path).info.get('dpi')
        assert abs(72 - img_dpi[0]) < dpi_tolerance
        assert abs(72 - img_dpi[1]) < dpi_tolerance

        save_image(data, fname_path, use_pillow=True, dpi=300)
        img_dpi = Image.open(fname_path).info.get('dpi')
        assert abs(300 - img_dpi[0]) < dpi_tolerance
        assert abs(300 - img_dpi[1]) < dpi_tolerance

        save_image(data, fname_path, use_pillow=True, dpi=(45, 45))
        img_dpi = Image.open(fname_path).info.get('dpi')
        assert abs(45 - img_dpi[0]) < dpi_tolerance
        assert abs(45 - img_dpi[1]) < dpi_tolerance

        save_image(data, fname_path, use_pillow=True, dpi=(300, 72))
        img_dpi = Image.open(fname_path).info.get('dpi')
        assert abs(300 - img_dpi[0]) < dpi_tolerance
        assert abs(72 - img_dpi[1]) < dpi_tolerance


def test_load_cubemap_texture():
    l_ext = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    for ext in l_ext:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)
            fname_path = pjoin(odir, f'test.{ext}')
            save_image(data, fname_path)

            fnames = [fname_path] * 5
            npt.assert_raises(IOError, load_cubemap_texture, fnames)

            fnames = [fname_path] * 6
            texture = load_cubemap_texture(fnames)
            npt.assert_equal(texture.GetCubeMap(), True)
            npt.assert_equal(texture.GetMipmap(), True)
            npt.assert_equal(texture.GetInterpolate(), 1)
            npt.assert_equal(texture.GetNumberOfInputPorts(), 6)
            npt.assert_equal(
                texture.GetInputDataObject(0, 0).GetDimensions(), (50, 50, 1)
            )

            fnames = [fname_path] * 7
            npt.assert_raises(IOError, load_cubemap_texture, fnames)


def test_load_sprite_sheet():
    sprite_URL = (
        'https://raw.githubusercontent.com/'
        'fury-gl/fury-data/master/unittests/fury_sprite.png'
    )

    with InTemporaryDirectory() as tdir:
        sprites = load_sprite_sheet(sprite_URL, 5, 5)

        for idx, sprite in enumerate(list(sprites.values())):
            img_name = f'{idx}.png'
            save_image(sprite, os.path.join(tdir, img_name))

        sprite_count = len(os.listdir(tdir))
        npt.assert_equal(sprite_count, 25)

        vtktype_sprites = load_sprite_sheet(sprite_URL, 5, 5, as_vtktype=True)

        for vtk_sprite in list(vtktype_sprites.values()):
            npt.assert_equal(isinstance(vtk_sprite, ImageData), True)


def test_load_text():
    with InTemporaryDirectory() as tdir:
        test_file_name = 'test.txt'

        # Test file does not exist
        npt.assert_raises(IOError, load_text, test_file_name)

        # Saving file with content
        test_file_contents = 'This is some test text.'
        test_fname = os.path.join(tdir, test_file_name)
        test_file = open(test_fname, 'w')
        test_file.write(test_file_contents)
        test_file.close()

        npt.assert_string_equal(load_text(test_fname), test_file_contents)
