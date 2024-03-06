import os
import warnings
from tempfile import TemporaryDirectory as InTemporaryDirectory
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

from fury.lib import (
    BMPReader,
    BMPWriter,
    ImageData,
    ImageFlip,
    JPEGReader,
    JPEGWriter,
    MNIObjectReader,
    MNIObjectWriter,
    OBJReader,
    PLYReader,
    PLYWriter,
    PNGReader,
    PNGWriter,
    PolyDataReader,
    PolyDataWriter,
    STLReader,
    STLWriter,
    Texture,
    TIFFReader,
    TIFFWriter,
    XMLPolyDataReader,
    XMLPolyDataWriter,
    numpy_support,
)
from fury.utils import set_input


def load_cubemap_texture(fnames, interpolate_on=True, mipmap_on=True):
    """Load a cube map texture from a list of 6 images.

    Parameters
    ----------
    fnames : list of strings
        List of 6 filenames with bmp, jpg, jpeg, png, tif or tiff extensions.
    interpolate_on : bool, optional
    mipmap_on : bool, optional

    Returns
    -------
    output : vtkTexture
        Cube map texture.

    """
    if len(fnames) != 6:
        raise IOError('Expected 6 filenames, got {}'.format(len(fnames)))
    texture = Texture()
    texture.CubeMapOn()
    for idx, fn in enumerate(fnames):
        if not os.path.isfile(fn):
            raise FileNotFoundError(fn)
        else:
            # Read the images
            vtk_img = load_image(fn, as_vtktype=True)
            # Flip the image horizontally
            img_flip = ImageFlip()
            img_flip.SetInputData(vtk_img)
            img_flip.SetFilteredAxis(1)  # flip y axis
            img_flip.Update()
            # Add the image to the cube map
            texture.SetInputDataObject(idx, img_flip.GetOutput())
    if interpolate_on:
        texture.InterpolateOn()
    if mipmap_on:
        texture.MipmapOn()
    return texture


def load_image(filename, as_vtktype=False, use_pillow=True):
    """Load an image.

    Parameters
    ----------
    filename: str
        should be png, bmp, jpeg or jpg files
    as_vtktype: bool, optional
        if True, return vtk output otherwise an ndarray. Default False.
    use_pillow: bool, optional
        Use pillow python library to load the files. Default True

    Returns
    -------
    image: ndarray or vtk output
        desired image array

    """
    is_url = filename.lower().startswith('http://') or filename.lower().startswith(
        'https://'
    )

    if is_url:
        image_name = os.path.basename(filename)

        if len(image_name.split('.')) < 2:
            raise IOError(f'{filename} is not a valid image URL')

        urlretrieve(filename, image_name)
        filename = image_name

    if use_pillow:
        with Image.open(filename) as pil_image:
            if pil_image.mode in ['P']:
                pil_image = pil_image.convert('RGB')

            if pil_image.mode in ['RGBA', 'RGB', 'L']:
                image = np.asarray(pil_image)
            elif pil_image.mode.startswith('I;16'):
                raw = pil_image.tobytes('raw', pil_image.mode)
                dtype = '>u2' if pil_image.mode.endswith('B') else '<u2'
                image = np.frombuffer(raw, dtype=dtype)
                image.reshape(pil_image.size[::-1]).astype('=u2')
            else:
                try:
                    image = pil_image.convert('RGBA')
                except ValueError:
                    raise RuntimeError('Unknown image mode {}'.format(pil_image.mode))
                image = np.asarray(pil_image)

        if as_vtktype:
            if image.ndim not in [2, 3]:
                raise IOError('only 2D (L, RGB, RGBA) or 3D image available')

            vtk_image = ImageData()
            depth = 1 if image.ndim == 2 else image.shape[2]

            # width, height
            vtk_image.SetDimensions(image.shape[1], image.shape[0], depth)
            vtk_image.SetExtent(0, image.shape[1] - 1, 0, image.shape[0] - 1, 0, 0)
            vtk_image.SetSpacing(1.0, 1.0, 1.0)
            vtk_image.SetOrigin(0.0, 0.0, 0.0)

            image = np.flipud(image)
            image = image.reshape(image.shape[1] * image.shape[0], depth)
            image = np.ascontiguousarray(image, dtype=image.dtype)
            vtk_array_type = numpy_support.get_vtk_array_type(image.dtype)
            uchar_array = numpy_support.numpy_to_vtk(
                image, deep=True, array_type=vtk_array_type
            )
            vtk_image.GetPointData().SetScalars(uchar_array)
            image = vtk_image

        if is_url:
            os.remove(filename)
        return image

    d_reader = {
        '.png': PNGReader,
        '.bmp': BMPReader,
        '.jpeg': JPEGReader,
        '.jpg': JPEGReader,
        '.tiff': TIFFReader,
        '.tif': TIFFReader,
    }

    extension = os.path.splitext(os.path.basename(filename).lower())[1]

    if extension.lower() not in d_reader.keys():
        raise IOError(
            'Impossible to read the file {0}: Unknown extension {1}'.format(
                filename, extension
            )
        )

    reader = d_reader.get(extension)()
    reader.SetFileName(filename)
    reader.Update()
    reader.GetOutput().GetPointData().GetArray(0).SetName('original')

    if not as_vtktype:
        w, h, _ = reader.GetOutput().GetDimensions()
        vtk_array = reader.GetOutput().GetPointData().GetScalars()

        components = vtk_array.GetNumberOfComponents()
        image = numpy_support.vtk_to_numpy(vtk_array).reshape(h, w, components)
        image = np.flipud(image)

    if is_url:
        os.remove(filename)
    return reader.GetOutput() if as_vtktype else image


def load_text(file):
    """Load a text file.

    Parameters
    ----------
    file: str
        Path to the text file.

    Returns
    -------
    text: str
        Text contained in the file.

    """
    if not os.path.isfile(file):
        raise IOError('File {} does not exist.'.format(file))
    with open(file) as f:
        text = f.read()
    return text


def save_image(
    arr,
    filename,
    compression_quality=75,
    compression_type='deflation',
    use_pillow=True,
    dpi=(72, 72),
):
    """Save a 2d or 3d image.

    Expect an image with the following shape: (H, W) or (H, W, 1) or
    (H, W, 3) or (H, W, 4).

    Parameters
    ----------
    arr : ndarray
        array to save
    filename : string
        should be png, bmp, jpeg or jpg files
    compression_quality : int, optional
        compression_quality for jpeg data.
        0 = Low quality, 100 = High quality
    compression_type : str, optional
        compression type for tiff file
        select between: None, lzw, deflation (default)
    use_pillow : bool, optional
        Use imageio python library to save the files.
    dpi : float or (float, float)
        Dots per inch (dpi) for saved image.
        Single values are applied as dpi for both dimensions.

    """
    if arr.ndim > 3:
        raise IOError('Image Dimensions should be <=3')

    if isinstance(dpi, (float, int)):
        dpi = (dpi, dpi)

    d_writer = {
        '.png': PNGWriter,
        '.bmp': BMPWriter,
        '.jpeg': JPEGWriter,
        '.jpg': JPEGWriter,
        '.tiff': TIFFWriter,
        '.tif': TIFFWriter,
    }

    extension = os.path.splitext(os.path.basename(filename).lower())[1]

    if extension.lower() not in d_writer.keys():
        raise IOError(
            'Impossible to save the file {0}: Unknown extension {1}'.format(
                filename, extension
            )
        )

    if use_pillow:
        im = Image.fromarray(arr)
        im.save(filename, quality=compression_quality, dpi=dpi)
    else:
        warnings.warn(UserWarning('DPI value is ignored while saving images via vtk.'))
        if arr.ndim == 2:
            arr = arr[..., None]

        shape = arr.shape
        arr = np.flipud(arr)
        if extension.lower() in [
            '.png',
        ]:
            arr = arr.astype(np.uint8)
        arr = arr.reshape((shape[1] * shape[0], shape[2]))
        arr = np.ascontiguousarray(arr, dtype=arr.dtype)
        vtk_array_type = numpy_support.get_vtk_array_type(arr.dtype)
        vtk_array = numpy_support.numpy_to_vtk(
            num_array=arr, deep=True, array_type=vtk_array_type
        )

        # Todo, look the following link for managing png 16bit
        # https://stackoverflow.com/questions/15667947/vtkpngwriter-printing-out-black-images
        vtk_data = ImageData()
        vtk_data.SetDimensions(shape[1], shape[0], shape[2])
        vtk_data.SetExtent(0, shape[1] - 1, 0, shape[0] - 1, 0, 0)
        vtk_data.SetSpacing(1.0, 1.0, 1.0)
        vtk_data.SetOrigin(0.0, 0.0, 0.0)
        vtk_data.GetPointData().SetScalars(vtk_array)

        writer = d_writer.get(extension)()
        writer.SetFileName(filename)
        writer.SetInputData(vtk_data)
        if extension.lower() in ['.jpg', '.jpeg']:
            writer.ProgressiveOn()
            writer.SetQuality(compression_quality)
        if extension.lower() in ['.tif', '.tiff']:
            compression_type = compression_type or 'nocompression'
            l_compression = ['nocompression', 'packbits', 'jpeg', 'deflate', 'lzw']

            if compression_type.lower() in l_compression:
                comp_id = l_compression.index(compression_type.lower())
                writer.SetCompression(comp_id)
            else:
                writer.SetCompressionToDeflate()
        writer.Write()


def load_polydata(file_name):
    """Load a vtk polydata to a supported format file.

    Supported file formats are VTK, VTP, FIB, PLY, STL XML and OBJ

    Parameters
    ----------
    file_name : string

    Returns
    -------
    output : vtkPolyData

    """
    # Check if file actually exists
    if not os.path.isfile(file_name):
        raise FileNotFoundError(file_name)

    file_extension = file_name.split('.')[-1].lower()

    poly_reader = {
        'vtk': PolyDataReader,
        'vtp': XMLPolyDataReader,
        'fib': PolyDataReader,
        'ply': PLYReader,
        'stl': STLReader,
        'xml': XMLPolyDataReader,
    }

    if file_extension in poly_reader.keys():
        reader = poly_reader.get(file_extension)()
    elif file_extension == 'obj':
        # Special case, since there is two obj format
        reader = OBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        if reader.GetOutput().GetNumberOfCells() == 0:
            reader = MNIObjectReader()
    else:
        raise IOError('.' + file_extension + ' is not supported by FURY')

    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()


def save_polydata(polydata, file_name, binary=False, color_array_name=None):
    """Save a vtk polydata to a supported format file.

    Save formats can be VTK, FIB, PLY, STL and XML.

    Parameters
    ----------
    polydata : vtkPolyData
    file_name : string
    binary : bool
    color_array_name: ndarray

    """
    # get file extension (type)
    file_extension = file_name.split('.')[-1].lower()
    poly_writer = {
        'vtk': PolyDataWriter,
        'vtp': XMLPolyDataWriter,
        'fib': PolyDataWriter,
        'ply': PLYWriter,
        'stl': STLWriter,
        'xml': XMLPolyDataWriter,
    }

    if file_extension in poly_writer.keys():
        writer = poly_writer.get(file_extension)()
    elif file_extension == 'obj':
        # Special case, since there is two obj format
        find_keyword = file_name.lower().split('.')
        if 'mni' in find_keyword or 'mnc' in find_keyword:
            writer = MNIObjectWriter()
        else:
            raise IOError(
                'Wavefront obj requires a scene \n'
                " for MNI obj, use '.mni.obj' extension"
            )
    else:
        raise IOError('.' + file_extension + ' is not supported by FURY')

    writer.SetFileName(file_name)
    writer = set_input(writer, polydata)
    if color_array_name is not None and file_extension == 'ply':
        writer.SetArrayName(color_array_name)

    if binary:
        writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()


def load_sprite_sheet(sheet_path, nb_rows, nb_cols, as_vtktype=False):
    """Process and load sprites from a sprite sheet.

    Parameters
    ----------
    sheet_path: str
        Path to the sprite sheet
    nb_rows: int
        Number of rows in the sprite sheet
    nb_cols: int
        Number of columns in the sprite sheet
    as_vtktype: bool, optional
        If True, the output is a vtkImageData

    Returns
    -------
    Dict containing the processed sprites.

    """
    sprite_dicts = {}
    sprite_sheet = load_image(sheet_path)
    width, height = sprite_sheet.shape[:2]

    sprite_size_x = int(np.ceil(width / nb_rows))
    sprite_size_y = int(np.ceil(height / nb_cols))

    for row, col in np.ndindex((nb_rows, nb_cols)):
        nxt_row = row + 1
        nxt_col = col + 1

        box = (
            row * sprite_size_x,
            col * sprite_size_y,
            nxt_row * sprite_size_x,
            nxt_col * sprite_size_y,
        )

        sprite_arr = sprite_sheet[box[0] : box[2], box[1] : box[3]]
        if as_vtktype:
            with InTemporaryDirectory() as tdir:
                tmp_img_path = os.path.join(tdir, f'{row}{col}.png')
                save_image(sprite_arr, tmp_img_path, compression_quality=100)

                sprite_dicts[(row, col)] = load_image(tmp_img_path, as_vtktype=True)
        else:
            sprite_dicts[(row, col)] = sprite_arr

    return sprite_dicts
