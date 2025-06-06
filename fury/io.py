"""I/O functions for loading and saving images, textures."""

import os

# from tempfile import TemporaryDirectory as InTemporaryDirectory
from urllib.request import urlretrieve

# import warnings
from PIL import Image
import numpy as np

# from fury.decorators import warn_on_args_to_kwargs
from fury.lib import (
    #     BMPReader,
    #     BMPWriter,
    #     ImageData,
    #     ImageFlip,
    #     JPEGReader,
    #     JPEGWriter,
    #     MNIObjectReader,
    #     MNIObjectWriter,
    #     OBJReader,
    #     PLYReader,
    #     PLYWriter,
    #     PNGReader,
    #     PNGWriter,
    #     PolyDataReader,
    #     PolyDataWriter,
    #     STLReader,
    #     STLWriter,
    #     TIFFReader,
    #     TIFFWriter,
    #     Texture,
    #     XMLPolyDataReader,
    #     XMLPolyDataWriter,
    #     numpy_support,
    Texture,
)

# from fury.utils import set_input


def load_cube_map_texture(fnames, *, size=None, generate_mipmaps=True):
    """Load a cube map texture from a list of images.

    Parameters
    ----------
    fnames : list
        Filenames to generate cube map.
    size : tuple, optional
        The display extent (width, height, depth) of the cubemap.
    generate_mipmaps : bool, optional
        Whether to automatically generate mipmaps when transferring data to the GPU.

    Returns
    -------
    Texture
        PyGfx Texture object.
    """
    images = []

    for fname in fnames:
        images.append(load_image(fname))

    if size is None:
        min_side = min(*images[0].shape[:2])
        for image in images:
            min_side = min(*image.shape[:2], min_side)
        size = (min_side, min_side, 6)

    data = np.stack(images, axis=0)

    return Texture(data, dim=2, size=size, generate_mipmaps=generate_mipmaps)


def load_image_texture(fname, *, size=None, generate_mipmaps=True):
    """Load an image texture from a file or URL.

    Parameters
    ----------
    fname : str
        Path to image file or URL. Should be png, bmp, jpeg or jpg files.
    size : tuple, optional
        The display extent (width, height) of the texture.
    generate_mipmaps : bool, optional
        Whether to automatically generate mipmaps when transferring data to the GPU.

    Returns
    -------
    Texture
        PyGfx Texture object.
    """
    image = load_image(fname)

    if size is None:
        size = image.shape[:2]

    return Texture(image, dim=2, size=size, generate_mipmaps=generate_mipmaps)


def load_image(filename):
    """Load an image from a file or URL.

    Parameters
    ----------
    filename : str
        Path to image file or URL. Should be png, bmp, jpeg or jpg files.

    Returns
    -------
    ndarray
        Loaded image array.
    """
    is_url = (filename.lower().startswith("http://")) or (
        filename.lower().startswith("https://")
    )

    if is_url:
        image_name = os.path.basename(filename)

        if len(image_name.split(".")) < 2:
            raise OSError(f"{filename} is not a valid image URL")

        urlretrieve(filename, image_name)
        filename = image_name

    with Image.open(filename) as pil_image:
        if pil_image.mode in ["P"]:
            pil_image = pil_image.convert("RGB")

        if pil_image.mode in ["RGBA", "RGB", "L"]:
            image = np.asarray(pil_image)
        elif pil_image.mode.startswith("I;16"):
            raw = pil_image.tobytes("raw", pil_image.mode)
            dtype = ">u2" if pil_image.mode.endswith("B") else "<u2"
            image = np.frombuffer(raw, dtype=dtype)
            image.reshape(pil_image.size[::-1]).astype("=u2")
        else:
            try:
                image = pil_image.convert("RGBA")
            except ValueError as err:
                raise RuntimeError(f"Unknown image mode {pil_image.mode}") from err
            image = np.asarray(pil_image)

    if is_url:
        os.remove(filename)
    return image


# def load_text(file):
#     """Load a text file.

#     Parameters
#     ----------
#     file : str
#         Path to the text file.

#     Returns
#     -------
#     text : str
#         Text contained in the file.
#     """
#     if not os.path.isfile(file):
#         raise IOError("File {} does not exist.".format(file))
#     with open(file) as f:
#         text = f.read()
#     return text


def save_image(
    arr,
    filename,
    *,
    compression_quality=75,
    compression_type="deflation",
    dpi=(72, 72),
):
    """Save a 2D or 3D image to a file.

    Parameters
    ----------
    arr : ndarray
        Array to save with shape (H, W) or (H, W, 1) or (H, W, 3) or (H, W, 4).
    filename : str
        Output filename. Should be png, bmp, jpeg, jpg, tiff, or tif file.
    compression_quality : int, optional
        Compression quality for jpeg data. 0 = Low quality, 100 = High quality.
    compression_type : str, optional
        Compression type for tiff file. Select between: None, lzw, deflation (default).
    dpi : float or tuple of float, optional
        Dots per inch (dpi) for saved image.
        Single values are applied as dpi for both dimensions.
    """
    if arr.ndim > 3:
        raise OSError("Image Dimensions should be <=3")

    if isinstance(dpi, (float, int)):
        dpi = (dpi, dpi)

    allowed_extensions = [".png", ".bmp", ".jpeg", ".jpg", ".tiff", ".tif"]

    extension = os.path.splitext(os.path.basename(filename).lower())[1]

    if extension.lower() not in allowed_extensions:
        raise OSError(
            f"Impossible to save the file {filename}: Unknown extension {extension}"
        )

    im = Image.fromarray(arr)
    im.save(filename, quality=compression_quality, dpi=dpi)


# def load_polydata(file_name):
#     """Load a vtk polydata to a supported format file.

#     Supported file formats are VTK, VTP, FIB, PLY, STL XML and OBJ

#     Parameters
#     ----------
#     file_name : string

#     Returns
#     -------
#     output : vtkPolyData

#     """
#     # Check if file actually exists
#     if not os.path.isfile(file_name):
#         raise FileNotFoundError(file_name)

#     file_extension = file_name.split(".")[-1].lower()

#     poly_reader = {
#         "vtk": PolyDataReader,
#         "vtp": XMLPolyDataReader,
#         "fib": PolyDataReader,
#         "ply": PLYReader,
#         "stl": STLReader,
#         "xml": XMLPolyDataReader,
#     }

#     if file_extension in poly_reader.keys():
#         reader = poly_reader.get(file_extension)()
#     elif file_extension == "obj":
#         # Special case, since there is two obj format
#         reader = OBJReader()
#         reader.SetFileName(file_name)
#         reader.Update()
#         if reader.GetOutput().GetNumberOfCells() == 0:
#             reader = MNIObjectReader()
#     else:
#         raise IOError("." + file_extension + " is not supported by FURY")

#     reader.SetFileName(file_name)
#     reader.Update()
#     return reader.GetOutput()


# @warn_on_args_to_kwargs()
# def save_polydata(polydata, file_name, *, binary=False, color_array_name=None):
#     """Save a vtk polydata to a supported format file.

#     Save formats can be VTK, FIB, PLY, STL and XML.

#     Parameters
#     ----------
#     polydata : vtkPolyData
#     file_name : string
#     binary : bool
#     color_array_name: ndarray

#     """
#     # get file extension (type)
#     file_extension = file_name.split(".")[-1].lower()
#     poly_writer = {
#         "vtk": PolyDataWriter,
#         "vtp": XMLPolyDataWriter,
#         "fib": PolyDataWriter,
#         "ply": PLYWriter,
#         "stl": STLWriter,
#         "xml": XMLPolyDataWriter,
#     }

#     if file_extension in poly_writer.keys():
#         writer = poly_writer.get(file_extension)()
#     elif file_extension == "obj":
#         # Special case, since there is two obj format
#         find_keyword = file_name.lower().split(".")
#         if "mni" in find_keyword or "mnc" in find_keyword:
#             writer = MNIObjectWriter()
#         else:
#             raise IOError(
#                 "Wavefront obj requires a scene \n"
#                 " for MNI obj, use '.mni.obj' extension"
#             )
#     else:
#         raise IOError("." + file_extension + " is not supported by FURY")

#     writer.SetFileName(file_name)
#     writer = set_input(writer, polydata)
#     if color_array_name is not None and file_extension == "ply":
#         writer.SetArrayName(color_array_name)

#     if binary:
#         writer.SetFileTypeToBinary()
#     writer.Update()
#     writer.Write()


# @warn_on_args_to_kwargs()
# def load_sprite_sheet(sheet_path, nb_rows, nb_cols, *, as_vtktype=False):
#     """Process and load sprites from a sprite sheet.

#     Parameters
#     ----------
#     sheet_path: str
#         Path to the sprite sheet
#     nb_rows: int
#         Number of rows in the sprite sheet
#     nb_cols: int
#         Number of columns in the sprite sheet
#     as_vtktype: bool, optional
#         If True, the output is a vtkImageData

#     Returns
#     -------
#     Dict containing the processed sprites.

#     """
#     sprite_dicts = {}
#     sprite_sheet = load_image(sheet_path)
#     width, height = sprite_sheet.shape[:2]

#     sprite_size_x = int(np.ceil(width / nb_rows))
#     sprite_size_y = int(np.ceil(height / nb_cols))

#     for row, col in np.ndindex((nb_rows, nb_cols)):
#         nxt_row = row + 1
#         nxt_col = col + 1

#         box = (
#             row * sprite_size_x,
#             col * sprite_size_y,
#             nxt_row * sprite_size_x,
#             nxt_col * sprite_size_y,
#         )

#         sprite_arr = sprite_sheet[
#             box[0] : box[2], box[1] : box[3]  # noqa: E203
#         ]
#         if as_vtktype:
#             with InTemporaryDirectory() as tdir:
#                 tmp_img_path = os.path.join(tdir, f"{row}{col}.png")
#                 save_image(sprite_arr, tmp_img_path, compression_quality=100)

#                 sprite_dicts[(row, col)] = load_image(
#                     tmp_img_path,
#                     as_vtktype=True,
#                 )
#         else:
#             sprite_dicts[(row, col)] = sprite_arr

#     return sprite_dicts
