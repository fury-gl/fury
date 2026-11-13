"""I/O functions for loading and saving images, textures."""

import os
from urllib.request import urlretrieve
import warnings

from PIL import Image
import numpy as np
import polyxios as px

from fury.lib import Texture, wgpu
from fury.network.parser import parse_network, stringify_network


def load_cube_map_texture(fnames, *, size=None, generate_mipmaps=True):
    """
    Load a cube map texture from a list of images.

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
    images = [load_image(fname) for fname in fnames]

    if size is None:
        min_side = min(*images[0].shape[:2])
        for image in images:
            min_side = min(*image.shape[:2], min_side)
        size = (min_side, min_side, 6)

    data = np.stack(images, axis=0)

    return Texture(data, dim=2, size=size, generate_mipmaps=generate_mipmaps)


def load_image_texture(fname, *, size=None, generate_mipmaps=True):
    """
    Load an image texture from a file or URL.

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
    """
    Load an image from a file or URL.

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

        if not get_extension(image_name):
            raise OSError(f"{filename} is not a valid image URL")

        urlretrieve(filename, image_name)
        filename = image_name

    with Image.open(filename) as pil_image:
        if pil_image.mode == "P":
            if "transparency" in pil_image.info:
                pil_image = pil_image.convert("RGBA")
            else:
                pil_image = pil_image.convert("RGB")

        if pil_image.mode in ["RGBA", "RGB", "L"]:
            image = np.asarray(pil_image)
        elif pil_image.mode == "F":
            image = np.array(pil_image, dtype=np.float32)
        elif pil_image.mode.startswith("I;16"):
            raw = pil_image.tobytes("raw", pil_image.mode)
            dtype = ">u2" if pil_image.mode.endswith("B") else "<u2"
            image = np.frombuffer(raw, dtype=dtype)
            image = image.reshape(pil_image.size[::-1]).astype("=u2")
        else:
            try:
                image = pil_image.convert("RGBA")
            except ValueError as err:
                raise RuntimeError(f"Unknown image mode {pil_image.mode}") from err
            image = np.asarray(pil_image)

    if is_url:
        os.remove(filename)
    return image


def load_image_as_wgpu_texture_view(filename, device):
    """
    Load an image from a file or URL as a wgpu Texture view.

    Parameters
    ----------
    filename : str
        Path to image file or URL. Should be png, bmp, jpeg or jpg files.
    device : wgpu.GPUDevice
        The wgpu device to create the texture on.

    Returns
    -------
    wgpu.GPUTextureView
        Loaded image as wgpu Texture view.
    """
    image = np.asarray(load_image(filename), dtype=np.uint8)

    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    if image.shape[-1] == 3:
        alpha_channel = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
        image = np.concatenate((image, alpha_channel), axis=-1)
    elif image.shape[-1] != 4:
        raise ValueError("Images must have 1, 3, or 4 channels.")

    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]

    valid_devices = (wgpu.GPUDevice,)
    if hasattr(wgpu, "WGPUDevice"):
        valid_devices = (wgpu.GPUDevice, wgpu.WGPUDevice)

    if device is None or not isinstance(device, valid_devices):
        raise ValueError("A valid wgpu device must be provided.")

    texture = device.create_texture(
        size=(width, height, 1),
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        mip_level_count=1,
        sample_count=1,
    )

    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        image,
        {"offset": 0, "bytes_per_row": 4 * width, "rows_per_image": height},
        {"width": width, "height": height, "depth_or_array_layers": 1},
    )

    return texture.create_view()


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
    dtype=None,
):
    """
    Save a 2D or 3D image to a file.

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
    dtype : numpy.dtype or type or None, optional
        Data type to which the array will be converted before saving.
        Default is None. When None, native supported types (uint8, 2D uint16
        for PNG/TIFF, and 2D float32 for TIFF) are preserved, while other
        types (e.g., float64, int64, bool) are converted to uint8.
        If an explicit dtype is provided, the array is converted to that dtype.
    """
    # 1. Validate the input
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError("Cannot save an empty image array.")

    if arr.ndim < 2 or arr.ndim > 3:
        raise ValueError(f"Image dimensions must be 2 or 3, but got {arr.ndim}.")

    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"3D image must have 1, 3, or 4 channels, but got {arr.shape[2]}."
        )

    if not (
        np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.floating)
        or np.issubdtype(arr.dtype, np.bool_)
    ):
        raise ValueError(
            f"Cannot save image with unsupported array dtype '{arr.dtype}'."
        )

    allowed_extensions = ["png", "bmp", "jpeg", "jpg", "tiff", "tif"]
    extension = get_extension(filename).lower()
    if extension not in allowed_extensions:
        raise ValueError(
            f"Impossible to save the file {filename}: Unknown extension '{extension}'"
        )

    if isinstance(dpi, (float, int)):
        dpi = (dpi, dpi)

    # Normalize (H, W, 1) to (H, W) for Pillow compatibility
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # 2. Resolve the target dtype
    if dtype is None:
        if arr.dtype == np.uint8:
            target_dtype = np.dtype(np.uint8)
        elif (
            arr.dtype == np.uint16
            and arr.ndim == 2
            and extension in ("png", "tiff", "tif")
        ):
            target_dtype = np.dtype(np.uint16)
        elif arr.dtype == np.float32 and arr.ndim == 2 and extension in ("tiff", "tif"):
            target_dtype = np.dtype(np.float32)
        else:
            target_dtype = np.dtype(np.uint8)
    else:
        target_dtype = np.dtype(dtype)

        if not (
            np.issubdtype(target_dtype, np.integer)
            or np.issubdtype(target_dtype, np.floating)
        ):
            raise ValueError(
                f"Unsupported target dtype '{target_dtype}' for image saving."
            )

        # Validate Pillow write capabilities
        if np.issubdtype(target_dtype, np.floating):
            if arr.ndim != 2:
                raise ValueError(
                    f"Floating-point images must be 2D; cannot write 3D float "
                    f"image with shape {arr.shape}."
                )
            if extension not in ("tiff", "tif"):
                raise ValueError(
                    f"Cannot write floating-point images to '{extension}' format. "
                    f"Only TIFF supports floating-point data."
                )

        if target_dtype == np.uint16:
            if arr.ndim != 2:
                raise ValueError(
                    f"16-bit images must be 2D; cannot write 3D 16-bit image "
                    f"with shape {arr.shape}."
                )
            if extension not in ("png", "tiff", "tif"):
                raise ValueError(
                    f"Cannot write 16-bit images to '{extension}' format. "
                    f"Only PNG and TIFF support 16-bit data."
                )

        if extension in ("jpeg", "jpg", "bmp") and target_dtype != np.uint8:
            raise ValueError(
                f"'{extension}' format only supports uint8 data, got '{target_dtype}'."
            )

        if arr.ndim == 3 and target_dtype != np.uint8:
            raise ValueError(
                f"3D images only support uint8 data, got '{target_dtype}'."
            )

    # 3. Conversions and Warning Policy
    if arr.dtype == np.bool_:
        warnings.warn(
            "Boolean image data is being converted to uint8.",
            UserWarning,
            stacklevel=2,
        )
        if target_dtype == np.uint8:
            arr = arr.astype(np.uint8) * 255
        else:
            arr = arr.astype(target_dtype)

    elif np.issubdtype(arr.dtype, np.floating) and np.issubdtype(
        target_dtype, np.integer
    ):
        if np.isnan(arr).any():
            raise ValueError("Cannot convert float image containing NaN to integer.")

        # Scale [0, 1] only when target is uint8
        if target_dtype == np.uint8:
            if arr.min() >= 0.0 and arr.max() <= 1.0 and arr.max() > 0.0:
                arr = arr * 255.0
                warnings.warn(
                    "Floating-point image data in range [0, 1] was scaled to [0, 255].",
                    UserWarning,
                    stacklevel=2,
                )

        arr = np.round(arr)

        info = np.iinfo(target_dtype)
        if arr.min() < info.min or arr.max() > info.max:
            warnings.warn(
                f"Image data contains values outside [{info.min}, {info.max}]; "
                f"values will be clipped.",
                UserWarning,
                stacklevel=2,
            )
            arr = np.clip(arr, info.min, info.max)

        arr = arr.astype(target_dtype)

    elif np.issubdtype(arr.dtype, np.integer) and np.issubdtype(
        target_dtype, np.integer
    ):
        info = np.iinfo(target_dtype)
        if arr.min() < info.min or arr.max() > info.max:
            warnings.warn(
                f"Integer image data contains values outside [{info.min}, {info.max}]; "
                f"values will be clipped.",
                UserWarning,
                stacklevel=2,
            )
            arr = np.clip(arr, info.min, info.max)

        arr = arr.astype(target_dtype)

    elif np.issubdtype(arr.dtype, np.floating) and np.issubdtype(
        target_dtype, np.floating
    ):
        arr = arr.astype(target_dtype)

    elif np.issubdtype(arr.dtype, np.integer) and np.issubdtype(
        target_dtype, np.floating
    ):
        arr = arr.astype(target_dtype)

    im = Image.fromarray(arr)

    save_kwargs = {"dpi": dpi}

    if extension in ("tiff", "tif"):
        if compression_type is not None and im.mode != "F":
            compression_map = {
                "lzw": "tiff_lzw",
                "deflation": "tiff_deflate",
            }
            pil_compression = compression_map.get(compression_type)
            if pil_compression is None:
                raise OSError(
                    f"Unknown compression type '{compression_type}'. "
                    f"Supported types for TIFF: {list(compression_map.keys())}"
                )
            save_kwargs["compression"] = pil_compression
    else:
        save_kwargs["quality"] = compression_quality

    im.save(filename, **save_kwargs)


def get_extension(file_path):
    """
    Get the file extension.

    Parameters
    ----------
    file_path : str
        The path or name of the file.

    Returns
    -------
    str
        The file extension without the leading dot. Returns an empty string
        if there is no extension.
    """
    root, ext = os.path.splitext(file_path)

    if ext.startswith("."):
        return ext[1:]
    return ext


def load_network(file_path, *, format=None):
    """
    Load a network from a file.

    Parameters
    ----------
    file_path : str
        The path to the network file.
    format : str, optional
        The specific file format of the network file (e.g., 'gexf', 'gml', or 'xnet').

    Returns
    -------
    tuple
        A tuple containing:
        - nodes_xyz (np.ndarray): Shape (N, 3) float32 array of node positions.
        - edges_indices (np.ndarray): Shape (E, 2) int32 array of edge connections.
        - colors (np.ndarray): Shape (N, 4) float32 array of node colors (RGBA).
    """
    if format is None:
        _, ext = os.path.splitext(file_path)
        format = ext.lstrip(".").lower()

    with open(file_path, encoding="utf-8") as f:
        data = f.read()

    return parse_network(data, format)


def save_network(network_data, file_path, *, format=None):
    """
    Save a network to a file.

    Parameters
    ----------
    network_data : tuple
        A tuple containing:
        - nodes_xyz (np.ndarray): Shape (N, 3) float32 array of node positions.
        - edges_indices (np.ndarray): Shape (E, 2) int32 array of edge connections.
        - colors (np.ndarray): Shape (N, 4) float32 array of node colors (RGBA).
    file_path : str
        The destination path where the network file will be written.
    format : str, optional
        The specific export format of the network file.
    """
    if format is None:
        _, ext = os.path.splitext(file_path)
        format = ext.lstrip(".").lower()

    data = stringify_network(network_data, format)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data)


def read_mesh(file_path, *, format=None):
    """
    Read a mesh from a file using polyxios.

    Parameters
    ----------
    file_path : str
        The path to the mesh file.
    format : str, optional
        The specific file format override (e.g. '.vtk'). Inferred from the file
        extension when None.

    Returns
    -------
    tuple
        A tuple containing:

        - vertices (np.ndarray): Shape (N, 3) float32 array of vertex positions.
        - faces (np.ndarray or None): Shape (M, 3) int32 array of triangle face
          indices, or None when the mesh has no surface elements.
        - colors (np.ndarray or None): Shape (N, 3) float32 array of per-vertex
          RGB colors in [0, 1], or None when no vertex colors are present.
    """
    poly = px.read(file_path, fmt=format)

    vertices = np.asarray(poly.vertices, dtype=np.float32)

    faces = poly.faces
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int32)

    colors = px.transforms.vertex_colors(poly)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)

    return vertices, faces, colors


def read_points(file_path, *, format=None):
    """
    Read point coordinates from a file using polyxios.

    Parameters
    ----------
    file_path : str
        The path to the mesh file.
    format : str, optional
        The specific file format override (e.g. '.vtk'). Inferred from the file
        extension when None.

    Returns
    -------
    tuple
        A tuple containing:

        - points (np.ndarray): Shape (N, 3) float32 array of point positions.
        - colors (np.ndarray or None): Shape (N, 3) float32 array of per-point
          RGB colors in [0, 1], or None when no vertex colors are present.
    """
    poly = px.read(file_path, fmt=format)

    points = np.asarray(poly.vertices, dtype=np.float32)

    colors = px.transforms.vertex_colors(poly)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)

    return points, colors


def read_lines(file_path, *, format=None):
    """
    Read line segments from a file using polyxios.

    Each line or poly_line element is translated into an array of its vertex
    positions, ready to be consumed by FURY line actors.

    Parameters
    ----------
    file_path : str
        The path to the mesh file.
    format : str, optional
        The specific file format override (e.g. '.vtk'). Inferred from the file
        extension when None.

    Returns
    -------
    tuple
        A tuple containing:

        - lines (list of np.ndarray): One Shape (P, 3) float32 array of vertex
          positions per line. Empty list when the file has no line elements.
        - colors (list of np.ndarray or None): One Shape (P, 3) float32 array of
          per-vertex RGB colors in [0, 1] per line, or None when no vertex
          colors are present.
    """
    poly = px.read(file_path, fmt=format)

    line_indices = poly.lines
    if line_indices is None:
        return [], None

    vertices = np.asarray(poly.vertices, dtype=np.float32)
    lines = [vertices[idx] for idx in line_indices]

    vertex_colors = px.transforms.vertex_colors(poly)
    if vertex_colors is None:
        colors = None
    else:
        vertex_colors = np.asarray(vertex_colors, dtype=np.float32)
        colors = [vertex_colors[idx] for idx in line_indices]

    return lines, colors


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
