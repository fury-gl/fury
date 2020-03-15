import os
import vtk
import numpy as np
from PIL import Image
from vtk.util import numpy_support
from fury.utils import set_input


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
    if use_pillow:
        with Image.open(filename) as pil_image:
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
                    raise RuntimeError('Unknown image mode {}'
                                       .format(pil_image.mode))
                image = np.asarray(pil_image)

        if as_vtktype:
            if image.ndim not in [2, 3]:
                raise IOError("only 2D (L, RGB, RGBA) or 3D image available")

            vtk_image = vtk.vtkImageData()
            depth = 1 if image.ndim == 2 else image.shape[2]

            # width, height
            vtk_image.SetDimensions(image.shape[1], image.shape[0], depth)
            vtk_image.SetExtent(0, image.shape[1] - 1,
                                0, image.shape[0] - 1,
                                0, 0)
            vtk_image.SetSpacing(1.0, 1.0, 1.0)
            vtk_image.SetOrigin(0.0, 0.0, 0.0)
            arr_tmp = np.flipud(image)
            arr_tmp = arr_tmp.reshape(image.shape[1] * image.shape[0], depth)
            arr_tmp = np.ascontiguousarray(arr_tmp, dtype=image.dtype)
            vtk_array_type = numpy_support.get_vtk_array_type(image.dtype)
            uchar_array = numpy_support.numpy_to_vtk(arr_tmp, deep=True,
                                                     array_type=vtk_array_type)
            vtk_image.GetPointData().SetScalars(uchar_array)
            image = vtk_image

        return image

    d_reader = {".png": vtk.vtkPNGReader,
                ".bmp": vtk.vtkBMPReader,
                ".jpeg": vtk.vtkJPEGReader,
                ".jpg": vtk.vtkJPEGReader,
                ".tiff": vtk.vtkTIFFReader,
                ".tif": vtk.vtkTIFFReader}

    extension = os.path.splitext(os.path.basename(filename).lower())[1]

    if extension.lower() not in d_reader.keys():
        raise IOError("Impossible to read the file {0}: Unknown extension {1}".
                      format(filename, extension))

    reader = d_reader.get(extension)()
    reader.SetFileName(filename)
    reader.Update()
    reader.GetOutput().GetPointData().GetArray(0).SetName("original")

    if not as_vtktype:
        w, h, _ = reader.GetOutput().GetDimensions()
        vtk_array = reader.GetOutput().GetPointData().GetScalars()

        components = vtk_array.GetNumberOfComponents()
        image = numpy_support.vtk_to_numpy(vtk_array).reshape(h, w, components)
        image = np.flipud(image)

    return reader.GetOutput() if as_vtktype else image


def save_image(arr, filename, compression_quality=75,
               compression_type='deflation', use_pillow=True):
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

    """
    if arr.ndim > 3:
        raise IOError("Image Dimensions should be <=3")

    d_writer = {".png": vtk.vtkPNGWriter,
                ".bmp": vtk.vtkBMPWriter,
                ".jpeg": vtk.vtkJPEGWriter,
                ".jpg": vtk.vtkJPEGWriter,
                ".tiff": vtk.vtkTIFFWriter,
                ".tif": vtk.vtkTIFFWriter,
                }

    extension = os.path.splitext(os.path.basename(filename).lower())[1]

    if extension.lower() not in d_writer.keys():
        raise IOError("Impossible to save the file {0}: Unknown extension {1}".
                      format(filename, extension))

    if use_pillow:
        im = Image.fromarray(arr)
        im.save(filename, quality=compression_quality)
        return

    if arr.ndim == 2:
        arr = arr[..., None]

    shape = arr.shape
    arr = np.flipud(arr)
    if extension.lower() in ['.png', ]:
        arr = arr.astype(np.uint8)
    arr = arr.reshape((shape[1] * shape[0], shape[2]))
    arr = np.ascontiguousarray(arr, dtype=arr.dtype)
    vtk_array_type = numpy_support.get_vtk_array_type(arr.dtype)
    vtk_array = numpy_support.numpy_to_vtk(num_array=arr,
                                           deep=True,
                                           array_type=vtk_array_type)

    # Todo, look the following link for managing png 16bit
    # https://stackoverflow.com/questions/15667947/vtkpngwriter-printing-out-black-images
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(shape[1], shape[0], shape[2])
    vtk_data.SetExtent(0, shape[1] - 1,
                       0, shape[0] - 1,
                       0, 0)
    vtk_data.SetSpacing(1.0, 1.0, 1.0)
    vtk_data.SetOrigin(0.0, 0.0, 0.0)
    vtk_data.GetPointData().SetScalars(vtk_array)

    writer = d_writer.get(extension)()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_data)
    if extension.lower() in [".jpg", ".jpeg"]:
        writer.ProgressiveOn()
        writer.SetQuality(compression_quality)
    if extension.lower() in [".tif", ".tiff"]:
        if not compression_type:
            writer.SetCompressionToNoCompression()
        elif compression_type.lower() == 'lzw':
            writer.SetCompressionToLZW()
        elif compression_type.lower() == 'deflation':
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
    file_extension = file_name.split(".")[-1].lower()

    poly_reader = {"vtk": vtk.vtkPolyDataReader,
                   "vtp": vtk.vtkXMLPolyDataWriter,
                   "fib": vtk.vtkPolyDataReader,
                   "ply": vtk.vtkPLYReader,
                   "stl": vtk.vtkSTLReader,
                   "xml": vtk.vtkXMLPolyDataReader}

    if file_extension in poly_reader.keys():
        reader = poly_reader.get(file_extension)()
    elif file_extension == "obj":
        # Special case, since there is two obj format
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        if reader.GetOutput().GetNumberOfCells() == 0:
            reader = vtk.vtkMNIObjectReader()
    else:
        raise IOError("." + file_extension + " is not supported by FURY")

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
    file_extension = file_name.split(".")[-1].lower()
    poly_writer = {"vtk": vtk.vtkPolyDataWriter,
                   "vtp": vtk.vtkXMLPolyDataWriter,
                   "fib": vtk.vtkPolyDataWriter,
                   "ply": vtk.vtkPLYWriter,
                   "stl": vtk.vtkSTLWriter,
                   "xml": vtk.vtkXMLPolyDataWriter}

    if file_extension in poly_writer.keys():
        writer = poly_writer.get(file_extension)()
    elif file_extension == "obj":
        # Special case, since there is two obj format
        find_keyword = file_name.lower().split(".")
        if "mni" in find_keyword or "mnc" in find_keyword:
            writer = vtk.vtkMNIObjectWriter()
        else:
            raise IOError("Wavefront obj requires a scene \n"
                          " for MNI obj, use '.mni.obj' extension")
    else:
        raise IOError("." + file_extension + " is not supported by FURY")

    writer.SetFileName(file_name)
    writer = set_input(writer, polydata)
    if color_array_name is not None and file_extension == "ply":
        writer.SetArrayName(color_array_name)

    if binary:
        writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()
