import os
import numpy as np
import vtk
from vtk.util import numpy_support
from fury.utils import set_input


def load_image(file_name, as_vtktype=False):
    """Load an image.

    Parameters
    ----------
    file_name: string
        should be png, bmp, jpeg or jpg files
    as_vtktype: bool, optional
        if True, return vtk output otherwise a ndarray. Default False.

    Returns
    -------
    image: ndarray or vtk output
        desired image array

    """
    d_reader = {".png": vtk.vtkPNGReader,
                ".bmp": vtk.vtkBMPReader,
                ".jpeg": vtk.vtkJPEGReader,
                ".jpg": vtk.vtkJPEGReader,
                ".tiff": vtk.vtkTIFFReader,
                ".tif": vtk.vtkTIFFReader,
                }

    extension = os.path.splitext(os.path.basename(file_name).lower())[1]

    if extension.lower() not in d_reader.keys():
        raise IOError("Impossible to read the file {0}: Unknown extension {1}".
                      format(file_name, extension))

    reader = d_reader.get(extension)()
    reader.SetFileName(file_name)
    reader.Update()
    reader.GetOutput().GetPointData().GetArray(0).SetName("original")

    if as_vtktype:
        return reader.GetOutput()

    h, w, _ = reader.GetOutput().GetDimensions()
    vtk_array = reader.GetOutput().GetPointData().GetScalars()

    components = vtk_array.GetNumberOfComponents()
    image = numpy_support.vtk_to_numpy(vtk_array).reshape(h, w, components)
    return image


def save_image(arr, file_name, compression_quality=100,
               compression_type='deflation'):
    """Save a 2d or 3d image.

    Parameters
    ----------
    arr: ndarray
        array to save
    file_name: string
        should be png, bmp, jpeg or jpg files
    compression_quality: int
        compression_quality for jpeg data.
        0 = Low quality, 100 = High quality
    compression_type: str
        compression type for tiff file
        select between: None, lzw, deflation (default)

    """
    if arr.ndim > 3:
        raise IOError("Image Dimensions should be <=3")
    if arr.ndim == 2:
        arr = arr[..., None]

    d_writer = {".png": vtk.vtkPNGWriter,
                ".bmp": vtk.vtkBMPWriter,
                ".jpeg": vtk.vtkJPEGWriter,
                ".jpg": vtk.vtkJPEGWriter,
                ".tiff": vtk.vtkTIFFWriter,
                ".tif": vtk.vtkTIFFWriter,
                }

    extension = os.path.splitext(os.path.basename(file_name).lower())[1]

    if extension.lower() not in d_writer.keys():
        raise IOError("Impossible to save the file {0}: Unknown extension {1}".
                      format(file_name, extension))

    vtk_array_type = numpy_support.get_vtk_array_type(arr.dtype)
    vtk_array = numpy_support.numpy_to_vtk(num_array=arr.ravel(), deep=True,
                                           array_type=vtk_array_type)

    # Todo, look the following link for managing png 16bit
    # https://stackoverflow.com/questions/15667947/vtkpngwriter-printing-out-black-images
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(arr.shape)
    vtk_data.SetSpacing(1.0, 1.0, 1.0)
    vtk_data.SetOrigin(0.0, 0.0, 0.0)
    vtk_data.GetPointData().SetScalars(vtk_array)

    writer = d_writer.get(extension)()
    writer.SetFileName(file_name)
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


def load_polydata(file_name, is_mni_obj=False):
    """Load a vtk polydata to a supported format file.

    Supported file formats are OBJ, VTK, FIB, PLY, STL and XML

    Parameters
    ----------
    file_name : string
    is_mni_obj : bool

    Returns
    -------
    output : vtkPolyData

    """
    file_extension = file_name.split(".")[-1].lower()

    if file_extension == "vtk":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "fib":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "ply":
        reader = vtk.vtkPLYReader()
    elif file_extension == "stl":
        reader = vtk.vtkSTLReader()
    elif file_extension == "xml":
        reader = vtk.vtkXMLPolyDataReader()
    elif file_extension == "obj" and is_mni_obj:
        reader = vtk.vtkMNIObjectReader()
    elif file_extension == "obj":
        try:  # try to read as a normal obj
            reader = vtk.vtkOBJReader()
        except Exception:  # than try load a MNI obj format
            reader = vtk.vtkMNIObjectReader()
    else:
        raise IOError("polydata " + file_extension + " is not suported")

    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()


def save_polydata(polydata, file_name, binary=False, color_array_name=None,
                  is_mni_obj=False):
    """Save a vtk polydata to a supported format file.

    Save formats can be VTK, FIB, PLY, STL and XML.

    Parameters
    ----------
    polydata : vtkPolyData
    file_name : string
    binary : bool
    color_array_name: ndarray
    is_mni_obj : bool

    """
    # get file extension (type)
    file_extension = file_name.split(".")[-1].lower()

    if file_extension == "vtk":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == "fib":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == "ply":
        writer = vtk.vtkPLYWriter()
    elif file_extension == "stl":
        writer = vtk.vtkSTLWriter()
    elif file_extension == "xml":
        writer = vtk.vtkXMLPolyDataWriter()
    elif file_extension == "obj":
        if is_mni_obj:
            writer = vtk.vtkMNIObjectWriter()
        else:
            # vtkObjWriter not available on python
            # vtk.vtkOBJWriter()
            raise IOError("OBJ Writer not available. MNI obj is the only"
                          " available writer so set mni_tag option to True")
    else:
        raise IOError("Unknown extension ({})".format(file_extension))

    writer.SetFileName(file_name)
    writer = set_input(writer, polydata)
    if color_array_name is not None and file_extension == "ply":
        writer.SetArrayName(color_array_name)

    if binary:
        writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()
