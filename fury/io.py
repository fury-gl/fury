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
    extension : str, optional
        The image file format assumed for reading the data. If not
        given, the format is deduced from the filename (By default, it
        will try PNG deduction failed.)
    as_vtktype: bool, optional
        if True, return vtk output otherwise an ndarray. Default False.
    use_pillow: bool
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
            vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, depth)

            # width, height
            vtk_image.SetDimensions(image.shape[1], image.shape[0], 1)
            vtk_image.SetExtent(0, image.shape[1] - 1,
                                0, image.shape[0] - 1,
                                0, 0)
            arr_tmp = np.swapaxes(image, 0, 1)
            arr_tmp = image.reshape(image.shape[1] * image.shape[0], 4)
            arr_tmp = np.ascontiguousarray(arr_tmp)
            uchar_array = numpy_support.numpy_to_vtk(arr_tmp, deep=True)
            vtk_image.GetPointData().SetScalars(uchar_array)
            image = vtk_image.GetOutput()

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
        h, w, _ = reader.GetOutput().GetDimensions()
        vtk_array = reader.GetOutput().GetPointData().GetScalars()

        components = vtk_array.GetNumberOfComponents()
        image = numpy_support.vtk_to_numpy(vtk_array).reshape(h, w, components)
        image = np.swapaxes(image, 0, 1)

    return reader.GetOutput() if as_vtktype else image


def save_image(arr, filename, compression_quality=75,
               compression_type='deflation', use_pillow=True):
    """Save a 2d or 3d image.

    Parameters
    ----------
    arr : ndarray
        array to save
    filename : string
        should be png, bmp, jpeg or jpg files
    compression_quality : int
        compression_quality for jpeg data.
        0 = Low quality, 100 = High quality
    compression_type : str
        compression type for tiff file
        select between: None, lzw, deflation (default)
    use_pillow : bool
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
