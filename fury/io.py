import numpy as np
import vtk
from fury.utils import set_input


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
    # get file extension (type) lower case
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
