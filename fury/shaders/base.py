import vtk
from vtk.util import numpy_support
from fury import enable_warnings

VTK_9_PLUS = vtk.vtkVersion.GetVTKMajorVersion() >= 9
SHADERS_TYPE = {"vertex": vtk.vtkShader.Vertex,
                "geometry": vtk.vtkShader.Geometry,
                "fragment": vtk.vtkShader.Fragment,
                }

SHADERS_BLOCK = {
    "position": "//VTK::PositionVC",  # frag position in VC
    "normal": "//VTK::Normal",  # optional normal declaration
    "light": "//VTK::Light",  # extra lighting parameters
    "tcoord": "//VTK::TCoord",  # Texture coordinates
    "color": "//VTK::Color",  # material property values
    "clip": "//VTK::Clip",  # clipping plane vars
    "camera": "//VTK::Camera",  # camera and actor matrix values
    "prim_id": "//VTK::PrimID",   # Apple Bug
    "valuepass": "//VTK::ValuePass",  # Value raster
    "output": "//VTK::Output",  # only for geometry shader
}


def shader_to_actor(actor, shader_type, impl_code="", decl_code="",
                    block="valuepass", keep_default=True,
                    replace_first=True, replace_all=False, debug=False):
    """Apply your own substitutions to the shader creation process.

    A bunch of string replacements is applied to a shader template. Using this
    function you can apply your own string replacements to add features you
    desire

    Parameters
    ----------
    actor : vtkActor
        Object where you want to add the shader code.
    shader_type : str
        Shader type: vertex, geometry, fragment
    impl_code : str, optional
        shader implementation code, should be a string or filename
    decl_code : str, optional
        shader declaration code, should be a string or filename
        by default None
    block : str, optional
        section name to be replaced. vtk use of heavy string replacments to
        to insert shader and make it flexible. Each section of the shader
        template have a specific name. For more informations:
        https://vtk.org/Wiki/Shaders_In_VTK. The possible values are:
        position, normal, light, tcoord, color, clip, camera, prim_id,
        valuepass. by default valuepass
    keep_default : bool, optional
        keep the default block tag to let VTK replace it with its default
        behavior. By default True
    replace_first : bool, optional
        If True, apply this change before the standard VTK replacements
        by default True
    replace_all : bool, optional
        [description], by default False
    debug : bool, optional
        introduce a small error to debug shader code.
        by default False

    """
    shader_type = shader_type.lower()
    shader_type = SHADERS_TYPE.get(shader_type, None)
    if shader_type is None:
        msg = "Invalid Shader Type. Please choose between "
        msg += ', '.join(SHADERS_TYPE.keys())
        raise ValueError(msg)

    block = block.lower()
    block = SHADERS_BLOCK.get(block, None)
    if block is None:
        msg = "Invalid Shader Type. Please choose between "
        msg += ', '.join(SHADERS_BLOCK.keys())
        raise ValueError(msg)

    block_dec = block + "::Dec"
    block_impl = block + "::Impl"

    if keep_default:
        decl_code = block_dec + "\n" + decl_code
        impl_code = block_impl + "\n" + impl_code

    if debug:
        enable_warnings()
        error_msg = "\n\n--- DEBUG: THIS LINE GENERATES AN ERROR ---\n\n"
        impl_code += error_msg

    sp = actor.GetShaderProperty() if VTK_9_PLUS else actor.GetMapper()

    sp.AddShaderReplacement(shader_type, block_dec, replace_first,
                            decl_code, replace_all)
    sp.AddShaderReplacement(shader_type, block_impl, replace_first,
                            impl_code, replace_all)


def replace_shader_in_actor(actor, shader_type, code):
    """Set and Replace the shader template with a new one.

    Parameters
    ----------
    actor : vtkActor
        Object where you want to set the shader code.
    shader_type : str
        Shader type: vertex, geometry, fragment
    code : str
        new shader template code

    """
    function_name = {
        "vertex": "SetVertexShaderCode",
        "fragment": "SetFragmentShaderCode",
        "geometry": "SetGeometryShaderCode"
    }
    shader_type = shader_type.lower()
    function = function_name.get(shader_type, None)
    if function is None:
        msg = "Invalid Shader Type. Please choose between "
        msg += ', '.join(function_name.keys())
        raise ValueError(msg)

    sp = actor.GetShaderProperty() if VTK_9_PLUS else actor.GetMapper()
    getattr(sp, function)(code)


def add_shader_callback(actor, callback):
    """Add a shader callback to the actor.

    Parameters
    ----------
    actor : vtkActor
        Rendered Object
    callback : callable
        function or class that contains 3 parameters: caller, event, calldata.
        This callback will be trigger at each `UpdateShaderEvent` event.

    """
    @vtk.calldata_type(vtk.VTK_OBJECT)
    def cbk(caller, event, calldata=None):
        callback(caller, event, calldata)

    mapper = actor.GetMapper()
    mapper.AddObserver(vtk.vtkCommand.UpdateShaderEvent, cbk)


def attribute_to_actor(actor, arr, attr_name, deep=True):
    """Link a numpy array with vertex attribute.

    Parameters
    ----------
    actor : vtkActor
        Rendered Object
    arr : ndarray
        array to link to vertices
    attr_name : str
        vertex attribute name. the vtk array will take the same name as the
        attribute.
    deep : bool, optional
        If True a deep copy is applied. Otherwise a shallow copy is applied,
        by default True

    """
    nb_components = arr.shape[1] if arr.ndim > 1 else arr.ndim
    vtk_array = numpy_support.numpy_to_vtk(arr, deep=deep)
    vtk_array.SetNumberOfComponents(nb_components)
    vtk_array.SetName(attr_name)
    actor.GetMapper().GetInput().GetPointData().AddArray(vtk_array)
    mapper = actor.GetMapper()
    mapper.MapDataArrayToVertexAttribute(
        attr_name, attr_name, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
