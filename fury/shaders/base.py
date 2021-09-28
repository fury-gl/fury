from functools import partial

from fury import enable_warnings
from fury.lib import (VTK_9_PLUS, numpy_support, Command, VTK_OBJECT,
                      calldata_type, DataObject, Shader)

SHADERS_TYPE = {"vertex": Shader.Vertex,
                "geometry": Shader.Geometry,
                "fragment": Shader.Fragment,
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

# See [1] for a more extensive list of OpenGL constants
# [1] https://docs.factorcode.org/content/vocab-opengl.gl.html
GL_NUMBERS = {
    "GL_SRC_ALPHA": 770,
    "GL_ONE": 1,
    "GL_ZERO": 0,
    "GL_BLEND": 3042,
    "GL_ONE_MINUS_SRC_ALPHA": 771,
    "GL_SRC_ALPHA": 770,
    "GL_DEPTH_TEST": 2929,
    "GL_DST_COLOR": 774,
    "GL_FUNC_SUBTRACT": 3277,
    "GL_CULL_FACE": 2884,
    "GL_ALPHA_TEST": 3008,
    "GL_CW": 2304,
    "GL_CCW": 2305,
    "GL_ONE_MINUS_SRC_COLOR": 769,
    "GL_SRC_COLOR": 768
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


def add_shader_callback(actor, callback, priority=0.):
    """Add a shader callback to the actor.

    Parameters
    ----------
    actor : vtkActor
        Rendered Object
    callback : callable
        function or class that contains 3 parameters: caller, event, calldata.
        This callback will be trigger at each `UpdateShaderEvent` event.
    priority : float, optional
        Commands with a higher priority are called first.

    Returns
    -------
    id_observer : int
        An unsigned Int tag which can be used later to remove the event
        or retrieve the vtkCommand used in the observer.
        See more at: https://vtk.org/doc/nightly/html/classvtkObject.html

    Examples
    ---------
    .. code-block:: python

        add_shader_callback(actor, func_call1)
        id_observer = add_shader_callback(actor, func_call2)
        actor.GetMapper().RemoveObserver(id_observer)

    Priority calls

    .. code-block:: python

        test_values = []
        def callbackLow(_caller, _event, calldata=None):
            program = calldata
            if program is not None:
                test_values.append(0)

        def callbackHigh(_caller, _event, calldata=None):
            program = calldata
            if program is not None:
                test_values.append(999)

        def callbackMean(_caller, _event, calldata=None):
            program = calldata
            if program is not None:
                test_values.append(500)

        fs.add_shader_callback(
                actor, callbackHigh, 999)
        fs.add_shader_callback(
                actor, callbackLow, 0)
        id_mean = fs.add_shader_callback(
                actor, callbackMean, 500)

        showm.start()
        # test_values = [999, 500, 0, 999, 500, 0, ...]

    """
    @calldata_type(VTK_OBJECT)
    def cbk(caller, event, calldata=None):
        callback(caller, event, calldata)

    if not isinstance(priority, (float, int)):
        raise TypeError("""
            add_shader_callback priority argument shoud be a float/int""")

    mapper = actor.GetMapper()
    id_observer = mapper.AddObserver(Command.UpdateShaderEvent, cbk, priority)

    return id_observer


def shader_apply_effects(
        window, actor, effects, priority=0):
    """This applies a specific opengl state (effect) or a list of effects just
    before the actor's shader is executed.

    Parameters
    ----------
    window : RenderWindow
        For example, this is provided by the ShowManager.window attribute.
    actor : actor
    effects : a function or a list of functions
    priority : float, optional
        Related with the shader callback command.
        Effects with a higher priority are applied first and
        can be override by the others.

    Returns
    -------
    id_observer : int
        An unsigned Int tag which can be used later to remove the event
        or retrieve the vtkCommand used in the observer.
        See more at: https://vtk.org/doc/nightly/html/classvtkObject.html

    """
    if not isinstance(effects, list):
        effects = [effects]

    def callback(
            _caller, _event, calldata=None,
            effects=None, window=None):
        program = calldata
        glState = window.GetState()
        if program is not None:
            for func in effects:
                func(glState)

    id_observer = add_shader_callback(
        actor, partial(
            callback,
            effects=effects, window=window), priority)

    return id_observer


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
        attr_name, attr_name, DataObject.FIELD_ASSOCIATION_POINTS, -1)
