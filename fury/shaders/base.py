from functools import partial
import os

from fury import enable_warnings
from fury.deprecator import deprecate_with_version
from fury.io import load_text
from fury.lib import (
    VTK_OBJECT,
    Command,
    DataObject,
    Shader,
    calldata_type,
    numpy_support,
)

SHADERS_DIR = os.path.join(os.path.dirname(__file__))

SHADERS_EXTS = ['.glsl', '.vert', '.tesc', '.tese', '.geom', '.frag', '.comp']

SHADERS_TYPE = {
    'vertex': Shader.Vertex,
    'geometry': Shader.Geometry,
    'fragment': Shader.Fragment,
}

REPLACEMENT_SHADERS_TYPES = {'vertex': Shader.Vertex, 'fragment': Shader.Fragment}

SHADERS_BLOCK = {
    'position': '//VTK::PositionVC',  # frag position in VC
    'normal': '//VTK::Normal',  # optional normal declaration
    'light': '//VTK::Light',  # extra lighting parameters
    'tcoord': '//VTK::TCoord',  # Texture coordinates
    'color': '//VTK::Color',  # material property values
    'clip': '//VTK::Clip',  # clipping plane vars
    'camera': '//VTK::Camera',  # camera and actor matrix values
    'prim_id': '//VTK::PrimID',  # Apple Bug
    'valuepass': '//VTK::ValuePass',  # Value raster
    'output': '//VTK::Output',  # only for geometry shader
    'coincident': '//VTK::Coincident',  # handle coincident offsets
    'zbufer': '//VTK::ZBuffer',
    'depth_peeling': '//VTK::DepthPeeling',  # Depth Peeling Support
    'picking': '//VTK::Picking',  # picking support
}

# See [1] for a more extensive list of OpenGL constants
# [1] https://docs.factorcode.org/content/vocab-opengl.gl.html
GL_NUMBERS = {
    'GL_ONE': 1,
    'GL_ZERO': 0,
    'GL_BLEND': 3042,
    'GL_ONE_MINUS_SRC_ALPHA': 771,
    'GL_SRC_ALPHA': 770,
    'GL_DEPTH_TEST': 2929,
    'GL_DST_COLOR': 774,
    'GL_FUNC_SUBTRACT': 3277,
    'GL_CULL_FACE': 2884,
    'GL_ALPHA_TEST': 3008,
    'GL_CW': 2304,
    'GL_CCW': 2305,
    'GL_ONE_MINUS_SRC_COLOR': 769,
    'GL_SRC_COLOR': 768,
}


def compose_shader(glsl_code):
    """Merge GLSL shader code from a list of strings.

    Parameters
    ----------
    glsl_code : list of str (code or filenames).

    Returns
    -------
    code : str
        GLSL shader code.

    """
    if not glsl_code:
        return ''

    if not all(isinstance(i, str) for i in glsl_code):
        raise IOError('The only supported format are string.')

    if isinstance(glsl_code, str):
        return glsl_code

    code = ''
    for content in glsl_code:
        code += '\n'
        code += content
    return code


def import_fury_shader(shader_file):
    """Import a Fury shader.

    Parameters
    ----------
    shader_file : str
        Filename of shader. The file must be in the fury/shaders directory and
        must have the one of the supported extensions specified by the Khronos
        Group
        (https://github.com/KhronosGroup/glslang#execution-of-standalone-wrapper).

    Returns
    -------
    code : str
        GLSL shader code.

    """
    shader_fname = os.path.join(SHADERS_DIR, shader_file)
    return load_shader(shader_fname)


def load_shader(shader_file):
    """Load a shader from a file.

    Parameters
    ----------
    shader_file : str
        Full path to a shader file ending with one of the file extensions
        defined by the Khronos Group
        (https://github.com/KhronosGroup/glslang#execution-of-standalone-wrapper).

    Returns
    -------
    code : str
        GLSL shader code.

    """
    file_ext = os.path.splitext(os.path.basename(shader_file))[1]
    if file_ext not in SHADERS_EXTS:
        raise IOError(
            'Shader file "{}" does not have one of the supported '
            'extensions: {}.'.format(shader_file, SHADERS_EXTS)
        )
    return load_text(shader_file)


@deprecate_with_version(
    message='Load function has been reimplemented as import_fury_shader.',
    since='0.8.1',
    until='0.9.0',
)
def load(filename):
    """Load a Fury shader file.

    Parameters
    ----------
    filename : str
        Filename of the shader file.

    Returns
    -------
    code: str
        Shader code.

    """
    with open(os.path.join(SHADERS_DIR, filename)) as shader_file:
        return shader_file.read()


def shader_to_actor(
    actor,
    shader_type,
    impl_code='',
    decl_code='',
    block='valuepass',
    keep_default=True,
    replace_first=True,
    replace_all=False,
    debug=False,
):
    """Apply your own substitutions to the shader creation process.

    A set of string replacements is applied to a shader template. This
    function let's apply custom string replacements.

    Parameters
    ----------
    actor : vtkActor
        Fury actor you want to set the shader code to.
    shader_type : str
        Shader type: vertex, fragment
    impl_code : str, optional
        Shader implementation code, should be a string or filename. Default
        None.
    decl_code : str, optional
        Shader declaration code, should be a string or filename. Default None.
    block : str, optional
        Section name to be replaced. VTK use of heavy string replacements to
        insert shader and make it flexible. Each section of the shader
        template have a specific name. For more information:
        https://vtk.org/Wiki/Shaders_In_VTK. The possible values are:
        position, normal, light, tcoord, color, clip, camera, prim_id,
        valuepass. by default valuepass
    keep_default : bool, optional
        Keep the default block tag to let VTK replace it with its default
        behavior. Default True.
    replace_first : bool, optional
        If True, apply this change before the standard VTK replacements.
        Default True.
    replace_all : bool, optional
        [description], by default False
    debug : bool, optional
        Introduce a small error to debug shader code. Default False.

    """
    shader_type = shader_type.lower()
    shader_type = REPLACEMENT_SHADERS_TYPES.get(shader_type, None)
    if shader_type is None:
        msg = 'Invalid Shader Type. Please choose between '
        msg += ', '.join(REPLACEMENT_SHADERS_TYPES.keys())
        raise ValueError(msg)

    block = block.lower()
    block = SHADERS_BLOCK.get(block, None)
    if block is None:
        msg = 'Invalid Shader Type. Please choose between '
        msg += ', '.join(SHADERS_BLOCK.keys())
        raise ValueError(msg)

    block_dec = block + '::Dec'
    block_impl = block + '::Impl'

    if keep_default:
        decl_code = block_dec + '\n' + decl_code
        impl_code = block_impl + '\n' + impl_code

    if debug:
        enable_warnings()
        error_msg = '\n\n--- DEBUG: THIS LINE GENERATES AN ERROR ---\n\n'
        impl_code += error_msg

    sp = actor.GetShaderProperty()

    sp.AddShaderReplacement(
        shader_type, block_dec, replace_first, decl_code, replace_all
    )
    sp.AddShaderReplacement(
        shader_type, block_impl, replace_first, impl_code, replace_all
    )


def replace_shader_in_actor(actor, shader_type, code):
    """Set and replace the shader template with a new one.

    Parameters
    ----------
    actor : vtkActor
        Fury actor you want to set the shader code to.
    shader_type : str
        Shader type: vertex, geometry, fragment.
    code : str
        New shader template code.

    """
    function_name = {
        'vertex': 'SetVertexShaderCode',
        'fragment': 'SetFragmentShaderCode',
        'geometry': 'SetGeometryShaderCode',
    }
    shader_type = shader_type.lower()
    function = function_name.get(shader_type, None)
    if function is None:
        msg = 'Invalid Shader Type. Please choose between '
        msg += ', '.join(function_name.keys())
        raise ValueError(msg)

    sp = actor.GetShaderProperty()
    getattr(sp, function)(code)


def add_shader_callback(actor, callback, priority=0.0):
    """Add a shader callback to the actor.

    Parameters
    ----------
    actor : vtkActor
        Fury actor you want to add the callback to.
    callback : callable
        Function or class that contains 3 parameters: caller, event, calldata.
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
    --------
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
        raise TypeError(
            """
            add_shader_callback priority argument should be a float/int"""
        )

    mapper = actor.GetMapper()
    id_observer = mapper.AddObserver(Command.UpdateShaderEvent, cbk, priority)

    return id_observer


def shader_apply_effects(window, actor, effects, priority=0):
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

    def callback(_caller, _event, calldata=None, effects=None, window=None):
        program = calldata
        glState = window.GetState()
        if program is not None:
            for func in effects:
                func(glState)

    id_observer = add_shader_callback(
        actor, partial(callback, effects=effects, window=window), priority
    )

    return id_observer


def attribute_to_actor(actor, arr, attr_name, deep=True):
    """Link a numpy array with vertex attribute.

    Parameters
    ----------
    actor : vtkActor
        Fury actor you want to add the vertex attribute to.
    arr : ndarray
        Array to link to vertices.
    attr_name : str
        Vertex attribute name. The vtk array will take the same name as the
        attribute.
    deep : bool, optional
        If True a deep copy is applied, otherwise a shallow copy is applied.
        Default True.

    """
    nb_components = arr.shape[1] if arr.ndim > 1 else arr.ndim
    vtk_array = numpy_support.numpy_to_vtk(arr, deep=deep)
    vtk_array.SetNumberOfComponents(nb_components)
    vtk_array.SetName(attr_name)
    actor.GetMapper().GetInput().GetPointData().AddArray(vtk_array)
    mapper = actor.GetMapper()
    mapper.MapDataArrayToVertexAttribute(
        attr_name, attr_name, DataObject.FIELD_ASSOCIATION_POINTS, -1
    )
