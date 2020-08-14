import vtk

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
}


def add_shader_to_actor(actor, shader_type, impl_code, decl_code=None,
                        block="", replace_first=True, replace_all=False,
                        debug=False):
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
    impl_code : str
        shader implementation code, should be a string or filename
    decl_code : str, optional
        shader declaration code, should be a string or filename
        by default None
    block : str, optional
        section name to be replaced, by default
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
        msg += ', '.join(SHADERS_TYPE.keys())
        raise ValueError(msg)

    block_dec = block + "::Dec"
    block_impl = block + "::Impl"

    if debug:
        error_msg = "\n\n--- DEBUG: THIS LINE GENERATES AN ERROR ---\n\n"
        impl_code += error_msg

    sp = actor.GetShaderProperty() if VTK_9_PLUS else actor.GetMapper()

    if decl_code is not None:
        sp.AddShaderReplacement(shader_type, block_dec, replace_first,
                                decl_code, replace_all)
    sp.AddShaderReplacement(shader_type, block_impl, replace_first,
                            impl_code, replace_all)


# def add_shader_callback(actor, ):
#     pass


# def add_vertex_attributes(actor):
#     pass