"""This simple example demonstrates how to use shaders to modify the fragments in
your scene. We will use the AddShaderReplacement() function to modify the
fragment shader with VTK's shader template system.

In this example, we will create a cube and use a fragment shader to modify
the color of the fragments.

This example borrows heavily from the FURY surfaces example.
http://fury.gl/dev/auto_examples/viz_surfaces.html
https://github.com/fury-gl/fury/blob/master/docs/examples/viz_surfaces.py

The code for the circle pattern is borrowed from The Book of Shaders.
https://thebookofshaders.com/09/
"""

import numpy as np

from fury import utils, window
from fury.utils import vtk

my_polydata = vtk.vtkPolyData()

my_vertices = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)

my_triangles = np.array(
    [
        [0, 6, 4],
        [0, 2, 6],
        [0, 3, 2],
        [0, 1, 3],
        [2, 7, 6],
        [2, 3, 7],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 7],
        [1, 7, 3],
    ],
    dtype='i8',
)

my_colors = my_vertices * 255  # transform from [0, 1] to [0, 255]

# use a FURY util to apply the above values to the polydata
utils.set_polydata_vertices(my_polydata, my_vertices)
utils.set_polydata_triangles(my_polydata, my_triangles)
utils.set_polydata_colors(my_polydata, my_colors)

# in VTK, shaders are applied at the mapper level
# get mapper from polydata
cube_actor = utils.get_actor_from_polydata(my_polydata)
mapper = cube_actor.GetMapper()

# add the cube to a scene and show it
scene = window.Scene()
scene.add(cube_actor)

scene.background((1, 1, 1))

window.show(scene, size=(500, 500))

# let's use a frag shader to change how the cube is rendered
# we'll render as usual if the fragment is far from the camera
# but we'll render small circles instead if the fragment is too close

# we'll need the window size for our circle effect, so we'll inject it into
# the shader as a uniform. uniforms are set using a callback so that their
# values can be updated

# first create a callback which gets the window size


@vtk.calldata_type(vtk.VTK_OBJECT)
def vtkShaderCallback(caller, event, calldata=None):
    window_size = scene.GetRenderWindow().GetSize()
    program = calldata
    if program is not None:
        program.SetUniform2f('windowSize', [window_size[0], window_size[1]])


# now register the event listener
mapper.AddObserver(vtk.vtkCommand.UpdateShaderEvent, vtkShaderCallback)

# now we augment VTK's default shaders with our own code
# first, declare the incoming uniform
# also define a function used to draw the circles
mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Color::Dec',  # target the Color block
    True,
    """
    // include the default
    //VTK::Color::Dec

    // declare the uniform
    uniform vec2 windowSize;

    // function to calculate if the fragment is inside a circle
    float circle(in vec2 _st, in float _radius){
        vec2 l = _st - vec2(0.5);
        return 1. - smoothstep(_radius-(_radius*0.01),
                               _radius+(_radius*0.01),
                               dot(l,l)*4.0);
    }
    """,
    False,
)

# now calculate the fragment color
mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Color::Impl',  # target the Color block
    True,
    """
    // include the default
    //VTK::Color::Impl

    // calculate values needed for circle rendering

    // normalized window coordinates
    vec2 st = vertexVCVSOutput.xy; //gl_FragCoord.xy / windowSize;
    st *= 20; // 50 = number circles in a row
    st = fract(st);

    // if close to camera, circlify it
    if (vertexVCVSOutput.z > -100) {
        diffuseColor *= vec3(circle(st, 0.25));
    }
    """,
    False,
)

# debug block
# uncomment this to force the shader code to print so you can see how your
# replacements are being inserted into the template
# mapper.AddShaderReplacement(
#     vtk.vtkShader.Fragment,
#     '//VTK::Coincident::Impl',
#     True,
#     '''
#     //VTK::Coincident::Impl
#     foo = bar;
#     ''',
#     False
# )

window.show(scene, size=(500, 500))
