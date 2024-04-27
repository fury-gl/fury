from viz_shader_canvas import cube
import vtk

from fury import window

scene = window.Scene()
showm = window.ShowManager(scene)

canvas_actor = cube()
canvas_actor.GetProperty().BackfaceCullingOff()
scene.add(canvas_actor)
scene.background((1, 1, 1))
mapper = canvas_actor.GetMapper()

# Modify the vertex shader to pass the position of the vertex
mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    '//VTK::Normal::Dec',  # replace the normal block
    True,  # before the standard replacements
    """
    //VTK::Normal::Dec  // we still want the default
    out vec4 myVertexMC;
    """,
    False,
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,  # replace the normal block
    '//VTK::Normal::Impl',  # before the standard replacements
    True,  # we still want the default
    """
    //VTK::Normal::Impl
    myVertexMC = vertexMC;
    """,
    False,
)

# // Define varying and uniforms for the fragment shader here
mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,  # // in the fragment shader
    '//VTK::Normal::Dec',  # // replace the normal block
    True,  # // before the standard replacements
    """
    //VTK::Normal::Dec  // we still want the default
    varying vec4 myVertexMC;
    uniform float time;

    float width = .2;

    float rand(vec2 p) {
        p = fract(p * vec2(234.51, 124.89));
        p += dot(p, p + 54.23);
        p = fract(p * vec2(121.80, 456.12));
        p += dot(p, p + 25.12);
        return fract(p.x);
    }
    """,
    False,  # // only do it once
)

global timer
timer = 0


def timer_callback(obj, event):
    global timer
    timer += 1.0
    showm.render()
    # scene.azimuth(10)


@window.vtk.calldata_type(window.vtk.VTK_OBJECT)
def vtk_shader_callback(caller, event, calldata=None):
    program = calldata
    global timer
    if program is not None:
        try:
            program.SetUniformf('time', timer)
        except ValueError:
            pass


mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,  # // in the fragment shader
    '//VTK::Light::Impl',  # // replace the light block
    False,  # // after the standard replacements
    """
    //VTK::Light::Impl  // we still want the default calc
    //fragOutput0 = vec4(myVertexMC.xyz, 1.0);
    //fragOutput0 = vec4(normalVCVSOutput.xyz, 1.0);

    float tScale = .1; // Original .2

    vec2 pos = 5. * vec2(
        sin(time * tScale) + .1 * time, cos(time * tScale) + .1 * time);
    vec3 col = vec3(0.); /* .5 +
        .5 * cos(time + (fragCoord / iResolution.xy).xyx + vec3(0, 2, 4));*/

    for(float i=5.; i < 10.; i += 1.) {
        /*vec2 uv = pos + ((20. - 1.8 * i) *
            (fragCoord - .5 * iResolution.xy) / iResolution.y);*/
        vec2 uv = pos + ((20. - 1.8 * i) * myVertexMC.xy);
        vec2 gv = fract(uv) - .5;
        vec2 id = floor(uv);
        vec3 col2 = (.5 +
            .2 * sin(time + (i / 2.) + .3 * uv.xyx + vec3(0, 2, 4)) *
                sin(time + (i / 2.) +
            .3 * uv.xyx +
            vec3(0, 2, 4)) +
            .5 * cos(time + (i / 2.) + .3 * uv.xyx + vec3(0, 2, 4))) *
            (i + 1.) / 11.;

        gv.x *= (float(rand(id * i) > .5) - .5) * 2.;

        float mask1 = smoothstep(-.01, .01, width - abs(gv.x + gv.y - .5 *
            sign(gv.x + gv.y + .01)));
        float mask2 = smoothstep(-.2, .2, width - abs(gv.x + gv.y - .5 *
            sign(gv.x + gv.y + .01)));

        // Output to screen
        col = - .3 * mask2 + .5 * (col2.r * col2.r + col2.g * col2.g + col2.b *
            col2.b + col2 * col2) * col2 * mask1 + col * (1. - mask1);
    }
    fragOutput0 = vec4(col,1.0);
    """,
    False,
)


showm.add_timer_callback(True, 100, timer_callback)
showm.start()
