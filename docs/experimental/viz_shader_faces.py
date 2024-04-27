import numpy as np
from viz_shader_canvas import glyph_dot
import vtk

from fury import actor, window

scene = window.Scene()

colors = np.array(
    [
        [0.85, 0.07, 0.21],
        [0.56, 0.14, 0.85],
        [0.16, 0.65, 0.20],
        [0.95, 0.73, 0.06],
        [0.95, 0.55, 0.05],
        [0.62, 0.42, 0.75],
        [0.26, 0.58, 0.85],
        [0.24, 0.82, 0.95],
        [0.95, 0.78, 0.25],
        [0.85, 0.58, 0.35],
        [1.0, 1.0, 1.0],
    ]
)
n_points = len(colors)
np.random.seed(42)
centers = np.random.rand(n_points, 3)

canvas_actor = glyph_dot(centers, colors)
scene.add(actor.axes())
scene.add(canvas_actor)
# scene.background((1, 1, 1))
mapper = canvas_actor.GetMapper()

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Light::Dec',
    True,
    """
    //VTK::Light::Dec
    varying vec4 myVertexMC;

    float circle(vec2 uv, vec2 p, float r, float blur) {
        float d = length(uv - p);
        float c = smoothstep(r, r - blur, d);
        return c;
    }
    """,
    False,
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Light::Impl',
    True,
    """
    vec3 color = vertexColorVSOutput.rgb;

    float xpos = 2 * gl_PointCoord.x - 1;
    float ypos = 1 - 2 * gl_PointCoord.y;
    vec2 p = vec2(xpos, ypos);

    // VTK Fake Spheres
    float p_len = length(p);
    if(p_len > 1) {
        discard;
    }

    // Smiley faces example
    float face = circle(p, vec2(0), 1, .05);
    face -= circle(p, vec2(-.35, .3), .2, .01);
    face -= circle(p, vec2(.35, .3), .2, .01);
    face -= circle(p, vec2(0.0, 0.0), .1, .01);

    float mouth = circle(p, vec2(0., 0.), .9, .02);
    mouth -= circle(p, vec2(0, .16), .9, .02);

    face -= mouth;

    fragOutput0 = vec4(color * face, 1);
    """,
    False,
)

window.show(scene)
