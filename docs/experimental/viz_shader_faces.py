from fury import actor, window
from viz_shader_canvas import glyph_dot


import numpy as np
import vtk


scene = window.Scene()

colors = np.array([
    [.85, .07, .21], [.56, .14, .85], [.16, .65, .20], [.95, .73, .06],
    [.95, .55, .05], [.62, .42, .75], [.26, .58, .85], [.24, .82, .95],
    [.95, .78, .25], [.85, .58, .35], [1., 1., 1.]
])
n_points = len(colors)
np.random.seed(42)
centers = np.random.rand(n_points, 3)

canvas_actor = glyph_dot(centers, colors)
scene.add(actor.axes())
scene.add(canvas_actor)
#scene.background((1, 1, 1))
mapper = canvas_actor.GetMapper()

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    "//VTK::Light::Dec",
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
    False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    "//VTK::Light::Impl",
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
    False
)

window.show(scene)
