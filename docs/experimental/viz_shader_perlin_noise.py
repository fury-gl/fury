import numpy as np
from scipy.spatial import Delaunay
from viz_shader_canvas import surface
import vtk

from fury import window

scene = window.Scene()

size = 11
vertices = list()
for i in range(-size, size):
    for j in range(-size, size):
        fact1 = -np.sin(i) * np.cos(j)
        fact2 = -np.exp(abs(1 - np.sqrt(i**2 + j**2) / np.pi))
        z_coord = -abs(fact1 * fact2)
        vertices.append([i, j, z_coord])

c_arr = np.random.rand(len(vertices), 3)
np.random.shuffle(vertices)
vertices = np.array(vertices)
tri = Delaunay(vertices[:, [0, 1]])
faces = tri.simplices

shader_actor = surface(vertices, smooth='loop')
scene.add(shader_actor)
mapper = shader_actor.GetMapper()

fragment_shader_functions = """
//VTK::System::Dec // we still want the default
//Classic Perlin 3D Noise  // add functions for noise calculation
//by Stefan Gustavson

vec4 permute(vec4 x) {
    return mod(((x * 34.) + 1.0) * x, 289.);
}

vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159 - .85373472095314 * r;
}

vec4 fade(vec4 t) {
    return t * t * t * (t * (t * 6. - 15.) + 10.);
}

float noise(vec4 P) {
    vec4 Pi0 = floor(P); // Integer part for indexing
    vec4 Pi1 = Pi0 + 1.; // Integer part + 1
    Pi0 = mod(Pi0, 289.);
    Pi1 = mod(Pi1, 289.);
    vec4 Pf0 = fract(P); // Fractional part for interpolation
    vec4 Pf1 = Pf0 - 1.; // Fractional part - 1.
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = vec4(Pi0.zzzz);
    vec4 iz1 = vec4(Pi1.zzzz);
    vec4 iw0 = vec4(Pi0.wwww);
    vec4 iw1 = vec4(Pi1.wwww);

    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);
    vec4 ixy00 = permute(ixy0 + iw0);
    vec4 ixy01 = permute(ixy0 + iw1);
    vec4 ixy10 = permute(ixy1 + iw0);
    vec4 ixy11 = permute(ixy1 + iw1);

    vec4 gx00 = ixy00 / 7.;
    vec4 gy00 = floor(gx00) / 7.;
    vec4 gz00 = floor(gy00) / 6.;
    gx00 = fract(gx00) - .5;
    gy00 = fract(gy00) - .5;
    gz00 = fract(gz00) - .5;
    vec4 gw00 = vec4(.75) - abs(gx00) - abs(gy00) - abs(gz00);
    vec4 sw00 = step(gw00, vec4(0.0));
    gx00 -= sw00 * (step(0.0, gx00) - .5);
    gy00 -= sw00 * (step(0.0, gy00) - .5);

    vec4 gx01 = ixy01 / 7.;
    vec4 gy01 = floor(gx01) / 7.;
    vec4 gz01 = floor(gy01) / 6.;
    gx01 = fract(gx01) - .5;
    gy01 = fract(gy01) - .5;
    gz01 = fract(gz01) - .5;
    vec4 gw01 = vec4(.75) - abs(gx01) - abs(gy01) - abs(gz01);
    vec4 sw01 = step(gw01, vec4(0.0));
    gx01 -= sw01 * (step(0.0, gx01) - .5);
    gy01 -= sw01 * (step(0.0, gy01) - .5);

    vec4 gx10 = ixy10 / 7.;
    vec4 gy10 = floor(gx10) / 7.;
    vec4 gz10 = floor(gy10) / 6.;
    gx10 = fract(gx10) - .5;
    gy10 = fract(gy10) - .5;
    gz10 = fract(gz10) - .5;
    vec4 gw10 = vec4(.75) - abs(gx10) - abs(gy10) - abs(gz10);
    vec4 sw10 = step(gw10, vec4(0.0));
    gx10 -= sw10 * (step(0.0, gx10) - .5);
    gy10 -= sw10 * (step(0.0, gy10) - .5);

    vec4 gx11 = ixy11 / 7.;
    vec4 gy11 = floor(gx11) / 7.;
    vec4 gz11 = floor(gy11) / 6.;
    gx11 = fract(gx11) - .5;
    gy11 = fract(gy11) - .5;
    gz11 = fract(gz11) - .5;
    vec4 gw11 = vec4(.75) - abs(gx11) - abs(gy11) - abs(gz11);
    vec4 sw11 = step(gw11, vec4(0.0));
    gx11 -= sw11 * (step(0.0, gx11) - .5);
    gy11 -= sw11 * (step(0.0, gy11) - .5);

    vec4 g0000 = vec4(gx00.x, gy00.x, gz00.x, gw00.x);
    vec4 g1000 = vec4(gx00.y, gy00.y, gz00.y, gw00.y);
    vec4 g0100 = vec4(gx00.z, gy00.z, gz00.z, gw00.z);
    vec4 g1100 = vec4(gx00.w, gy00.w, gz00.w, gw00.w);
    vec4 g0010 = vec4(gx10.x, gy10.x, gz10.x, gw10.x);
    vec4 g1010 = vec4(gx10.y, gy10.y, gz10.y, gw10.y);
    vec4 g0110 = vec4(gx10.z, gy10.z, gz10.z, gw10.z);
    vec4 g1110 = vec4(gx10.w, gy10.w, gz10.w, gw10.w);
    vec4 g0001 = vec4(gx01.x, gy01.x, gz01.x, gw01.x);
    vec4 g1001 = vec4(gx01.y, gy01.y, gz01.y, gw01.y);
    vec4 g0101 = vec4(gx01.z, gy01.z, gz01.z, gw01.z);
    vec4 g1101 = vec4(gx01.w, gy01.w, gz01.w, gw01.w);
    vec4 g0011 = vec4(gx11.x, gy11.x, gz11.x, gw11.x);
    vec4 g1011 = vec4(gx11.y, gy11.y, gz11.y, gw11.y);
    vec4 g0111 = vec4(gx11.z, gy11.z, gz11.z, gw11.z);
    vec4 g1111 = vec4(gx11.w, gy11.w, gz11.w, gw11.w);

    vec4 norm00 = taylorInvSqrt(vec4(dot(g0000, g0000), dot(g0100, g0100),
        dot(g1000, g1000), dot(g1100, g1100)));
    g0000 *= norm00.x;
    g0100 *= norm00.y;
    g1000 *= norm00.z;
    g1100 *= norm00.w;

    vec4 norm01 = taylorInvSqrt(vec4(dot(g0001, g0001), dot(g0101, g0101),
        dot(g1001, g1001), dot(g1101, g1101)));
    g0001 *= norm01.x;
    g0101 *= norm01.y;
    g1001 *= norm01.z;
    g1101 *= norm01.w;

    vec4 norm10 = taylorInvSqrt(vec4(dot(g0010, g0010), dot(g0110, g0110),
        dot(g1010, g1010), dot(g1110, g1110)));
    g0010 *= norm10.x;
    g0110 *= norm10.y;
    g1010 *= norm10.z;
    g1110 *= norm10.w;

    vec4 norm11 = taylorInvSqrt(vec4(dot(g0011, g0011), dot(g0111, g0111),
        dot(g1011, g1011), dot(g1111, g1111)));
    g0011 *= norm11.x;
    g0111 *= norm11.y;
    g1011 *= norm11.z;
    g1111 *= norm11.w;

    float n0000 = dot(g0000, Pf0);
    float n1000 = dot(g1000, vec4(Pf1.x, Pf0.yzw));
    float n0100 = dot(g0100, vec4(Pf0.x, Pf1.y, Pf0.zw));
    float n1100 = dot(g1100, vec4(Pf1.xy, Pf0.zw));
    float n0010 = dot(g0010, vec4(Pf0.xy, Pf1.z, Pf0.w));
    float n1010 = dot(g1010, vec4(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
    float n0110 = dot(g0110, vec4(Pf0.x, Pf1.yz, Pf0.w));
    float n1110 = dot(g1110, vec4(Pf1.xyz, Pf0.w));
    float n0001 = dot(g0001, vec4(Pf0.xyz, Pf1.w));
    float n1001 = dot(g1001, vec4(Pf1.x, Pf0.yz, Pf1.w));
    float n0101 = dot(g0101, vec4(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
    float n1101 = dot(g1101, vec4(Pf1.xy, Pf0.z, Pf1.w));
    float n0011 = dot(g0011, vec4(Pf0.xy, Pf1.zw));
    float n1011 = dot(g1011, vec4(Pf1.x, Pf0.y, Pf1.zw));
    float n0111 = dot(g0111, vec4(Pf0.x, Pf1.yzw));
    float n1111 = dot(g1111, Pf1);

    vec4 fade_xyzw = fade(Pf0);
    vec4 n_0w = mix(vec4(n0000, n1000, n0100, n1100),
        vec4(n0001, n1001, n0101, n1101), fade_xyzw.w);
    vec4 n_1w = mix(vec4(n0010, n1010, n0110, n1110),
        vec4(n0011, n1011, n0111, n1111), fade_xyzw.w);
    vec4 n_zw = mix(n_0w, n_1w, fade_xyzw.z);
    vec2 n_yzw = mix(n_zw.xy, n_zw.zw, fade_xyzw.y);
    float n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
    return 2.2 * n_xyzw;
}
"""

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

# // Add the code to generate noise
#   // These functions need to be defined outside of main. Use the System::Dec
#   // to declare and implement
mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::System::Dec',
    False,  # before the standard replacements
    fragment_shader_functions,
    False,  # only do it once
)

# // Define varying and uniforms for the fragment shader here
mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,  # // in the fragment shader
    '//VTK::Normal::Dec',  # // replace the normal block
    True,  # // before the standard replacements
    """
    //VTK::Normal::Dec  // we still want the default
    varying vec4 myVertexMC;
    uniform float k = 1.;
    """,
    False,  # // only do it once
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,  # // in the fragment shader
    '//VTK::Light::Impl',  # // replace the light block
    False,  # // after the standard replacements
    """
    //VTK::Light::Impl  // we still want the default calc
    #define pnoise(x) ((noise(x) + 1.) / 2.)
    vec3 noisyColor;
    noisyColor.r = noise(k * 10. * myVertexMC);
    noisyColor.g = noise(k * 11. * myVertexMC);
    noisyColor.b = noise(k * 12. * myVertexMC);

    // map ranges of noise values into different colors

    int i;
    float lowerValue = .3;
    float upperValue = .6;
    for(i=0; i<3; i+=1) {
        noisyColor[i] = (noisyColor[i] + 1.) / 2.;
        if(noisyColor[i] < lowerValue) {
            noisyColor[i] = lowerValue;
        } else {
            if(noisyColor[i] < upperValue) {
                noisyColor[i] = upperValue;
            } else {
                noisyColor[i] = 1.;
            }
        }
    }
    fragOutput0.rgb = opacity *
        vec3(ambientColor + noisyColor * diffuse + specular);
    fragOutput0.a = opacity;
    """,
    False,
)

window.show(scene)
