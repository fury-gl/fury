"""
This script includes TEXTURE experimentation for passing SH coefficients
"""
import os

import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf

from fury import actor, window
from fury.lib import FloatArray, Texture
from fury.shaders import (
    attribute_to_actor,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)
from fury.utils import numpy_to_vtk_image_data, set_polydata_tcoords

if __name__ == "__main__":
    show_man = window.ShowManager(size=(1920, 1080))
    show_man.scene.background((1, 1, 1))

    # fmt: off
    coeffs = np.array([
        [
            0.2820735, 0.15236554, -0.04038717, -0.11270988, -0.04532376,
            0.14921817, 0.00257928, 0.0040734, -0.05313807, 0.03486542,
            0.04083064, 0.02105767, -0.04389586, -0.04302812, 0.1048641
        ],
        [
            0.28549338, 0.0978267, -0.11544838, 0.12525354, -0.00126003,
            0.00320594, 0.04744155, -0.07141446, 0.03211689, 0.04711322,
            0.08064896, 0.00154299, 0.00086506, 0.00162543, -0.00444893
        ],
        [
            0.28208936, -0.13133252, -0.04701012, -0.06303016, -0.0468775,
            0.02348355, 0.03991898, 0.02587433, 0.02645416, 0.00668765,
            0.00890633, 0.02189304, 0.00387415, 0.01665629, -0.01427194
        ]
    ])
    # fmt: on

    centers = np.array([[0, -1, 0], [1, -1, 0], [2, -1, 0]])
    scales = np.array([1, 2, 2])

    odf_actor = actor.box(centers=centers, scales=1.0)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(odf_actor, big_scales, "scale")
    
    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, 'minmax')

    odf_actor_pd = odf_actor.GetMapper().GetInput()

    # fmt: off
    uv_vals = np.array(
        [
            [0, 2 / 3], [0, 1], [1, 1], [1, 2 / 3],
            [0, 2 / 3], [0, 1], [1, 1], [1, 2 / 3],  # glyph1
            [0, 1 / 3], [0, 2 / 3], [1, 2 / 3], [1, 1 / 3],
            [0, 1 / 3], [0, 2 / 3], [1, 2 / 3], [1, 1 / 3],  # glyph2
            [0, 0], [0, 1 / 3], [1, 1 / 3], [1, 0],
            [0, 0], [0, 1 / 3], [1, 1 / 3], [1, 0]  # glyph3
        ]
    ) + [[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1],
         [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1],
         [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1],
         [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1],
         [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1],
         [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]]
    # fmt: on

    num_pnts = uv_vals.shape[0]

    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pnts)
    [t_coords.SetTuple(i, uv_vals[i]) for i in range(num_pnts)]

    set_polydata_tcoords(odf_actor_pd, t_coords)

    min = coeffs.min(axis=1)
    max = coeffs.max(axis=1)
    newmin = 0
    newmax = 1
    arr = np.array([(coeffs[i] - min[i])*((newmax - newmin) / (max[i] - min[i])) + newmin for i in range(coeffs.shape[0])])
    arr *= 255
    grid = numpy_to_vtk_image_data(arr.astype(np.uint8))

    texture = Texture()
    texture.SetInputDataObject(grid)
    texture.Update()

    odf_actor.GetProperty().SetTexture("texture0", texture)

    # TODO: Set int uniform
    odf_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf(
        "numCoeffs", 15
    )

    vs_dec = """
    in vec3 center;
    in float scale;
    in vec2 minmax;

    out vec4 vertexMCVSOutput;
    out vec3 centerMCVSOutput;
    out float scaleVSOutput;
    out vec2 minmaxVSOutput;
    """

    vs_impl = """
    vertexMCVSOutput = vertexMC;
    centerMCVSOutput = center;
    scaleVSOutput = scale;
    minmaxVSOutput = minmax;
    vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    """

    shader_to_actor(odf_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    fs_defs = "#define PI 3.1415926535898"

    fs_unifs = """
    uniform mat4 MCVCMatrix;
    uniform samplerCube texture_0;
    //uniform int k;
    """

    fs_vs_vars = """
    in vec4 vertexMCVSOutput;
    in vec3 centerMCVSOutput;
    in float scaleVSOutput;
    in vec2 minmaxVSOutput;
    """

    coeffs_norm = """
    float coeffsNorm(float coef)
    {
        float min = 0;
        float max = 1;
        float newMin = minmaxVSOutput.x;
        float newMax = minmaxVSOutput.y;
        return (coef - min) * ((newMax - newMin) / (max - min)) + newMin;
    }
    """

    # Associated Legendre Polynomial
    legendre_polys = """
    int factorial(int v)
    {
        int t = 1;
        for(int i = 2; i <= v; i++)
        {
            t *= i;
        }
        return t;
    }

    // Adapted from https://patapom.com/blog/SHPortal/
    // "Evaluate an Associated Legendre Polynomial P(l,m,x) at x
    // For more, see “Numerical Methods in C: The Art of Scientific Computing”, Cambridge University Press, 1992, pp 252-254" 
    float P(int l, int m, float x )
    {
        float pmm = 1.0;
        
        float somx2 = sqrt((1.0-x)*(1.0+x));
        float fact = 1.0;
        for ( int i=1; i<=m; i++ ) {
            pmm *= (-fact) * somx2;
            fact += 2.0;
        }
        
        if( l == m )
            return pmm;

        float pmmp1 = x * (2.0*float(m)+1.0) * pmm;
        if ( l == m+1 )
            return pmmp1;

        float pll = 0.0;
        for ( float ll=float(m+2); ll<=float(l); ll+=1.0 ) {
            pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+float(m)-1.0)*pmm ) / (ll-float(m));
            pmm = pmmp1;
            pmmp1 = pll;
        }

        return pll;
    }


    float K(int l, int m)
    {
        float n = float((2*l+1)*factorial(l - m));
        float d = 4.0 * PI * float(factorial(l + m));
        return sqrt(n/d);
    }
    """

    spherical_harmonics = """
    float SH(int l, int m, in vec3 s)
    {
        vec3 ns = normalize(s);
        float thetax = ns.y;
        float phi = atan(ns.z, ns.x)+PI/2.;
        float v = K(l, abs(m)) * P(l, abs(m), thetax);
        if(m != 0)
            v *= sqrt(2.0);
        if(m > 0)
            v *= sin(float(m)*phi);
        if(m < 0)
            v *= cos(float(-m)*phi);
        
        return v;
    }
    """

    sdf_map = """
    vec3 map( in vec3 p )
    {
        p = p - centerMCVSOutput;
        vec3 p00 = p;

        float r, d; vec3 n, s, res;

        #define SHAPE (vec3(d-abs(r), sign(r),d))
        //#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))
        d=length(p00);
        n=p00 / d;
        // ================================================================
        float i = 1 / (numCoeffs * 2);

        float c = texture(texture0, vec2(i, tcoordVCVSOutput.y)).x;
        r = coeffsNorm(c) * SH(0, 0, n);

        c = texture(texture0, vec2(i + 1 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, -2, n);

        c = texture(texture0, vec2(i + 2 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, -1, n);

        c = texture(texture0, vec2(i + 3 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, 0, n);

        c = texture(texture0, vec2(i + 4 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, 1, n);

        c = texture(texture0, vec2(i + 5 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, 2, n);

        c = texture(texture0, vec2(i + 6 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, -4, n);

        c = texture(texture0, vec2(i + 7 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, -3, n);

        c = texture(texture0, vec2(i + 8 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, -2, n);

        c = texture(texture0, vec2(i + 9 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, -1, n);

        c = texture(texture0, vec2(i + 10 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 0, n);

        c = texture(texture0, vec2(i + 11 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 1, n);

        c = texture(texture0, vec2(i + 12 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 2, n);

        c = texture(texture0, vec2(i + 13 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 3, n);

        c = texture(texture0, vec2(i + 14 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 4, n);

        r *= scaleVSOutput;
        // ================================================================
        s = SHAPE;
        res = s;
        return vec3(res.x, .5 + .5 * res.y, res.z);
    }
    """

    central_diffs_normals = """
    vec3 centralDiffsNormals(in vec3 pos)
    {
        //vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
        vec2 e = vec2(.001, -1);
        return normalize(
            e.xyy * map(pos + e.xyy).x + e.yyx * map(pos + e.yyx).x +
            e.yxy * map(pos + e.yxy).x + e.xxx * map(pos + e.xxx).x );
    }
    """

    """
    central_diffs_normals = import_fury_shader(
        os.path.join("sdf", "central_diffs.frag")
    )
    """

    cast_ray = """
    vec3 castRay(in vec3 ro, vec3 rd)
    {
        vec3 res = vec3(1e10, -1, 1);

        float maxd = 4;
        float h = 1;
        float t = 0;
        vec2  m = vec2(-1);

        for(int i = 0; i < 2000; i++)
        {
            if(h < .01 || t > maxd)
                break;
            vec3 res = map(ro + rd * t);
            h = res.x;
            m = res.yz;
            t += h * .1;
        }

        if(t < maxd && t < res.x)
            res = vec3(t, m);

        return res;
    }
    """

    blinn_phong_model = import_fury_shader(
        os.path.join("lighting", "blinn_phong_model.frag")
    )
    
    common = """
    void eval_sh_2(out float out_shs[6], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
}


void eval_sh_4(out float out_shs[15], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
}


void eval_sh_6(out float out_shs[28], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    out_shs[21] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[20] = -c1 * d;
    out_shs[22] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[19] = c0 * d;
    out_shs[23] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[18] = -c1 * d;
    out_shs[24] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[17] = c0 * d;
    out_shs[25] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[16] = -c1 * d;
    out_shs[26] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[15] = c0 * d;
    out_shs[27] = s0 * d;
}


void eval_sh_8(out float out_shs[45], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    out_shs[21] = d;
    b = z2 * a - 0.251748252 * b;
    a = b - 0.251282051 * a;
    d = 58.473368113 * a;
    out_shs[36] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[20] = -c1 * d;
    out_shs[22] = s1 * d;
    a = z2 * b - 0.244755245 * a;
    b = a - 0.246153846 * b;
    d = -77.964490818 * b;
    out_shs[35] = -c1 * d;
    out_shs[37] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[19] = c0 * d;
    out_shs[23] = s0 * d;
    b = z2 * a - 0.223776224 * b;
    a = b - 0.230769231 * a;
    d = 65.229772956 * a;
    out_shs[34] = c0 * d;
    out_shs[38] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[18] = -c1 * d;
    out_shs[24] = s1 * d;
    a = z2 * b - 0.188811189 * a;
    b = a - 0.205128205 * b;
    d = -48.175380057 * b;
    out_shs[33] = -c1 * d;
    out_shs[39] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[17] = c0 * d;
    out_shs[25] = s0 * d;
    b = z2 * (a - 0.13986014);
    a = b - 0.169230769 * a;
    d = 31.097074109 * a;
    out_shs[32] = c0 * d;
    out_shs[40] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[16] = -c1 * d;
    out_shs[26] = s1 * d;
    a = (z2 - 7.69230769e-02) * z;
    b = a - 0.123076923 * z;
    d = -17.24955311 * b;
    out_shs[31] = -c1 * d;
    out_shs[41] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[15] = c0 * d;
    out_shs[27] = s0 * d;
    a = z2 - 6.66666667e-02;
    d = 7.984991491 * a;
    out_shs[30] = c0 * d;
    out_shs[42] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.915706641 * z;
    out_shs[29] = -c1 * d;
    out_shs[43] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.72892666;
    out_shs[28] = c0 * d;
    out_shs[44] = s0 * d;
}


void eval_sh_10(out float out_shs[66], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    out_shs[21] = d;
    b = z2 * a - 0.251748252 * b;
    a = b - 0.251282051 * a;
    d = 58.473368113 * a;
    out_shs[36] = d;
    b = z2 * a - 0.250980392 * b;
    a = b - 0.250773994 * a;
    d = 233.240148813 * a;
    out_shs[55] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[20] = -c1 * d;
    out_shs[22] = s1 * d;
    a = z2 * b - 0.244755245 * a;
    b = a - 0.246153846 * b;
    d = -77.964490818 * b;
    out_shs[35] = -c1 * d;
    out_shs[37] = s1 * d;
    a = z2 * b - 0.247058824 * a;
    b = a - 0.247678019 * b;
    d = -314.500952502 * b;
    out_shs[54] = -c1 * d;
    out_shs[56] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[19] = c0 * d;
    out_shs[23] = s0 * d;
    b = z2 * a - 0.223776224 * b;
    a = b - 0.230769231 * a;
    d = 65.229772956 * a;
    out_shs[34] = c0 * d;
    out_shs[38] = s0 * d;
    b = z2 * a - 0.235294118 * b;
    a = b - 0.238390093 * a;
    d = 272.365814381 * a;
    out_shs[53] = c0 * d;
    out_shs[57] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[18] = -c1 * d;
    out_shs[24] = s1 * d;
    a = z2 * b - 0.188811189 * a;
    b = a - 0.205128205 * b;
    d = -48.175380057 * b;
    out_shs[33] = -c1 * d;
    out_shs[39] = s1 * d;
    a = z2 * b - 0.215686275 * a;
    b = a - 0.222910217 * b;
    d = -213.661323441 * b;
    out_shs[52] = -c1 * d;
    out_shs[58] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[17] = c0 * d;
    out_shs[25] = s0 * d;
    b = z2 * (a - 0.13986014);
    a = b - 0.169230769 * a;
    d = 31.097074109 * a;
    out_shs[32] = c0 * d;
    out_shs[40] = s0 * d;
    b = z2 * a - 0.188235294 * b;
    a = b - 0.20123839 * a;
    d = 151.081370682 * a;
    out_shs[51] = c0 * d;
    out_shs[59] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[16] = -c1 * d;
    out_shs[26] = s1 * d;
    a = (z2 - 7.69230769e-02) * z;
    b = a - 0.123076923 * z;
    d = -17.24955311 * b;
    out_shs[31] = -c1 * d;
    out_shs[41] = s1 * d;
    a = z2 * b - 0.152941176 * a;
    b = a - 0.173374613 * b;
    d = -95.552248675 * b;
    out_shs[50] = -c1 * d;
    out_shs[60] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[15] = c0 * d;
    out_shs[27] = s0 * d;
    a = z2 - 6.66666667e-02;
    d = 7.984991491 * a;
    out_shs[30] = c0 * d;
    out_shs[42] = s0 * d;
    b = z2 * (a - 0.109803922);
    a = b - 0.139318885 * a;
    d = 53.41533086 * a;
    out_shs[49] = c0 * d;
    out_shs[61] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.915706641 * z;
    out_shs[29] = -c1 * d;
    out_shs[43] = s1 * d;
    a = (z2 - 5.88235294e-02) * z;
    b = a - 9.90712074e-02 * z;
    d = -25.910241313 * b;
    out_shs[48] = -c1 * d;
    out_shs[62] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.72892666;
    out_shs[28] = c0 * d;
    out_shs[44] = s0 * d;
    a = z2 - 5.26315789e-02;
    d = 10.577811722 * a;
    out_shs[47] = c0 * d;
    out_shs[63] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -3.4318953 * z;
    out_shs[46] = -c1 * d;
    out_shs[64] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.767395118;
    out_shs[45] = c0 * d;
    out_shs[65] = s0 * d;
}


void eval_sh_12(out float out_shs[91], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    out_shs[21] = d;
    b = z2 * a - 0.251748252 * b;
    a = b - 0.251282051 * a;
    d = 58.473368113 * a;
    out_shs[36] = d;
    b = z2 * a - 0.250980392 * b;
    a = b - 0.250773994 * a;
    d = 233.240148813 * a;
    out_shs[55] = d;
    b = z2 * a - 0.250626566 * b;
    a = b - 0.250517598 * a;
    d = 931.186918633 * a;
    out_shs[78] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[20] = -c1 * d;
    out_shs[22] = s1 * d;
    a = z2 * b - 0.244755245 * a;
    b = a - 0.246153846 * b;
    d = -77.964490818 * b;
    out_shs[35] = -c1 * d;
    out_shs[37] = s1 * d;
    a = z2 * b - 0.247058824 * a;
    b = a - 0.247678019 * b;
    d = -314.500952502 * b;
    out_shs[54] = -c1 * d;
    out_shs[56] = s1 * d;
    a = z2 * b - 0.248120301 * a;
    b = a - 0.248447205 * b;
    d = -1265.233874957 * b;
    out_shs[77] = -c1 * d;
    out_shs[79] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[19] = c0 * d;
    out_shs[23] = s0 * d;
    b = z2 * a - 0.223776224 * b;
    a = b - 0.230769231 * a;
    d = 65.229772956 * a;
    out_shs[34] = c0 * d;
    out_shs[38] = s0 * d;
    b = z2 * a - 0.235294118 * b;
    a = b - 0.238390093 * a;
    d = 272.365814381 * a;
    out_shs[53] = c0 * d;
    out_shs[57] = s0 * d;
    b = z2 * a - 0.240601504 * b;
    a = b - 0.242236025 * a;
    d = 1121.509962433 * a;
    out_shs[76] = c0 * d;
    out_shs[80] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[18] = -c1 * d;
    out_shs[24] = s1 * d;
    a = z2 * b - 0.188811189 * a;
    b = a - 0.205128205 * b;
    d = -48.175380057 * b;
    out_shs[33] = -c1 * d;
    out_shs[39] = s1 * d;
    a = z2 * b - 0.215686275 * a;
    b = a - 0.222910217 * b;
    d = -213.661323441 * b;
    out_shs[52] = -c1 * d;
    out_shs[58] = s1 * d;
    a = z2 * b - 0.228070175 * a;
    b = a - 0.231884058 * b;
    d = -915.709049803 * b;
    out_shs[75] = -c1 * d;
    out_shs[81] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[17] = c0 * d;
    out_shs[25] = s0 * d;
    b = z2 * (a - 0.13986014);
    a = b - 0.169230769 * a;
    d = 31.097074109 * a;
    out_shs[32] = c0 * d;
    out_shs[40] = s0 * d;
    b = z2 * a - 0.188235294 * b;
    a = b - 0.20123839 * a;
    d = 151.081370682 * a;
    out_shs[51] = c0 * d;
    out_shs[59] = s0 * d;
    b = z2 * a - 0.210526316 * b;
    a = b - 0.217391304 * a;
    d = 686.781787352 * a;
    out_shs[74] = c0 * d;
    out_shs[82] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[16] = -c1 * d;
    out_shs[26] = s1 * d;
    a = (z2 - 7.69230769e-02) * z;
    b = a - 0.123076923 * z;
    d = -17.24955311 * b;
    out_shs[31] = -c1 * d;
    out_shs[41] = s1 * d;
    a = z2 * b - 0.152941176 * a;
    b = a - 0.173374613 * b;
    d = -95.552248675 * b;
    out_shs[50] = -c1 * d;
    out_shs[60] = s1 * d;
    a = z2 * b - 0.187969925 * a;
    b = a - 0.198757764 * b;
    d = -471.12841933 * b;
    out_shs[73] = -c1 * d;
    out_shs[83] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[15] = c0 * d;
    out_shs[27] = s0 * d;
    a = z2 - 6.66666667e-02;
    d = 7.984991491 * a;
    out_shs[30] = c0 * d;
    out_shs[42] = s0 * d;
    b = z2 * (a - 0.109803922);
    a = b - 0.139318885 * a;
    d = 53.41533086 * a;
    out_shs[49] = c0 * d;
    out_shs[61] = s0 * d;
    b = z2 * a - 0.160401003 * b;
    a = b - 0.175983437 * a;
    d = 293.800188384 * a;
    out_shs[72] = c0 * d;
    out_shs[84] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.915706641 * z;
    out_shs[29] = -c1 * d;
    out_shs[43] = s1 * d;
    a = (z2 - 5.88235294e-02) * z;
    b = a - 9.90712074e-02 * z;
    d = -25.910241313 * b;
    out_shs[48] = -c1 * d;
    out_shs[62] = s1 * d;
    a = z2 * b - 0.127819549 * a;
    b = a - 0.149068323 * b;
    d = -165.101452729 * b;
    out_shs[71] = -c1 * d;
    out_shs[85] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.72892666;
    out_shs[28] = c0 * d;
    out_shs[44] = s0 * d;
    a = z2 - 5.26315789e-02;
    d = 10.577811722 * a;
    out_shs[47] = c0 * d;
    out_shs[63] = s0 * d;
    b = z2 * (a - 9.02255639e-02);
    a = b - 0.118012422 * a;
    d = 82.550726364 * a;
    out_shs[70] = c0 * d;
    out_shs[86] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -3.4318953 * z;
    out_shs[46] = -c1 * d;
    out_shs[64] = s1 * d;
    a = (z2 - 4.76190476e-02) * z;
    b = a - 8.2815735e-02 * z;
    d = -36.028090689 * b;
    out_shs[69] = -c1 * d;
    out_shs[87] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.767395118;
    out_shs[45] = c0 * d;
    out_shs[65] = s0 * d;
    a = z2 - 4.34782609e-02;
    d = 13.3042542 * a;
    out_shs[68] = c0 * d;
    out_shs[88] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -3.923210529 * z;
    out_shs[67] = -c1 * d;
    out_shs[89] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.800821996;
    out_shs[66] = c0 * d;
    out_shs[90] = s0 * d;
}


void eval_sh_grad_2(out float out_shs[6], out vec3 out_grads[6], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    c0 = 1.0;
    s0 = 0.0;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    out_grads[2][0] = -c0 * d;
    out_grads[4][0] = s0 * d;
    out_grads[2][1] = s0 * d;
    out_grads[4][1] = c0 * d;
    d = 1.892349392 * z;
    out_grads[3][2] = d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    d = 1.092548431;
    out_grads[1][0] = c1 * d;
    out_grads[5][0] = s1 * d;
    out_grads[1][1] = -s1 * d;
    out_grads[5][1] = c1 * d;
    d = -1.092548431;
    out_grads[2][2] = -c1 * d;
    out_grads[4][2] = s1 * d;
    out_grads[0][0] = 0.0;
    out_grads[0][1] = 0.0;
    out_grads[0][2] = 0.0;
    out_grads[1][2] = 0.0;
    out_grads[3][0] = 0.0;
    out_grads[3][1] = 0.0;
    out_grads[5][2] = 0.0;
}


void eval_sh_grad_4(out float out_shs[15], out vec3 out_grads[15], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    c0 = 1.0;
    s0 = 0.0;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    out_grads[2][0] = -c0 * d;
    out_grads[4][0] = s0 * d;
    out_grads[2][1] = s0 * d;
    out_grads[4][1] = c0 * d;
    d = 1.892349392 * z;
    out_grads[3][2] = d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    out_grads[9][0] = -c0 * d;
    out_grads[11][0] = s0 * d;
    out_grads[9][1] = s0 * d;
    out_grads[11][1] = c0 * d;
    d = 14.809976568 * b;
    out_grads[10][2] = d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    d = 1.092548431;
    out_grads[1][0] = c1 * d;
    out_grads[5][0] = s1 * d;
    out_grads[1][1] = -s1 * d;
    out_grads[5][1] = c1 * d;
    d = -1.092548431;
    out_grads[2][2] = -c1 * d;
    out_grads[4][2] = s1 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    d = 6.62322287 * a;
    out_grads[8][0] = c1 * d;
    out_grads[12][0] = s1 * d;
    out_grads[8][1] = -s1 * d;
    out_grads[12][1] = c1 * d;
    d = -14.049977415 * a;
    out_grads[9][2] = -c1 * d;
    out_grads[11][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    d = -5.310392309 * z;
    out_grads[7][0] = -c0 * d;
    out_grads[13][0] = s0 * d;
    out_grads[7][1] = s0 * d;
    out_grads[13][1] = c0 * d;
    d = 6.62322287 * z;
    out_grads[8][2] = c0 * d;
    out_grads[12][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
    d = 2.503342942;
    out_grads[6][0] = c1 * d;
    out_grads[14][0] = s1 * d;
    out_grads[6][1] = -s1 * d;
    out_grads[14][1] = c1 * d;
    d = -1.77013077;
    out_grads[7][2] = -c1 * d;
    out_grads[13][2] = s1 * d;
    out_grads[0][0] = 0.0;
    out_grads[0][1] = 0.0;
    out_grads[0][2] = 0.0;
    out_grads[1][2] = 0.0;
    out_grads[3][0] = 0.0;
    out_grads[3][1] = 0.0;
    out_grads[5][2] = 0.0;
    out_grads[6][2] = 0.0;
    out_grads[10][0] = 0.0;
    out_grads[10][1] = 0.0;
    out_grads[14][2] = 0.0;
}


void eval_sh_grad_6(out float out_shs[28], out vec3 out_grads[28], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    c0 = 1.0;
    s0 = 0.0;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    out_shs[21] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    out_grads[2][0] = -c0 * d;
    out_grads[4][0] = s0 * d;
    out_grads[2][1] = s0 * d;
    out_grads[4][1] = c0 * d;
    d = 1.892349392 * z;
    out_grads[3][2] = d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    out_grads[9][0] = -c0 * d;
    out_grads[11][0] = s0 * d;
    out_grads[9][1] = s0 * d;
    out_grads[11][1] = c0 * d;
    d = 14.809976568 * b;
    out_grads[10][2] = d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[20] = -c1 * d;
    out_shs[22] = s1 * d;
    out_grads[20][0] = -c0 * d;
    out_grads[22][0] = s0 * d;
    out_grads[20][1] = s0 * d;
    out_grads[22][1] = c0 * d;
    d = 88.106914343 * b;
    out_grads[21][2] = d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    d = 1.092548431;
    out_grads[1][0] = c1 * d;
    out_grads[5][0] = s1 * d;
    out_grads[1][1] = -s1 * d;
    out_grads[5][1] = c1 * d;
    d = -1.092548431;
    out_grads[2][2] = -c1 * d;
    out_grads[4][2] = s1 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    d = 6.62322287 * a;
    out_grads[8][0] = c1 * d;
    out_grads[12][0] = s1 * d;
    out_grads[8][1] = -s1 * d;
    out_grads[12][1] = c1 * d;
    d = -14.049977415 * a;
    out_grads[9][2] = -c1 * d;
    out_grads[11][2] = s1 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[19] = c0 * d;
    out_shs[23] = s0 * d;
    d = 30.399773564 * a;
    out_grads[19][0] = c1 * d;
    out_grads[23][0] = s1 * d;
    out_grads[19][1] = -s1 * d;
    out_grads[23][1] = c1 * d;
    d = -96.132524816 * a;
    out_grads[20][2] = -c1 * d;
    out_grads[22][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    d = -5.310392309 * z;
    out_grads[7][0] = -c0 * d;
    out_grads[13][0] = s0 * d;
    out_grads[7][1] = s0 * d;
    out_grads[13][1] = c0 * d;
    d = 6.62322287 * z;
    out_grads[8][2] = c0 * d;
    out_grads[12][2] = s0 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[18] = -c1 * d;
    out_shs[24] = s1 * d;
    d = -30.399773564 * b;
    out_grads[18][0] = -c0 * d;
    out_grads[24][0] = s0 * d;
    out_grads[18][1] = s0 * d;
    out_grads[24][1] = c0 * d;
    d = 60.799547128 * b;
    out_grads[19][2] = c0 * d;
    out_grads[23][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
    d = 2.503342942;
    out_grads[6][0] = c1 * d;
    out_grads[14][0] = s1 * d;
    out_grads[6][1] = -s1 * d;
    out_grads[14][1] = c1 * d;
    d = -1.77013077;
    out_grads[7][2] = -c1 * d;
    out_grads[13][2] = s1 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[17] = c0 * d;
    out_shs[25] = s0 * d;
    d = 22.200855632 * a;
    out_grads[17][0] = c1 * d;
    out_grads[25][0] = s1 * d;
    out_grads[17][1] = -s1 * d;
    out_grads[25][1] = c1 * d;
    d = -30.399773564 * a;
    out_grads[18][2] = -c1 * d;
    out_grads[24][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[16] = -c1 * d;
    out_shs[26] = s1 * d;
    d = -11.833095811 * z;
    out_grads[16][0] = -c0 * d;
    out_grads[26][0] = s0 * d;
    out_grads[16][1] = s0 * d;
    out_grads[26][1] = c0 * d;
    d = 11.100427816 * z;
    out_grads[17][2] = c0 * d;
    out_grads[25][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[15] = c0 * d;
    out_shs[27] = s0 * d;
    d = 4.099104631;
    out_grads[15][0] = c1 * d;
    out_grads[27][0] = s1 * d;
    out_grads[15][1] = -s1 * d;
    out_grads[27][1] = c1 * d;
    d = -2.366619162;
    out_grads[16][2] = -c1 * d;
    out_grads[26][2] = s1 * d;
    out_grads[0][0] = 0.0;
    out_grads[0][1] = 0.0;
    out_grads[0][2] = 0.0;
    out_grads[1][2] = 0.0;
    out_grads[3][0] = 0.0;
    out_grads[3][1] = 0.0;
    out_grads[5][2] = 0.0;
    out_grads[6][2] = 0.0;
    out_grads[10][0] = 0.0;
    out_grads[10][1] = 0.0;
    out_grads[14][2] = 0.0;
    out_grads[15][2] = 0.0;
    out_grads[21][0] = 0.0;
    out_grads[21][1] = 0.0;
    out_grads[27][2] = 0.0;
}


void eval_sh_grad_8(out float out_shs[45], out vec3 out_grads[45], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    c0 = 1.0;
    s0 = 0.0;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    out_shs[21] = d;
    b = z2 * a - 0.251748252 * b;
    a = b - 0.251282051 * a;
    d = 58.473368113 * a;
    out_shs[36] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    out_grads[2][0] = -c0 * d;
    out_grads[4][0] = s0 * d;
    out_grads[2][1] = s0 * d;
    out_grads[4][1] = c0 * d;
    d = 1.892349392 * z;
    out_grads[3][2] = d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    out_grads[9][0] = -c0 * d;
    out_grads[11][0] = s0 * d;
    out_grads[9][1] = s0 * d;
    out_grads[11][1] = c0 * d;
    d = 14.809976568 * b;
    out_grads[10][2] = d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[20] = -c1 * d;
    out_shs[22] = s1 * d;
    out_grads[20][0] = -c0 * d;
    out_grads[22][0] = s0 * d;
    out_grads[20][1] = s0 * d;
    out_grads[22][1] = c0 * d;
    d = 88.106914343 * b;
    out_grads[21][2] = d;
    a = z2 * b - 0.244755245 * a;
    b = a - 0.246153846 * b;
    d = -77.964490818 * b;
    out_shs[35] = -c1 * d;
    out_shs[37] = s1 * d;
    out_grads[35][0] = -c0 * d;
    out_grads[37][0] = s0 * d;
    out_grads[35][1] = s0 * d;
    out_grads[37][1] = c0 * d;
    d = 467.786944906 * b;
    out_grads[36][2] = d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    d = 1.092548431;
    out_grads[1][0] = c1 * d;
    out_grads[5][0] = s1 * d;
    out_grads[1][1] = -s1 * d;
    out_grads[5][1] = c1 * d;
    d = -1.092548431;
    out_grads[2][2] = -c1 * d;
    out_grads[4][2] = s1 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    d = 6.62322287 * a;
    out_grads[8][0] = c1 * d;
    out_grads[12][0] = s1 * d;
    out_grads[8][1] = -s1 * d;
    out_grads[12][1] = c1 * d;
    d = -14.049977415 * a;
    out_grads[9][2] = -c1 * d;
    out_grads[11][2] = s1 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[19] = c0 * d;
    out_shs[23] = s0 * d;
    d = 30.399773564 * a;
    out_grads[19][0] = c1 * d;
    out_grads[23][0] = s1 * d;
    out_grads[19][1] = -s1 * d;
    out_grads[23][1] = c1 * d;
    d = -96.132524816 * a;
    out_grads[20][2] = -c1 * d;
    out_grads[22][2] = s1 * d;
    b = z2 * a - 0.223776224 * b;
    a = b - 0.230769231 * a;
    d = 65.229772956 * a;
    out_shs[34] = c0 * d;
    out_shs[38] = s0 * d;
    d = 130.459545912 * a;
    out_grads[34][0] = c1 * d;
    out_grads[38][0] = s1 * d;
    out_grads[34][1] = -s1 * d;
    out_grads[38][1] = c1 * d;
    d = -545.751435723 * a;
    out_grads[35][2] = -c1 * d;
    out_grads[37][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    d = -5.310392309 * z;
    out_grads[7][0] = -c0 * d;
    out_grads[13][0] = s0 * d;
    out_grads[7][1] = s0 * d;
    out_grads[13][1] = c0 * d;
    d = 6.62322287 * z;
    out_grads[8][2] = c0 * d;
    out_grads[12][2] = s0 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[18] = -c1 * d;
    out_shs[24] = s1 * d;
    d = -30.399773564 * b;
    out_grads[18][0] = -c0 * d;
    out_grads[24][0] = s0 * d;
    out_grads[18][1] = s0 * d;
    out_grads[24][1] = c0 * d;
    d = 60.799547128 * b;
    out_grads[19][2] = c0 * d;
    out_grads[23][2] = s0 * d;
    a = z2 * b - 0.188811189 * a;
    b = a - 0.205128205 * b;
    d = -48.175380057 * b;
    out_shs[33] = -c1 * d;
    out_shs[39] = s1 * d;
    d = -144.52614017 * b;
    out_grads[33][0] = -c0 * d;
    out_grads[39][0] = s0 * d;
    out_grads[33][1] = s0 * d;
    out_grads[39][1] = c0 * d;
    d = 391.378637737 * b;
    out_grads[34][2] = c0 * d;
    out_grads[38][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
    d = 2.503342942;
    out_grads[6][0] = c1 * d;
    out_grads[14][0] = s1 * d;
    out_grads[6][1] = -s1 * d;
    out_grads[14][1] = c1 * d;
    d = -1.77013077;
    out_grads[7][2] = -c1 * d;
    out_grads[13][2] = s1 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[17] = c0 * d;
    out_shs[25] = s0 * d;
    d = 22.200855632 * a;
    out_grads[17][0] = c1 * d;
    out_grads[25][0] = s1 * d;
    out_grads[17][1] = -s1 * d;
    out_grads[25][1] = c1 * d;
    d = -30.399773564 * a;
    out_grads[18][2] = -c1 * d;
    out_grads[24][2] = s1 * d;
    b = z2 * (a - 0.13986014);
    a = b - 0.169230769 * a;
    d = 31.097074109 * a;
    out_shs[32] = c0 * d;
    out_shs[40] = s0 * d;
    d = 124.388296437 * a;
    out_grads[32][0] = c1 * d;
    out_grads[40][0] = s1 * d;
    out_grads[32][1] = -s1 * d;
    out_grads[40][1] = c1 * d;
    d = -240.876900283 * a;
    out_grads[33][2] = -c1 * d;
    out_grads[39][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[16] = -c1 * d;
    out_shs[26] = s1 * d;
    d = -11.833095811 * z;
    out_grads[16][0] = -c0 * d;
    out_grads[26][0] = s0 * d;
    out_grads[16][1] = s0 * d;
    out_grads[26][1] = c0 * d;
    d = 11.100427816 * z;
    out_grads[17][2] = c0 * d;
    out_grads[25][2] = s0 * d;
    a = (z2 - 7.69230769e-02) * z;
    b = a - 0.123076923 * z;
    d = -17.24955311 * b;
    out_shs[31] = -c1 * d;
    out_shs[41] = s1 * d;
    d = -86.247765552 * b;
    out_grads[31][0] = -c0 * d;
    out_grads[41][0] = s0 * d;
    out_grads[31][1] = s0 * d;
    out_grads[41][1] = c0 * d;
    d = 124.388296437 * b;
    out_grads[32][2] = c0 * d;
    out_grads[40][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[15] = c0 * d;
    out_shs[27] = s0 * d;
    d = 4.099104631;
    out_grads[15][0] = c1 * d;
    out_grads[27][0] = s1 * d;
    out_grads[15][1] = -s1 * d;
    out_grads[27][1] = c1 * d;
    d = -2.366619162;
    out_grads[16][2] = -c1 * d;
    out_grads[26][2] = s1 * d;
    a = z2 - 6.66666667e-02;
    d = 7.984991491 * a;
    out_shs[30] = c0 * d;
    out_shs[42] = s0 * d;
    d = 47.909948945 * a;
    out_grads[30][0] = c1 * d;
    out_grads[42][0] = s1 * d;
    out_grads[30][1] = -s1 * d;
    out_grads[42][1] = c1 * d;
    d = -51.748659331 * a;
    out_grads[31][2] = -c1 * d;
    out_grads[41][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.915706641 * z;
    out_shs[29] = -c1 * d;
    out_shs[43] = s1 * d;
    d = -20.409946485 * z;
    out_grads[29][0] = -c0 * d;
    out_grads[43][0] = s0 * d;
    out_grads[29][1] = s0 * d;
    out_grads[43][1] = c0 * d;
    d = 15.969982982 * z;
    out_grads[30][2] = c0 * d;
    out_grads[42][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.72892666;
    out_shs[28] = c0 * d;
    out_shs[44] = s0 * d;
    d = 5.831413281;
    out_grads[28][0] = c1 * d;
    out_grads[44][0] = s1 * d;
    out_grads[28][1] = -s1 * d;
    out_grads[44][1] = c1 * d;
    d = -2.915706641;
    out_grads[29][2] = -c1 * d;
    out_grads[43][2] = s1 * d;
    out_grads[0][0] = 0.0;
    out_grads[0][1] = 0.0;
    out_grads[0][2] = 0.0;
    out_grads[1][2] = 0.0;
    out_grads[3][0] = 0.0;
    out_grads[3][1] = 0.0;
    out_grads[5][2] = 0.0;
    out_grads[6][2] = 0.0;
    out_grads[10][0] = 0.0;
    out_grads[10][1] = 0.0;
    out_grads[14][2] = 0.0;
    out_grads[15][2] = 0.0;
    out_grads[21][0] = 0.0;
    out_grads[21][1] = 0.0;
    out_grads[27][2] = 0.0;
    out_grads[28][2] = 0.0;
    out_grads[36][0] = 0.0;
    out_grads[36][1] = 0.0;
    out_grads[44][2] = 0.0;
}


void eval_sh_grad_10(out float out_shs[66], out vec3 out_grads[66], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    c0 = 1.0;
    s0 = 0.0;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    out_shs[21] = d;
    b = z2 * a - 0.251748252 * b;
    a = b - 0.251282051 * a;
    d = 58.473368113 * a;
    out_shs[36] = d;
    b = z2 * a - 0.250980392 * b;
    a = b - 0.250773994 * a;
    d = 233.240148813 * a;
    out_shs[55] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    out_grads[2][0] = -c0 * d;
    out_grads[4][0] = s0 * d;
    out_grads[2][1] = s0 * d;
    out_grads[4][1] = c0 * d;
    d = 1.892349392 * z;
    out_grads[3][2] = d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    out_grads[9][0] = -c0 * d;
    out_grads[11][0] = s0 * d;
    out_grads[9][1] = s0 * d;
    out_grads[11][1] = c0 * d;
    d = 14.809976568 * b;
    out_grads[10][2] = d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[20] = -c1 * d;
    out_shs[22] = s1 * d;
    out_grads[20][0] = -c0 * d;
    out_grads[22][0] = s0 * d;
    out_grads[20][1] = s0 * d;
    out_grads[22][1] = c0 * d;
    d = 88.106914343 * b;
    out_grads[21][2] = d;
    a = z2 * b - 0.244755245 * a;
    b = a - 0.246153846 * b;
    d = -77.964490818 * b;
    out_shs[35] = -c1 * d;
    out_shs[37] = s1 * d;
    out_grads[35][0] = -c0 * d;
    out_grads[37][0] = s0 * d;
    out_grads[35][1] = s0 * d;
    out_grads[37][1] = c0 * d;
    d = 467.786944906 * b;
    out_grads[36][2] = d;
    a = z2 * b - 0.247058824 * a;
    b = a - 0.247678019 * b;
    d = -314.500952502 * b;
    out_shs[54] = -c1 * d;
    out_shs[56] = s1 * d;
    out_grads[54][0] = -c0 * d;
    out_grads[56][0] = s0 * d;
    out_grads[54][1] = s0 * d;
    out_grads[56][1] = c0 * d;
    d = 2332.401488133 * b;
    out_grads[55][2] = d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    d = 1.092548431;
    out_grads[1][0] = c1 * d;
    out_grads[5][0] = s1 * d;
    out_grads[1][1] = -s1 * d;
    out_grads[5][1] = c1 * d;
    d = -1.092548431;
    out_grads[2][2] = -c1 * d;
    out_grads[4][2] = s1 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    d = 6.62322287 * a;
    out_grads[8][0] = c1 * d;
    out_grads[12][0] = s1 * d;
    out_grads[8][1] = -s1 * d;
    out_grads[12][1] = c1 * d;
    d = -14.049977415 * a;
    out_grads[9][2] = -c1 * d;
    out_grads[11][2] = s1 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[19] = c0 * d;
    out_shs[23] = s0 * d;
    d = 30.399773564 * a;
    out_grads[19][0] = c1 * d;
    out_grads[23][0] = s1 * d;
    out_grads[19][1] = -s1 * d;
    out_grads[23][1] = c1 * d;
    d = -96.132524816 * a;
    out_grads[20][2] = -c1 * d;
    out_grads[22][2] = s1 * d;
    b = z2 * a - 0.223776224 * b;
    a = b - 0.230769231 * a;
    d = 65.229772956 * a;
    out_shs[34] = c0 * d;
    out_shs[38] = s0 * d;
    d = 130.459545912 * a;
    out_grads[34][0] = c1 * d;
    out_grads[38][0] = s1 * d;
    out_grads[34][1] = -s1 * d;
    out_grads[38][1] = c1 * d;
    d = -545.751435723 * a;
    out_grads[35][2] = -c1 * d;
    out_grads[37][2] = s1 * d;
    b = z2 * a - 0.235294118 * b;
    a = b - 0.238390093 * a;
    d = 272.365814381 * a;
    out_shs[53] = c0 * d;
    out_shs[57] = s0 * d;
    d = 544.731628762 * a;
    out_grads[53][0] = c1 * d;
    out_grads[57][0] = s1 * d;
    out_grads[53][1] = -s1 * d;
    out_grads[57][1] = c1 * d;
    d = -2830.508572514 * a;
    out_grads[54][2] = -c1 * d;
    out_grads[56][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    d = -5.310392309 * z;
    out_grads[7][0] = -c0 * d;
    out_grads[13][0] = s0 * d;
    out_grads[7][1] = s0 * d;
    out_grads[13][1] = c0 * d;
    d = 6.62322287 * z;
    out_grads[8][2] = c0 * d;
    out_grads[12][2] = s0 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[18] = -c1 * d;
    out_shs[24] = s1 * d;
    d = -30.399773564 * b;
    out_grads[18][0] = -c0 * d;
    out_grads[24][0] = s0 * d;
    out_grads[18][1] = s0 * d;
    out_grads[24][1] = c0 * d;
    d = 60.799547128 * b;
    out_grads[19][2] = c0 * d;
    out_grads[23][2] = s0 * d;
    a = z2 * b - 0.188811189 * a;
    b = a - 0.205128205 * b;
    d = -48.175380057 * b;
    out_shs[33] = -c1 * d;
    out_shs[39] = s1 * d;
    d = -144.52614017 * b;
    out_grads[33][0] = -c0 * d;
    out_grads[39][0] = s0 * d;
    out_grads[33][1] = s0 * d;
    out_grads[39][1] = c0 * d;
    d = 391.378637737 * b;
    out_grads[34][2] = c0 * d;
    out_grads[38][2] = s0 * d;
    a = z2 * b - 0.215686275 * a;
    b = a - 0.222910217 * b;
    d = -213.661323441 * b;
    out_shs[52] = -c1 * d;
    out_shs[58] = s1 * d;
    d = -640.983970322 * b;
    out_grads[52][0] = -c0 * d;
    out_grads[58][0] = s0 * d;
    out_grads[52][1] = s0 * d;
    out_grads[58][1] = c0 * d;
    d = 2178.926515046 * b;
    out_grads[53][2] = c0 * d;
    out_grads[57][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
    d = 2.503342942;
    out_grads[6][0] = c1 * d;
    out_grads[14][0] = s1 * d;
    out_grads[6][1] = -s1 * d;
    out_grads[14][1] = c1 * d;
    d = -1.77013077;
    out_grads[7][2] = -c1 * d;
    out_grads[13][2] = s1 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[17] = c0 * d;
    out_shs[25] = s0 * d;
    d = 22.200855632 * a;
    out_grads[17][0] = c1 * d;
    out_grads[25][0] = s1 * d;
    out_grads[17][1] = -s1 * d;
    out_grads[25][1] = c1 * d;
    d = -30.399773564 * a;
    out_grads[18][2] = -c1 * d;
    out_grads[24][2] = s1 * d;
    b = z2 * (a - 0.13986014);
    a = b - 0.169230769 * a;
    d = 31.097074109 * a;
    out_shs[32] = c0 * d;
    out_shs[40] = s0 * d;
    d = 124.388296437 * a;
    out_grads[32][0] = c1 * d;
    out_grads[40][0] = s1 * d;
    out_grads[32][1] = -s1 * d;
    out_grads[40][1] = c1 * d;
    d = -240.876900283 * a;
    out_grads[33][2] = -c1 * d;
    out_grads[39][2] = s1 * d;
    b = z2 * a - 0.188235294 * b;
    a = b - 0.20123839 * a;
    d = 151.081370682 * a;
    out_shs[51] = c0 * d;
    out_shs[59] = s0 * d;
    d = 604.325482728 * a;
    out_grads[51][0] = c1 * d;
    out_grads[59][0] = s1 * d;
    out_grads[51][1] = -s1 * d;
    out_grads[59][1] = c1 * d;
    d = -1495.629264084 * a;
    out_grads[52][2] = -c1 * d;
    out_grads[58][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[16] = -c1 * d;
    out_shs[26] = s1 * d;
    d = -11.833095811 * z;
    out_grads[16][0] = -c0 * d;
    out_grads[26][0] = s0 * d;
    out_grads[16][1] = s0 * d;
    out_grads[26][1] = c0 * d;
    d = 11.100427816 * z;
    out_grads[17][2] = c0 * d;
    out_grads[25][2] = s0 * d;
    a = (z2 - 7.69230769e-02) * z;
    b = a - 0.123076923 * z;
    d = -17.24955311 * b;
    out_shs[31] = -c1 * d;
    out_shs[41] = s1 * d;
    d = -86.247765552 * b;
    out_grads[31][0] = -c0 * d;
    out_grads[41][0] = s0 * d;
    out_grads[31][1] = s0 * d;
    out_grads[41][1] = c0 * d;
    d = 124.388296437 * b;
    out_grads[32][2] = c0 * d;
    out_grads[40][2] = s0 * d;
    a = z2 * b - 0.152941176 * a;
    b = a - 0.173374613 * b;
    d = -95.552248675 * b;
    out_shs[50] = -c1 * d;
    out_shs[60] = s1 * d;
    d = -477.761243376 * b;
    out_grads[50][0] = -c0 * d;
    out_grads[60][0] = s0 * d;
    out_grads[50][1] = s0 * d;
    out_grads[60][1] = c0 * d;
    d = 906.488224092 * b;
    out_grads[51][2] = c0 * d;
    out_grads[59][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[15] = c0 * d;
    out_shs[27] = s0 * d;
    d = 4.099104631;
    out_grads[15][0] = c1 * d;
    out_grads[27][0] = s1 * d;
    out_grads[15][1] = -s1 * d;
    out_grads[27][1] = c1 * d;
    d = -2.366619162;
    out_grads[16][2] = -c1 * d;
    out_grads[26][2] = s1 * d;
    a = z2 - 6.66666667e-02;
    d = 7.984991491 * a;
    out_shs[30] = c0 * d;
    out_shs[42] = s0 * d;
    d = 47.909948945 * a;
    out_grads[30][0] = c1 * d;
    out_grads[42][0] = s1 * d;
    out_grads[30][1] = -s1 * d;
    out_grads[42][1] = c1 * d;
    d = -51.748659331 * a;
    out_grads[31][2] = -c1 * d;
    out_grads[41][2] = s1 * d;
    b = z2 * (a - 0.109803922);
    a = b - 0.139318885 * a;
    d = 53.41533086 * a;
    out_shs[49] = c0 * d;
    out_shs[61] = s0 * d;
    d = 320.491985161 * a;
    out_grads[49][0] = c1 * d;
    out_grads[61][0] = s1 * d;
    out_grads[49][1] = -s1 * d;
    out_grads[61][1] = c1 * d;
    d = -477.761243376 * a;
    out_grads[50][2] = -c1 * d;
    out_grads[60][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.915706641 * z;
    out_shs[29] = -c1 * d;
    out_shs[43] = s1 * d;
    d = -20.409946485 * z;
    out_grads[29][0] = -c0 * d;
    out_grads[43][0] = s0 * d;
    out_grads[29][1] = s0 * d;
    out_grads[43][1] = c0 * d;
    d = 15.969982982 * z;
    out_grads[30][2] = c0 * d;
    out_grads[42][2] = s0 * d;
    a = (z2 - 5.88235294e-02) * z;
    b = a - 9.90712074e-02 * z;
    d = -25.910241313 * b;
    out_shs[48] = -c1 * d;
    out_shs[62] = s1 * d;
    d = -181.371689194 * b;
    out_grads[48][0] = -c0 * d;
    out_grads[62][0] = s0 * d;
    out_grads[48][1] = s0 * d;
    out_grads[62][1] = c0 * d;
    d = 213.661323441 * b;
    out_grads[49][2] = c0 * d;
    out_grads[61][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.72892666;
    out_shs[28] = c0 * d;
    out_shs[44] = s0 * d;
    d = 5.831413281;
    out_grads[28][0] = c1 * d;
    out_grads[44][0] = s1 * d;
    out_grads[28][1] = -s1 * d;
    out_grads[44][1] = c1 * d;
    d = -2.915706641;
    out_grads[29][2] = -c1 * d;
    out_grads[43][2] = s1 * d;
    a = z2 - 5.26315789e-02;
    d = 10.577811722 * a;
    out_shs[47] = c0 * d;
    out_shs[63] = s0 * d;
    d = 84.622493774 * a;
    out_grads[47][0] = c1 * d;
    out_grads[63][0] = s1 * d;
    out_grads[47][1] = -s1 * d;
    out_grads[63][1] = c1 * d;
    d = -77.73072394 * a;
    out_grads[48][2] = -c1 * d;
    out_grads[62][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -3.4318953 * z;
    out_shs[46] = -c1 * d;
    out_shs[64] = s1 * d;
    d = -30.887057699 * z;
    out_grads[46][0] = -c0 * d;
    out_grads[64][0] = s0 * d;
    out_grads[46][1] = s0 * d;
    out_grads[64][1] = c0 * d;
    d = 21.155623443 * z;
    out_grads[47][2] = c0 * d;
    out_grads[63][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.767395118;
    out_shs[45] = c0 * d;
    out_shs[65] = s0 * d;
    d = 7.673951182;
    out_grads[45][0] = c1 * d;
    out_grads[65][0] = s1 * d;
    out_grads[45][1] = -s1 * d;
    out_grads[65][1] = c1 * d;
    d = -3.4318953;
    out_grads[46][2] = -c1 * d;
    out_grads[64][2] = s1 * d;
    out_grads[0][0] = 0.0;
    out_grads[0][1] = 0.0;
    out_grads[0][2] = 0.0;
    out_grads[1][2] = 0.0;
    out_grads[3][0] = 0.0;
    out_grads[3][1] = 0.0;
    out_grads[5][2] = 0.0;
    out_grads[6][2] = 0.0;
    out_grads[10][0] = 0.0;
    out_grads[10][1] = 0.0;
    out_grads[14][2] = 0.0;
    out_grads[15][2] = 0.0;
    out_grads[21][0] = 0.0;
    out_grads[21][1] = 0.0;
    out_grads[27][2] = 0.0;
    out_grads[28][2] = 0.0;
    out_grads[36][0] = 0.0;
    out_grads[36][1] = 0.0;
    out_grads[44][2] = 0.0;
    out_grads[45][2] = 0.0;
    out_grads[55][0] = 0.0;
    out_grads[55][1] = 0.0;
    out_grads[65][2] = 0.0;
}


void eval_sh_grad_12(out float out_shs[91], out vec3 out_grads[91], vec3 point) {
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    c0 = 1.0;
    s0 = 0.0;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    out_shs[21] = d;
    b = z2 * a - 0.251748252 * b;
    a = b - 0.251282051 * a;
    d = 58.473368113 * a;
    out_shs[36] = d;
    b = z2 * a - 0.250980392 * b;
    a = b - 0.250773994 * a;
    d = 233.240148813 * a;
    out_shs[55] = d;
    b = z2 * a - 0.250626566 * b;
    a = b - 0.250517598 * a;
    d = 931.186918633 * a;
    out_shs[78] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    out_grads[2][0] = -c0 * d;
    out_grads[4][0] = s0 * d;
    out_grads[2][1] = s0 * d;
    out_grads[4][1] = c0 * d;
    d = 1.892349392 * z;
    out_grads[3][2] = d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[9] = -c1 * d;
    out_shs[11] = s1 * d;
    out_grads[9][0] = -c0 * d;
    out_grads[11][0] = s0 * d;
    out_grads[9][1] = s0 * d;
    out_grads[11][1] = c0 * d;
    d = 14.809976568 * b;
    out_grads[10][2] = d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[20] = -c1 * d;
    out_shs[22] = s1 * d;
    out_grads[20][0] = -c0 * d;
    out_grads[22][0] = s0 * d;
    out_grads[20][1] = s0 * d;
    out_grads[22][1] = c0 * d;
    d = 88.106914343 * b;
    out_grads[21][2] = d;
    a = z2 * b - 0.244755245 * a;
    b = a - 0.246153846 * b;
    d = -77.964490818 * b;
    out_shs[35] = -c1 * d;
    out_shs[37] = s1 * d;
    out_grads[35][0] = -c0 * d;
    out_grads[37][0] = s0 * d;
    out_grads[35][1] = s0 * d;
    out_grads[37][1] = c0 * d;
    d = 467.786944906 * b;
    out_grads[36][2] = d;
    a = z2 * b - 0.247058824 * a;
    b = a - 0.247678019 * b;
    d = -314.500952502 * b;
    out_shs[54] = -c1 * d;
    out_shs[56] = s1 * d;
    out_grads[54][0] = -c0 * d;
    out_grads[56][0] = s0 * d;
    out_grads[54][1] = s0 * d;
    out_grads[56][1] = c0 * d;
    d = 2332.401488133 * b;
    out_grads[55][2] = d;
    a = z2 * b - 0.248120301 * a;
    b = a - 0.248447205 * b;
    d = -1265.233874957 * b;
    out_shs[77] = -c1 * d;
    out_shs[79] = s1 * d;
    out_grads[77][0] = -c0 * d;
    out_grads[79][0] = s0 * d;
    out_grads[77][1] = s0 * d;
    out_grads[79][1] = c0 * d;
    d = 11174.243023595 * b;
    out_grads[78][2] = d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
    d = 1.092548431;
    out_grads[1][0] = c1 * d;
    out_grads[5][0] = s1 * d;
    out_grads[1][1] = -s1 * d;
    out_grads[5][1] = c1 * d;
    d = -1.092548431;
    out_grads[2][2] = -c1 * d;
    out_grads[4][2] = s1 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[8] = c0 * d;
    out_shs[12] = s0 * d;
    d = 6.62322287 * a;
    out_grads[8][0] = c1 * d;
    out_grads[12][0] = s1 * d;
    out_grads[8][1] = -s1 * d;
    out_grads[12][1] = c1 * d;
    d = -14.049977415 * a;
    out_grads[9][2] = -c1 * d;
    out_grads[11][2] = s1 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[19] = c0 * d;
    out_shs[23] = s0 * d;
    d = 30.399773564 * a;
    out_grads[19][0] = c1 * d;
    out_grads[23][0] = s1 * d;
    out_grads[19][1] = -s1 * d;
    out_grads[23][1] = c1 * d;
    d = -96.132524816 * a;
    out_grads[20][2] = -c1 * d;
    out_grads[22][2] = s1 * d;
    b = z2 * a - 0.223776224 * b;
    a = b - 0.230769231 * a;
    d = 65.229772956 * a;
    out_shs[34] = c0 * d;
    out_shs[38] = s0 * d;
    d = 130.459545912 * a;
    out_grads[34][0] = c1 * d;
    out_grads[38][0] = s1 * d;
    out_grads[34][1] = -s1 * d;
    out_grads[38][1] = c1 * d;
    d = -545.751435723 * a;
    out_grads[35][2] = -c1 * d;
    out_grads[37][2] = s1 * d;
    b = z2 * a - 0.235294118 * b;
    a = b - 0.238390093 * a;
    d = 272.365814381 * a;
    out_shs[53] = c0 * d;
    out_shs[57] = s0 * d;
    d = 544.731628762 * a;
    out_grads[53][0] = c1 * d;
    out_grads[57][0] = s1 * d;
    out_grads[53][1] = -s1 * d;
    out_grads[57][1] = c1 * d;
    d = -2830.508572514 * a;
    out_grads[54][2] = -c1 * d;
    out_grads[56][2] = s1 * d;
    b = z2 * a - 0.240601504 * b;
    a = b - 0.242236025 * a;
    d = 1121.509962433 * a;
    out_shs[76] = c0 * d;
    out_shs[80] = s0 * d;
    d = 2243.019924866 * a;
    out_grads[76][0] = c1 * d;
    out_grads[80][0] = s1 * d;
    out_grads[76][1] = -s1 * d;
    out_grads[80][1] = c1 * d;
    d = -13917.572624524 * a;
    out_grads[77][2] = -c1 * d;
    out_grads[79][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[7] = -c1 * d;
    out_shs[13] = s1 * d;
    d = -5.310392309 * z;
    out_grads[7][0] = -c0 * d;
    out_grads[13][0] = s0 * d;
    out_grads[7][1] = s0 * d;
    out_grads[13][1] = c0 * d;
    d = 6.62322287 * z;
    out_grads[8][2] = c0 * d;
    out_grads[12][2] = s0 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[18] = -c1 * d;
    out_shs[24] = s1 * d;
    d = -30.399773564 * b;
    out_grads[18][0] = -c0 * d;
    out_grads[24][0] = s0 * d;
    out_grads[18][1] = s0 * d;
    out_grads[24][1] = c0 * d;
    d = 60.799547128 * b;
    out_grads[19][2] = c0 * d;
    out_grads[23][2] = s0 * d;
    a = z2 * b - 0.188811189 * a;
    b = a - 0.205128205 * b;
    d = -48.175380057 * b;
    out_shs[33] = -c1 * d;
    out_shs[39] = s1 * d;
    d = -144.52614017 * b;
    out_grads[33][0] = -c0 * d;
    out_grads[39][0] = s0 * d;
    out_grads[33][1] = s0 * d;
    out_grads[39][1] = c0 * d;
    d = 391.378637737 * b;
    out_grads[34][2] = c0 * d;
    out_grads[38][2] = s0 * d;
    a = z2 * b - 0.215686275 * a;
    b = a - 0.222910217 * b;
    d = -213.661323441 * b;
    out_shs[52] = -c1 * d;
    out_shs[58] = s1 * d;
    d = -640.983970322 * b;
    out_grads[52][0] = -c0 * d;
    out_grads[58][0] = s0 * d;
    out_grads[52][1] = s0 * d;
    out_grads[58][1] = c0 * d;
    d = 2178.926515046 * b;
    out_grads[53][2] = c0 * d;
    out_grads[57][2] = s0 * d;
    a = z2 * b - 0.228070175 * a;
    b = a - 0.231884058 * b;
    d = -915.709049803 * b;
    out_shs[75] = -c1 * d;
    out_shs[81] = s1 * d;
    d = -2747.127149409 * b;
    out_grads[75][0] = -c0 * d;
    out_grads[81][0] = s0 * d;
    out_grads[75][1] = s0 * d;
    out_grads[81][1] = c0 * d;
    d = 11215.099624332 * b;
    out_grads[76][2] = c0 * d;
    out_grads[80][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[6] = c0 * d;
    out_shs[14] = s0 * d;
    d = 2.503342942;
    out_grads[6][0] = c1 * d;
    out_grads[14][0] = s1 * d;
    out_grads[6][1] = -s1 * d;
    out_grads[14][1] = c1 * d;
    d = -1.77013077;
    out_grads[7][2] = -c1 * d;
    out_grads[13][2] = s1 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[17] = c0 * d;
    out_shs[25] = s0 * d;
    d = 22.200855632 * a;
    out_grads[17][0] = c1 * d;
    out_grads[25][0] = s1 * d;
    out_grads[17][1] = -s1 * d;
    out_grads[25][1] = c1 * d;
    d = -30.399773564 * a;
    out_grads[18][2] = -c1 * d;
    out_grads[24][2] = s1 * d;
    b = z2 * (a - 0.13986014);
    a = b - 0.169230769 * a;
    d = 31.097074109 * a;
    out_shs[32] = c0 * d;
    out_shs[40] = s0 * d;
    d = 124.388296437 * a;
    out_grads[32][0] = c1 * d;
    out_grads[40][0] = s1 * d;
    out_grads[32][1] = -s1 * d;
    out_grads[40][1] = c1 * d;
    d = -240.876900283 * a;
    out_grads[33][2] = -c1 * d;
    out_grads[39][2] = s1 * d;
    b = z2 * a - 0.188235294 * b;
    a = b - 0.20123839 * a;
    d = 151.081370682 * a;
    out_shs[51] = c0 * d;
    out_shs[59] = s0 * d;
    d = 604.325482728 * a;
    out_grads[51][0] = c1 * d;
    out_grads[59][0] = s1 * d;
    out_grads[51][1] = -s1 * d;
    out_grads[59][1] = c1 * d;
    d = -1495.629264084 * a;
    out_grads[52][2] = -c1 * d;
    out_grads[58][2] = s1 * d;
    b = z2 * a - 0.210526316 * b;
    a = b - 0.217391304 * a;
    d = 686.781787352 * a;
    out_shs[74] = c0 * d;
    out_shs[82] = s0 * d;
    d = 2747.127149409 * a;
    out_grads[74][0] = c1 * d;
    out_grads[82][0] = s1 * d;
    out_grads[74][1] = -s1 * d;
    out_grads[82][1] = c1 * d;
    d = -8241.381448228 * a;
    out_grads[75][2] = -c1 * d;
    out_grads[81][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[16] = -c1 * d;
    out_shs[26] = s1 * d;
    d = -11.833095811 * z;
    out_grads[16][0] = -c0 * d;
    out_grads[26][0] = s0 * d;
    out_grads[16][1] = s0 * d;
    out_grads[26][1] = c0 * d;
    d = 11.100427816 * z;
    out_grads[17][2] = c0 * d;
    out_grads[25][2] = s0 * d;
    a = (z2 - 7.69230769e-02) * z;
    b = a - 0.123076923 * z;
    d = -17.24955311 * b;
    out_shs[31] = -c1 * d;
    out_shs[41] = s1 * d;
    d = -86.247765552 * b;
    out_grads[31][0] = -c0 * d;
    out_grads[41][0] = s0 * d;
    out_grads[31][1] = s0 * d;
    out_grads[41][1] = c0 * d;
    d = 124.388296437 * b;
    out_grads[32][2] = c0 * d;
    out_grads[40][2] = s0 * d;
    a = z2 * b - 0.152941176 * a;
    b = a - 0.173374613 * b;
    d = -95.552248675 * b;
    out_shs[50] = -c1 * d;
    out_shs[60] = s1 * d;
    d = -477.761243376 * b;
    out_grads[50][0] = -c0 * d;
    out_grads[60][0] = s0 * d;
    out_grads[50][1] = s0 * d;
    out_grads[60][1] = c0 * d;
    d = 906.488224092 * b;
    out_grads[51][2] = c0 * d;
    out_grads[59][2] = s0 * d;
    a = z2 * b - 0.187969925 * a;
    b = a - 0.198757764 * b;
    d = -471.12841933 * b;
    out_shs[73] = -c1 * d;
    out_shs[83] = s1 * d;
    d = -2355.642096651 * b;
    out_grads[73][0] = -c0 * d;
    out_grads[83][0] = s0 * d;
    out_grads[73][1] = s0 * d;
    out_grads[83][1] = c0 * d;
    d = 5494.254298819 * b;
    out_grads[74][2] = c0 * d;
    out_grads[82][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[15] = c0 * d;
    out_shs[27] = s0 * d;
    d = 4.099104631;
    out_grads[15][0] = c1 * d;
    out_grads[27][0] = s1 * d;
    out_grads[15][1] = -s1 * d;
    out_grads[27][1] = c1 * d;
    d = -2.366619162;
    out_grads[16][2] = -c1 * d;
    out_grads[26][2] = s1 * d;
    a = z2 - 6.66666667e-02;
    d = 7.984991491 * a;
    out_shs[30] = c0 * d;
    out_shs[42] = s0 * d;
    d = 47.909948945 * a;
    out_grads[30][0] = c1 * d;
    out_grads[42][0] = s1 * d;
    out_grads[30][1] = -s1 * d;
    out_grads[42][1] = c1 * d;
    d = -51.748659331 * a;
    out_grads[31][2] = -c1 * d;
    out_grads[41][2] = s1 * d;
    b = z2 * (a - 0.109803922);
    a = b - 0.139318885 * a;
    d = 53.41533086 * a;
    out_shs[49] = c0 * d;
    out_shs[61] = s0 * d;
    d = 320.491985161 * a;
    out_grads[49][0] = c1 * d;
    out_grads[61][0] = s1 * d;
    out_grads[49][1] = -s1 * d;
    out_grads[61][1] = c1 * d;
    d = -477.761243376 * a;
    out_grads[50][2] = -c1 * d;
    out_grads[60][2] = s1 * d;
    b = z2 * a - 0.160401003 * b;
    a = b - 0.175983437 * a;
    d = 293.800188384 * a;
    out_shs[72] = c0 * d;
    out_shs[84] = s0 * d;
    d = 1762.801130306 * a;
    out_grads[72][0] = c1 * d;
    out_grads[84][0] = s1 * d;
    out_grads[72][1] = -s1 * d;
    out_grads[84][1] = c1 * d;
    d = -3297.898935312 * a;
    out_grads[73][2] = -c1 * d;
    out_grads[83][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.915706641 * z;
    out_shs[29] = -c1 * d;
    out_shs[43] = s1 * d;
    d = -20.409946485 * z;
    out_grads[29][0] = -c0 * d;
    out_grads[43][0] = s0 * d;
    out_grads[29][1] = s0 * d;
    out_grads[43][1] = c0 * d;
    d = 15.969982982 * z;
    out_grads[30][2] = c0 * d;
    out_grads[42][2] = s0 * d;
    a = (z2 - 5.88235294e-02) * z;
    b = a - 9.90712074e-02 * z;
    d = -25.910241313 * b;
    out_shs[48] = -c1 * d;
    out_shs[62] = s1 * d;
    d = -181.371689194 * b;
    out_grads[48][0] = -c0 * d;
    out_grads[62][0] = s0 * d;
    out_grads[48][1] = s0 * d;
    out_grads[62][1] = c0 * d;
    d = 213.661323441 * b;
    out_grads[49][2] = c0 * d;
    out_grads[61][2] = s0 * d;
    a = z2 * b - 0.127819549 * a;
    b = a - 0.149068323 * b;
    d = -165.101452729 * b;
    out_shs[71] = -c1 * d;
    out_shs[85] = s1 * d;
    d = -1155.7101691 * b;
    out_grads[71][0] = -c0 * d;
    out_grads[85][0] = s0 * d;
    out_grads[71][1] = s0 * d;
    out_grads[85][1] = c0 * d;
    d = 1762.801130306 * b;
    out_grads[72][2] = c0 * d;
    out_grads[84][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.72892666;
    out_shs[28] = c0 * d;
    out_shs[44] = s0 * d;
    d = 5.831413281;
    out_grads[28][0] = c1 * d;
    out_grads[44][0] = s1 * d;
    out_grads[28][1] = -s1 * d;
    out_grads[44][1] = c1 * d;
    d = -2.915706641;
    out_grads[29][2] = -c1 * d;
    out_grads[43][2] = s1 * d;
    a = z2 - 5.26315789e-02;
    d = 10.577811722 * a;
    out_shs[47] = c0 * d;
    out_shs[63] = s0 * d;
    d = 84.622493774 * a;
    out_grads[47][0] = c1 * d;
    out_grads[63][0] = s1 * d;
    out_grads[47][1] = -s1 * d;
    out_grads[63][1] = c1 * d;
    d = -77.73072394 * a;
    out_grads[48][2] = -c1 * d;
    out_grads[62][2] = s1 * d;
    b = z2 * (a - 9.02255639e-02);
    a = b - 0.118012422 * a;
    d = 82.550726364 * a;
    out_shs[70] = c0 * d;
    out_shs[86] = s0 * d;
    d = 660.405810914 * a;
    out_grads[70][0] = c1 * d;
    out_grads[86][0] = s1 * d;
    out_grads[70][1] = -s1 * d;
    out_grads[86][1] = c1 * d;
    d = -825.507263643 * a;
    out_grads[71][2] = -c1 * d;
    out_grads[85][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -3.4318953 * z;
    out_shs[46] = -c1 * d;
    out_shs[64] = s1 * d;
    d = -30.887057699 * z;
    out_grads[46][0] = -c0 * d;
    out_grads[64][0] = s0 * d;
    out_grads[46][1] = s0 * d;
    out_grads[64][1] = c0 * d;
    d = 21.155623443 * z;
    out_grads[47][2] = c0 * d;
    out_grads[63][2] = s0 * d;
    a = (z2 - 4.76190476e-02) * z;
    b = a - 8.2815735e-02 * z;
    d = -36.028090689 * b;
    out_shs[69] = -c1 * d;
    out_shs[87] = s1 * d;
    d = -324.252816204 * b;
    out_grads[69][0] = -c0 * d;
    out_grads[87][0] = s0 * d;
    out_grads[69][1] = s0 * d;
    out_grads[87][1] = c0 * d;
    d = 330.202905457 * b;
    out_grads[70][2] = c0 * d;
    out_grads[86][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.767395118;
    out_shs[45] = c0 * d;
    out_shs[65] = s0 * d;
    d = 7.673951182;
    out_grads[45][0] = c1 * d;
    out_grads[65][0] = s1 * d;
    out_grads[45][1] = -s1 * d;
    out_grads[65][1] = c1 * d;
    d = -3.4318953;
    out_grads[46][2] = -c1 * d;
    out_grads[64][2] = s1 * d;
    a = z2 - 4.34782609e-02;
    d = 13.3042542 * a;
    out_shs[68] = c0 * d;
    out_shs[88] = s0 * d;
    d = 133.042542003 * a;
    out_grads[68][0] = c1 * d;
    out_grads[88][0] = s1 * d;
    out_grads[68][1] = -s1 * d;
    out_grads[88][1] = c1 * d;
    d = -108.084272068 * a;
    out_grads[69][2] = -c1 * d;
    out_grads[87][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -3.923210529 * z;
    out_shs[67] = -c1 * d;
    out_shs[89] = s1 * d;
    d = -43.155315818 * z;
    out_grads[67][0] = -c0 * d;
    out_grads[89][0] = s0 * d;
    out_grads[67][1] = s0 * d;
    out_grads[89][1] = c0 * d;
    d = 26.608508401 * z;
    out_grads[68][2] = c0 * d;
    out_grads[88][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.800821996;
    out_shs[66] = c0 * d;
    out_shs[90] = s0 * d;
    d = 9.609863949;
    out_grads[66][0] = c1 * d;
    out_grads[90][0] = s1 * d;
    out_grads[66][1] = -s1 * d;
    out_grads[90][1] = c1 * d;
    d = -3.923210529;
    out_grads[67][2] = -c1 * d;
    out_grads[89][2] = s1 * d;
    out_grads[0][0] = 0.0;
    out_grads[0][1] = 0.0;
    out_grads[0][2] = 0.0;
    out_grads[1][2] = 0.0;
    out_grads[3][0] = 0.0;
    out_grads[3][1] = 0.0;
    out_grads[5][2] = 0.0;
    out_grads[6][2] = 0.0;
    out_grads[10][0] = 0.0;
    out_grads[10][1] = 0.0;
    out_grads[14][2] = 0.0;
    out_grads[15][2] = 0.0;
    out_grads[21][0] = 0.0;
    out_grads[21][1] = 0.0;
    out_grads[27][2] = 0.0;
    out_grads[28][2] = 0.0;
    out_grads[36][0] = 0.0;
    out_grads[36][1] = 0.0;
    out_grads[44][2] = 0.0;
    out_grads[45][2] = 0.0;
    out_grads[55][0] = 0.0;
    out_grads[55][1] = 0.0;
    out_grads[65][2] = 0.0;
    out_grads[66][2] = 0.0;
    out_grads[78][0] = 0.0;
    out_grads[78][1] = 0.0;
    out_grads[90][2] = 0.0;
}
    """
    
    new_code = """
    // Supplemental code for "Ray Tracing Spherical Harmonics Glyphs":
// https://momentsingraphics.de/VMV2023.html
// View results of this shader here: https://www.shadertoy.com/view/dlGSDV
// (c) 2023, Christoph Peters
// This work is licensed under a CC0 1.0 Universal License. To the extent
// possible under law, Christoph Peters has waived all copyright and related or
// neighboring rights to the following code. This work is published from
// Germany. https://creativecommons.org/publicdomain/zero/1.0/

// The index of the highest used band of the spherical harmonics basis. Must be
// even, at least 2 and at most 12.
#define SH_DEGREE 4
// The number of spherical harmonics basis functions
#define SH_COUNT (((SH_DEGREE + 1) * (SH_DEGREE + 2)) / 2)
// Degree of polynomials for which we have to find roots
#define MAX_DEGREE (2 * SH_DEGREE + 2)
// If GL_EXT_control_flow_attributes is available, these defines should be
// defined as [[unroll]] and [[loop]] to give reasonable hints to the compiler.
// That avoids register spilling, which makes execution considerably faster.
#ifndef _unroll_
#define _unroll_
#endif
#ifndef _loop_
#define _loop_
#endif
// When there are fewer intersections/roots than theoretically possible, some
// array entries are set to this value
#define NO_INTERSECTION 3.4e38
// pi and its reciprocal
#define M_PI 3.141592653589793238462643
#define M_INV_PI 0.318309886183790671537767526745


// Searches a single root of a polynomial within a given interval.
// \param out_root The location of the found root.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \param error_tolerance The error tolerance for the returned root location.
//        Typically the error will be much lower but in theory it can be
//        bigger.
// return true if a root was found, false if no root exists.
bool newton_bisection(out float out_root, out float out_end_value,
    float poly[MAX_DEGREE + 1], float begin, float end,
    float begin_value, float error_tolerance)
{
    if (begin == end) {
        out_end_value = begin_value;
        return false;
    }
    // Evaluate the polynomial at the end of the interval
    out_end_value = poly[MAX_DEGREE];
    _unroll_
    for (int i = MAX_DEGREE - 1; i != -1; --i)
        out_end_value = out_end_value * end + poly[i];
    // If the values at both ends have the same non-zero sign, there is no root
    if (begin_value * out_end_value > 0.0)
        return false;
    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);
    _loop_
    for (int i = 0; i != 90; ++i) {
        // Evaluate the polynomial and its derivative
        float value = poly[MAX_DEGREE] * current + poly[MAX_DEGREE - 1];
        float derivative = poly[MAX_DEGREE];
        _unroll_
        for (int j = MAX_DEGREE - 2; j != -1; --j) {
            derivative = derivative * current + value;
            value = value * current + poly[j];
        }
        // Shorten the interval
        bool right = begin_value * value > 0.0;
        begin = right ? current : begin;
        end = right ? end : current;
        // Apply Newton's method
        float guess = current - value / derivative;
        // Pick a guess
        float middle = 0.5 * (begin + end);
        float next = (guess >= begin && guess <= end) ? guess : middle;
        // Move along or terminate
        bool done = abs(next - current) < error_tolerance;
        current = next;
        if (done)
            break;
    }
    out_root = current;
    return true;
}


// Finds all roots of the given polynomial in the interval [begin, end] and
// writes them to out_roots. Some entries will be NO_INTERSECTION but other
// than that the array is sorted. The last entry is always NO_INTERSECTION.
void find_roots(out float out_roots[MAX_DEGREE + 1], float poly[MAX_DEGREE + 1], float begin, float end) {
    float tolerance = (end - begin) * 1.0e-4;
    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float derivative[MAX_DEGREE + 1];
    derivative[0] = poly[MAX_DEGREE - 2];
    derivative[1] = float(MAX_DEGREE - 1) * poly[MAX_DEGREE - 1];
    derivative[2] = (0.5 * float((MAX_DEGREE - 1) * MAX_DEGREE)) * poly[MAX_DEGREE - 0];
    _unroll_
    for (int i = 3; i != MAX_DEGREE + 1; ++i)
        derivative[i] = 0.0;
    // Compute its two roots using the quadratic formula
    float discriminant = derivative[1] * derivative[1] - 4.0 * derivative[0] * derivative[2];
    if (discriminant >= 0.0) {
        float sqrt_discriminant = sqrt(discriminant);
        float scaled_root = derivative[1] + ((derivative[1] > 0.0) ? sqrt_discriminant : (-sqrt_discriminant));
        float root_0 = clamp(-2.0 * derivative[0] / scaled_root, begin, end);
        float root_1 = clamp(-0.5 * scaled_root / derivative[2], begin, end);
        out_roots[MAX_DEGREE - 2] = min(root_0, root_1);
        out_roots[MAX_DEGREE - 1] = max(root_0, root_1);
    }
    else {
        // Indicate that the cubic derivative has a single root
        out_roots[MAX_DEGREE - 2] = begin;
        out_roots[MAX_DEGREE - 1] = begin;
    }
    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched roots are set to begin
    out_roots[MAX_DEGREE] = end;
    _unroll_
    for (int i = 0; i != MAX_DEGREE - 2; ++i)
        out_roots[i] = begin;
    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree MAX_DEGREE and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.
    _loop_
    for (int degree = 3; degree != MAX_DEGREE + 1; ++degree) {
        // Take the integral of the previous derivative (scaled such that the
        // constant coefficient can still be copied directly from poly)
        float prev_derivative_order = float(MAX_DEGREE + 1 - degree);
        _unroll_
        for (int i = MAX_DEGREE; i != 0; --i)
            derivative[i] = derivative[i - 1] * (prev_derivative_order * (1.0 / float(i)));
        // Copy the constant coefficient without causing spilling. This part
        // would be harder if the derivative were not scaled the way it is.
        _unroll_
        for (int i = 0; i != MAX_DEGREE - 2; ++i)
            derivative[0] = (degree == MAX_DEGREE - i) ? poly[i] : derivative[0];
        // Determine the value of this derivative at begin
        float begin_value = derivative[MAX_DEGREE];
        _unroll_
        for (int i = MAX_DEGREE - 1; i != -1; --i)
            begin_value = begin_value * begin + derivative[i];
        // Iterate over the intervals where roots may be found
        _unroll_
        for (int i = 0; i != MAX_DEGREE; ++i) {
            if (i < MAX_DEGREE - degree)
                continue;
            float current_begin = out_roots[i];
            float current_end = out_roots[i + 1];
            // Try to find a root
            float root;
            if (newton_bisection(root, begin_value, derivative, current_begin, current_end, begin_value, tolerance))
                out_roots[i] = root;
            else if (degree < MAX_DEGREE)
                // Create an empty interval for the next iteration
                out_roots[i] = out_roots[i - 1];
            else
                out_roots[i] = NO_INTERSECTION;
        }
    }
    // We no longer need this array entry
    out_roots[MAX_DEGREE] = NO_INTERSECTION;
}


// Evaluates the spherical harmonics basis in bands 0, 2, ..., SH_DEGREE.
// Conventions are as in the following paper.
// M. Descoteaux, E. Angelino, S. Fitzgibbons, and R. Deriche. Regularized,
// fast, and robust analytical q-ball imaging. Magnetic Resonance in Medicine,
// 58(3), 2007. https://doi.org/10.1002/mrm.21277
// \param out_shs Values of SH basis functions in bands 0, 2, ..., SH_DEGREE in
//        this order.
// \param point The point on the unit sphere where the basis should be
//        evaluated.
void eval_sh(out float out_shs[SH_COUNT], vec3 point) {
#if SH_DEGREE == 2
    eval_sh_2(out_shs, point);
#elif SH_DEGREE == 4
    eval_sh_4(out_shs, point);
#elif SH_DEGREE == 6
    eval_sh_6(out_shs, point);
#elif SH_DEGREE == 8
    eval_sh_8(out_shs, point);
#elif SH_DEGREE == 10
    eval_sh_10(out_shs, point);
#elif SH_DEGREE == 12
    eval_sh_12(out_shs, point);
#endif
}


// Evaluates the gradient of each basis function given by eval_sh() and the
// basis itself
void eval_sh_grad(out float out_shs[SH_COUNT], out vec3 out_grads[SH_COUNT], vec3 point) {
#if SH_DEGREE == 2
    eval_sh_grad_2(out_shs, out_grads, point);
#elif SH_DEGREE == 4
    eval_sh_grad_4(out_shs, out_grads, point);
#elif SH_DEGREE == 6
    eval_sh_grad_6(out_shs, out_grads, point);
#elif SH_DEGREE == 8
    eval_sh_grad_8(out_shs, out_grads, point);
#elif SH_DEGREE == 10
    eval_sh_grad_10(out_shs, out_grads, point);
#elif SH_DEGREE == 12
    eval_sh_grad_12(out_shs, out_grads, point);
#endif
}


// Outputs a matrix that turns equidistant samples on the unit circle of a
// homogeneous polynomial into coefficients of that polynomial.
void get_inv_vandermonde(out float v[(SH_DEGREE + 1) * (SH_DEGREE + 1)]) {
#if SH_DEGREE == 2
    v[0*3 + 0] = -0.3333333333;     v[0*3 + 1] = 0.6666666667;      v[0*3 + 2] = 0.6666666667;
    v[1*3 + 0] = -0.0;              v[1*3 + 1] = 1.1547005384;      v[1*3 + 2] = -1.1547005384;
    v[2*3 + 0] = 1.0;               v[2*3 + 1] = 0.0;               v[2*3 + 2] = 0.0;
#elif SH_DEGREE == 4
    v[0*5 + 0] = 0.2;               v[0*5 + 1] = -0.2472135955;     v[0*5 + 2] = 0.6472135955;      v[0*5 + 3] = 0.6472135955;      v[0*5 + 4] = -0.2472135955;
    v[1*5 + 0] = 0.0;               v[1*5 + 1] = -0.1796111906;     v[1*5 + 2] = 1.9919186279;      v[1*5 + 3] = -1.9919186279;     v[1*5 + 4] = 0.1796111906;
    v[2*5 + 0] = -2.0;              v[2*5 + 1] = 2.3416407865;      v[2*5 + 2] = -0.3416407865;     v[2*5 + 3] = -0.3416407865;     v[2*5 + 4] = 2.3416407865;
    v[3*5 + 0] = -0.0;              v[3*5 + 1] = 1.7013016167;      v[3*5 + 2] = -1.0514622242;     v[3*5 + 3] = 1.0514622242;      v[3*5 + 4] = -1.7013016167;
    v[4*5 + 0] = 1.0;               v[4*5 + 1] = 0.0;               v[4*5 + 2] = 0.0;               v[4*5 + 3] = 0.0;               v[4*5 + 4] = 0.0;
#elif SH_DEGREE == 6
    v[0*7 + 0] = -0.1428571429;     v[0*7 + 1] = 0.1585594663;      v[0*7 + 2] = -0.2291250674;     v[0*7 + 3] = 0.6419941725;      v[0*7 + 4] = 0.6419941725;      v[0*7 + 5] = -0.2291250674;     v[0*7 + 6] = 0.1585594663;
    v[1*7 + 0] = -0.0;              v[1*7 + 1] = 0.0763582145;      v[1*7 + 2] = -0.2873137468;     v[1*7 + 3] = 2.8127602518;      v[1*7 + 4] = -2.8127602518;     v[1*7 + 5] = 0.2873137468;      v[1*7 + 6] = -0.0763582145;
    v[2*7 + 0] = 3.0;               v[2*7 + 1] = -3.2929766145;     v[2*7 + 2] = 4.4513463718;      v[2*7 + 3] = -1.1583697574;     v[2*7 + 4] = -1.1583697574;     v[2*7 + 5] = 4.4513463718;      v[2*7 + 6] = -3.2929766145;
    v[3*7 + 0] = 0.0;               v[3*7 + 1] = -1.5858139579;     v[3*7 + 2] = 5.5818117995;      v[3*7 + 3] = -5.0751495106;     v[3*7 + 4] = 5.0751495106;      v[3*7 + 5] = -5.5818117995;     v[3*7 + 6] = 1.5858139579;
    v[4*7 + 0] = -5.0;              v[4*7 + 1] = 4.7858935686;      v[4*7 + 2] = -1.0200067492;     v[4*7 + 3] = 0.2341131806;      v[4*7 + 4] = 0.2341131806;      v[4*7 + 5] = -1.0200067492;     v[4*7 + 6] = 4.7858935686;
    v[5*7 + 0] = -0.0;              v[5*7 + 1] = 2.3047648710;      v[5*7 + 2] = -1.2790480077;     v[5*7 + 3] = 1.0257168633;      v[5*7 + 4] = -1.0257168633;     v[5*7 + 5] = 1.2790480077;      v[5*7 + 6] = -2.3047648710;
    v[6*7 + 0] = 1.0;               v[6*7 + 1] = 0.0;               v[6*7 + 2] = 0.0;               v[6*7 + 3] = 0.0;               v[6*7 + 4] = 0.0;               v[6*7 + 5] = 0.0;               v[6*7 + 6] = 0.0;
#elif SH_DEGREE == 8
    v[0*9 + 0] = 0.1111111111;      v[0*9 + 1] = -0.1182419747;     v[0*9 + 2] = 0.1450452544;      v[0*9 + 3] = -0.2222222222;     v[0*9 + 4] = 0.6398633870;      v[0*9 + 5] = 0.6398633870;      v[0*9 + 6] = -0.2222222222;     v[0*9 + 7] = 0.1450452544;      v[0*9 + 8] = -0.1182419747;
    v[1*9 + 0] = 0.0;               v[1*9 + 1] = -0.0430365592;     v[1*9 + 2] = 0.1217074194;      v[1*9 + 3] = -0.3849001795;     v[1*9 + 4] = 3.6288455938;      v[1*9 + 5] = -3.6288455938;     v[1*9 + 6] = 0.3849001795;      v[1*9 + 7] = -0.1217074194;     v[1*9 + 8] = 0.0430365592;
    v[2*9 + 0] = -4.0;              v[2*9 + 1] = 4.2410470634;      v[2*9 + 2] = -5.1195045066;     v[2*9 + 3] = 7.3333333333;      v[2*9 + 4] = -2.4548758901;     v[2*9 + 5] = -2.4548758901;     v[2*9 + 6] = 7.3333333333;      v[2*9 + 7] = -5.1195045066;     v[2*9 + 8] = 4.2410470634;
    v[3*9 + 0] = -0.0;              v[3*9 + 1] = 1.5436148932;      v[3*9 + 2] = -4.2957743433;     v[3*9 + 3] = 12.7017059222;     v[3*9 + 4] = -13.9222930051;    v[3*9 + 5] = 13.9222930051;     v[3*9 + 6] = -12.7017059222;    v[3*9 + 7] = 4.2957743433;      v[3*9 + 8] = -1.5436148932;
    v[4*9 + 0] = 14.0;              v[4*9 + 1] = -14.3366589404;    v[4*9 + 2] = 14.6711193836;     v[4*9 + 3] = -6.0;              v[4*9 + 4] = 1.6655395568;      v[4*9 + 5] = 1.6655395568;      v[4*9 + 6] = -6.0;              v[4*9 + 7] = 14.6711193836;     v[4*9 + 8] = -14.3366589404;
    v[5*9 + 0] = 0.0;               v[5*9 + 1] = -5.2181171131;     v[5*9 + 2] = 12.3105308637;     v[5*9 + 3] = -10.3923048454;    v[5*9 + 4] = 9.4457442082;      v[5*9 + 5] = -9.4457442082;     v[5*9 + 6] = 10.3923048454;     v[5*9 + 7] = -12.3105308637;    v[5*9 + 8] = 5.2181171131;
    v[6*9 + 0] = -9.3333333333;     v[6*9 + 1] = 8.0330865684;      v[6*9 + 2] = -1.8540394597;     v[6*9 + 3] = 0.6666666667;      v[6*9 + 4] = -0.1790471086;     v[6*9 + 5] = -0.1790471086;     v[6*9 + 6] = 0.6666666667;      v[6*9 + 7] = -1.8540394597;     v[6*9 + 8] = 8.0330865684;
    v[7*9 + 0] = -0.0;              v[7*9 + 1] = 2.9238044002;      v[7*9 + 2] = -1.5557238269;     v[7*9 + 3] = 1.1547005384;      v[7*9 + 4] = -1.0154266119;     v[7*9 + 5] = 1.0154266119;      v[7*9 + 6] = -1.1547005384;     v[7*9 + 7] = 1.5557238269;      v[7*9 + 8] = -2.9238044002;
    v[8*9 + 0] = 1.0;               v[8*9 + 1] = 0.0;               v[8*9 + 2] = 0.0;               v[8*9 + 3] = 0.0;               v[8*9 + 4] = 0.0;               v[8*9 + 5] = 0.0;               v[8*9 + 6] = 0.0;               v[8*9 + 7] = 0.0;               v[8*9 + 8] = 0.0;
#elif SH_DEGREE == 10
    v[0*11 + 0] = -0.0909090909;    v[0*11 + 1] = 0.0947470106;     v[0*11 + 2] = -0.1080638444;    v[0*11 + 3] = 0.1388220215;     v[0*11 + 4] = -0.2188392043;    v[0*11 + 5] = 0.6387885621;     v[0*11 + 6] = 0.6387885621;     v[0*11 + 7] = -0.2188392043;    v[0*11 + 8] = 0.1388220215;     v[0*11 + 9] = -0.1080638444;    v[0*11 + 10] = 0.0947470106;
    v[1*11 + 0] = -0.0;             v[1*11 + 1] = 0.0278202324;     v[1*11 + 2] = -0.0694484159;    v[1*11 + 3] = 0.1602091533;     v[1*11 + 4] = -0.4791910159;    v[1*11 + 5] = 4.4428720384;     v[1*11 + 6] = -4.4428720384;    v[1*11 + 7] = 0.4791910159;     v[1*11 + 8] = -0.1602091533;    v[1*11 + 9] = 0.0694484159;     v[1*11 + 10] = -0.0278202324;
    v[2*11 + 0] = 5.0;              v[2*11 + 1] = -5.2029168239;    v[2*11 + 2] = 5.8988796576;     v[2*11 + 3] = -7.4503199653;    v[2*11 + 4] = 10.9868742757;    v[2*11 + 5] = -4.2325171441;    v[2*11 + 6] = -4.2325171441;    v[2*11 + 7] = 10.9868742757;    v[2*11 + 8] = -7.4503199653;    v[2*11 + 9] = 5.8988796576;     v[2*11 + 10] = -5.2029168239;
    v[3*11 + 0] = 0.0;              v[3*11 + 1] = -1.5277142200;    v[3*11 + 2] = 3.7909797649;     v[3*11 + 3] = -8.5981275876;    v[3*11 + 4] = 24.0578988657;    v[3*11 + 5] = -29.4378033460;   v[3*11 + 6] = 29.4378033460;    v[3*11 + 7] = -24.0578988657;   v[3*11 + 8] = 8.5981275876;     v[3*11 + 9] = -3.7909797649;    v[3*11 + 10] = 1.5277142200;
    v[4*11 + 0] = -30.0;            v[4*11 + 1] = 30.8179361182;    v[4*11 + 2] = -33.2247539061;   v[4*11 + 3] = 35.8884989085;    v[4*11 + 4] = -19.5374870834;   v[4*11 + 5] = 6.0558059629;     v[4*11 + 6] = 6.0558059629;     v[4*11 + 7] = -19.5374870834;   v[4*11 + 8] = 35.8884989085;    v[4*11 + 9] = -33.2247539061;   v[4*11 + 10] = 30.8179361182;
    v[5*11 + 0] = -0.0;             v[5*11 + 1] = 9.0489625020;     v[5*11 + 2] = -21.3522528115;   v[5*11 + 3] = 41.4175356200;    v[5*11 + 4] = -42.7811292411;   v[5*11 + 5] = 42.1190556280;    v[5*11 + 6] = -42.1190556280;   v[5*11 + 7] = 42.7811292411;    v[5*11 + 8] = -41.4175356200;   v[5*11 + 9] = 21.3522528115;    v[5*11 + 10] = -9.0489625020;
    v[6*11 + 0] = 42.0;             v[6*11 + 1] = -41.1161037573;   v[6*11 + 2] = 36.2032364762;    v[6*11 + 3] = -16.3373898141;   v[6*11 + 4] = 7.4261062994;     v[6*11 + 5] = -2.1758492042;    v[6*11 + 6] = -2.1758492042;    v[6*11 + 7] = 7.4261062994;     v[6*11 + 8] = -16.3373898141;   v[6*11 + 9] = 36.2032364762;    v[6*11 + 10] = -41.1161037573;
    v[7*11 + 0] = 0.0;              v[7*11 + 1] = -12.0727773496;   v[7*11 + 2] = 23.2664073304;    v[7*11 + 3] = -18.8543529304;   v[7*11 + 4] = 16.2609045881;    v[7*11 + 5] = -15.1333636234;   v[7*11 + 6] = 15.1333636234;    v[7*11 + 7] = -16.2609045881;   v[7*11 + 8] = 18.8543529304;    v[7*11 + 9] = -23.2664073304;   v[7*11 + 10] = 12.0727773496;
    v[8*11 + 0] = -15.0;            v[8*11 + 1] = 12.0883694702;    v[8*11 + 2] = -2.8781222629;    v[8*11 + 3] = 1.1465503415;     v[8*11 + 4] = -0.5020543475;    v[8*11 + 5] = 0.1452567988;     v[8*11 + 6] = 0.1452567988;     v[8*11 + 7] = -0.5020543475;    v[8*11 + 8] = 1.1465503415;     v[8*11 + 9] = -2.8781222629;    v[8*11 + 10] = 12.0883694702;
    v[9*11 + 0] = -0.0;             v[9*11 + 1] = 3.5494655329;     v[9*11 + 2] = -1.8496568659;    v[9*11 + 3] = 1.3231896304;     v[9*11 + 4] = -1.0993456751;    v[9*11 + 5] = 1.0102832265;     v[9*11 + 6] = -1.0102832265;    v[9*11 + 7] = 1.0993456751;     v[9*11 + 8] = -1.3231896304;    v[9*11 + 9] = 1.8496568659;     v[9*11 + 10] = -3.5494655329;
    v[10*11 + 0] = 1.0;             v[10*11 + 1] = 0.0;             v[10*11 + 2] = 0.0;             v[10*11 + 3] = 0.0;             v[10*11 + 4] = 0.0;             v[10*11 + 5] = 0.0;             v[10*11 + 6] = 0.0;             v[10*11 + 7] = 0.0;             v[10*11 + 8] = 0.0;             v[10*11 + 9] = 0.0;             v[10*11 + 10] = 0.0;
#elif SH_DEGREE == 12
    v[0*13 + 0] = 0.0769230769;     v[0*13 + 1] = -0.0792252178;    v[0*13 + 2] = 0.0868739663;     v[0*13 + 3] = -0.1027681661;    v[0*13 + 4] = 0.1354125166;     v[0*13 + 5] = -0.2169261613;    v[0*13 + 6] = 0.6381715239;     v[0*13 + 7] = 0.6381715239;     v[0*13 + 8] = -0.2169261613;    v[0*13 + 9] = 0.1354125166;     v[0*13 + 10] = -0.1027681661;   v[0*13 + 11] = 0.0868739663;    v[0*13 + 12] = -0.0792252178;
    v[1*13 + 0] = -0.0;             v[1*13 + 1] = -0.0195272624;    v[1*13 + 2] = 0.0455949748;     v[1*13 + 3] = -0.0910446506;    v[1*13 + 4] = 0.1961788986;     v[1*13 + 5] = -0.5719872785;    v[1*13 + 6] = 5.2558153553;     v[1*13 + 7] = -5.2558153553;    v[1*13 + 8] = 0.5719872785;     v[1*13 + 9] = -0.1961788986;    v[1*13 + 10] = 0.0910446506;    v[1*13 + 11] = -0.0455949748;   v[1*13 + 12] = 0.0195272624;
    v[2*13 + 0] = -6.0;             v[2*13 + 1] = 6.1747539478;     v[2*13 + 2] = -6.7522392818;    v[2*13 + 3] = 7.9352584366;     v[2*13 + 4] = -10.2779620900;   v[2*13 + 5] = 15.4120340799;    v[2*13 + 6] = -6.4918450925;    v[2*13 + 7] = -6.4918450925;    v[2*13 + 8] = 15.4120340799;    v[2*13 + 9] = -10.2779620900;   v[2*13 + 10] = 7.9352584366;    v[2*13 + 11] = -6.7522392818;   v[2*13 + 12] = 6.1747539478;
    v[3*13 + 0] = -0.0;             v[3*13 + 1] = 1.5219401578;     v[3*13 + 2] = -3.5438485554;    v[3*13 + 3] = 7.0300255289;     v[3*13 + 4] = -14.8901987371;   v[3*13 + 5] = 40.6381940129;    v[3*13 + 6] = -53.4651544987;   v[3*13 + 7] = 53.4651544987;    v[3*13 + 8] = -40.6381940129;   v[3*13 + 9] = 14.8901987371;    v[3*13 + 10] = -7.0300255289;   v[3*13 + 11] = 3.5438485554;    v[3*13 + 12] = -1.5219401578;
    v[4*13 + 0] = 55.0;             v[4*13 + 1] = -56.2709061445;   v[4*13 + 2] = 60.2549306937;    v[4*13 + 3] = -67.2511796347;   v[4*13 + 4] = 75.2477722397;    v[4*13 + 5] = -47.9480941911;   v[4*13 + 6] = 15.9674770369;    v[4*13 + 7] = 15.9674770369;    v[4*13 + 8] = -47.9480941911;   v[4*13 + 9] = 75.2477722397;    v[4*13 + 10] = -67.2511796347;  v[4*13 + 11] = 60.2549306937;   v[4*13 + 12] = -56.2709061445;
    v[5*13 + 0] = 0.0;              v[5*13 + 1] = -13.8695326974;   v[5*13 + 2] = 31.6242271914;    v[5*13 + 3] = -59.5793462127;   v[5*13 + 4] = 109.0152185187;   v[5*13 + 5] = -126.4287338180;  v[5*13 + 6] = 131.5040045727;   v[5*13 + 7] = -131.5040045727;  v[5*13 + 8] = 126.4287338180;   v[5*13 + 9] = -109.0152185187;  v[5*13 + 10] = 59.5793462127;   v[5*13 + 11] = -31.6242271914;  v[5*13 + 12] = 13.8695326974;
    v[6*13 + 0] = -132.0;           v[6*13 + 1] = 132.5319409049;   v[6*13 + 2] = -132.4780513404;  v[6*13 + 3] = 123.5674782081;   v[6*13 + 4] = -74.4320682907;   v[6*13 + 5] = 38.8801193717;    v[6*13 + 6] = -12.0694188537;   v[6*13 + 7] = -12.0694188537;   v[6*13 + 8] = 38.8801193717;    v[6*13 + 9] = -74.4320682907;   v[6*13 + 10] = 123.5674782081;  v[6*13 + 11] = -132.4780513404; v[6*13 + 12] = 132.5319409049;
    v[7*13 + 0] = -0.0;             v[7*13 + 1] = 32.6661895777;    v[7*13 + 2] = -69.5298450306;   v[7*13 + 3] = 109.4712331409;   v[7*13 + 4] = -107.8334673306;  v[7*13 + 5] = 102.5184492897;   v[7*13 + 6] = -99.4006071501;   v[7*13 + 7] = 99.4006071501;    v[7*13 + 8] = -102.5184492897;  v[7*13 + 9] = 107.8334673306;   v[7*13 + 10] = -109.4712331409; v[7*13 + 11] = 69.5298450306;   v[7*13 + 12] = -32.6661895777;
    v[8*13 + 0] = 99.0;             v[8*13 + 1] = -93.9113626635;   v[8*13 + 2] = 75.3147168618;    v[8*13 + 3] = -35.2795800772;   v[8*13 + 4] = 18.0521608541;    v[8*13 + 5] = -8.8650350126;    v[8*13 + 6] = 2.6891000373;     v[8*13 + 7] = 2.6891000373;     v[8*13 + 8] = -8.8650350126;    v[8*13 + 9] = 18.0521608541;    v[8*13 + 10] = -35.2795800772;  v[8*13 + 11] = 75.3147168618;   v[8*13 + 12] = -93.9113626635;
    v[9*13 + 0] = 0.0;              v[9*13 + 1] = -23.1470719837;   v[9*13 + 2] = 39.5282127035;    v[9*13 + 3] = -31.2549806126;   v[9*13 + 4] = 26.1530700733;    v[9*13 + 5] = -23.3751762359;   v[9*13 + 6] = 22.1467313083;    v[9*13 + 7] = -22.1467313083;   v[9*13 + 8] = 23.3751762359;    v[9*13 + 9] = -26.1530700733;   v[9*13 + 10] = 31.2549806126;   v[9*13 + 11] = -39.5282127035;  v[9*13 + 12] = 23.1470719837;
    v[10*13 + 0] = -22.0;           v[10*13 + 1] = 16.9531714429;   v[10*13 + 2] = -4.0999479387;   v[10*13 + 3] = 1.7021989010;    v[10*13 + 4] = -0.8387165175;   v[10*13 + 5] = 0.4056079008;    v[10*13 + 6] = -0.1223137885;   v[10*13 + 7] = -0.1223137885;   v[10*13 + 8] = 0.4056079008;    v[10*13 + 9] = -0.8387165175;   v[10*13 + 10] = 1.7021989010;   v[10*13 + 11] = -4.0999479387;  v[10*13 + 12] = 16.9531714429;
    v[11*13 + 0] = -0.0;            v[11*13 + 1] = 4.1785814689;    v[11*13 + 2] = -2.1518186743;   v[11*13 + 3] = 1.5080166355;    v[11*13 + 4] = -1.2150906493;   v[11*13 + 5] = 1.0695001374;    v[11*13 + 6] = -1.0073446769;   v[11*13 + 7] = 1.0073446769;    v[11*13 + 8] = -1.0695001374;   v[11*13 + 9] = 1.2150906493;    v[11*13 + 10] = -1.5080166355;  v[11*13 + 11] = 2.1518186743;   v[11*13 + 12] = -4.1785814689;
    v[12*13 + 0] = 1.0;             v[12*13 + 1] = 0.0;             v[12*13 + 2] = 0.0;             v[12*13 + 3] = 0.0;             v[12*13 + 4] = 0.0;             v[12*13 + 5] = 0.0;             v[12*13 + 6] = 0.0;             v[12*13 + 7] = 0.0;             v[12*13 + 8] = 0.0;             v[12*13 + 9] = 0.0;             v[12*13 + 10] = 0.0;            v[12*13 + 11] = 0.0;            v[12*13 + 12] = 0.0;
#endif
}


// Determines all intersections between a ray and a spherical harmonics glyph.
// \param out_ray_params The ray parameters at intersection points. The points
//        themselves are at ray_origin + out_ray_params[i] * ray_dir. Some
//        entries may be NO_INTERSECTION but other than that the array is
//        sorted.
// \param sh_coeffs SH_COUNT spherical harmonic coefficients defining the
//        glyph. Their exact meaning is defined by eval_sh().
// \param ray_origin The origin of the ray, relative to the glyph center.
// \param ray_dir The normalized direction vector of the ray.
void ray_sh_glyph_intersections(out float out_ray_params[MAX_DEGREE], float sh_coeffs[SH_COUNT], vec3 ray_origin, vec3 ray_dir) {
    // Determine the direction from the glyph center to the closest point on
    // the ray
    float dir_dot_origin = dot(ray_dir, ray_origin);
    vec3 closest_dir = normalize(ray_origin - dir_dot_origin * ray_dir);
    // Evaluate the SH polynomial at SH_DEGREE + 1 points. That is enough to
    // know its value everywhere along the ray.
    float sh_values[SH_DEGREE + 1];
    _unroll_
    for (int i = 0; i != SH_DEGREE + 1; ++i) {
        vec3 point = cos(float(i) * (M_PI / float(SH_DEGREE + 1))) * ray_dir
                   + sin(float(i) * (M_PI / float(SH_DEGREE + 1))) * closest_dir;
        float shs[SH_COUNT];
        eval_sh(shs, point);
        sh_values[i] = 0.0;
        _unroll_
        for (int j = 0; j != SH_COUNT; ++j)
            sh_values[i] += sh_coeffs[j] * shs[j];
    }
    // Compute coefficients of the SH polynomial along the ray in the
    // coordinate frame given by ray_dir and closest_dir
    float radius_poly[SH_DEGREE + 1];
    float inv_vander[(SH_DEGREE + 1) * (SH_DEGREE + 1)];
    get_inv_vandermonde(inv_vander);
    _unroll_
    for (int i = 0; i != SH_DEGREE + 1; ++i) {
        radius_poly[i] = 0.0;
        _unroll_
        for (int j = 0; j != SH_DEGREE + 1; ++j)
            radius_poly[i] += inv_vander[i * (SH_DEGREE + 1) + j] * sh_values[j];
    }
    // Compute a bounding circle around the glyph in the relevant plane
    float radius_max = 0.0;
    _unroll_
    for (int i = 0; i != SH_DEGREE + 1; ++i) {
        float bound = sqrt(pow(float(i), float(i)) * pow(float(SH_DEGREE - i), float(SH_DEGREE - i)) / pow(float(SH_DEGREE), float(SH_DEGREE)));
        // Workaround for buggy compilers where 0^0 is 0
        bound = (i == 0 || i == SH_DEGREE) ? 1.0 : bound;
        radius_max += bound * abs(radius_poly[i]);
    }
    // Figure out the interval, where (if at all) the ray intersects the circle
    float closest_dot_origin = dot(closest_dir, ray_origin);
    if (radius_max < abs(closest_dot_origin)) {
        _unroll_
        for (int i = 0; i != MAX_DEGREE; ++i)
            out_ray_params[i] = NO_INTERSECTION;
        return;
    }
    float radius_over_dot = radius_max / closest_dot_origin;
    float u_max = sqrt(radius_over_dot * radius_over_dot - 1.0);
    // Take the square of radius_poly
    float poly[MAX_DEGREE + 1];
    _unroll_
    for (int i = 0; i != MAX_DEGREE + 1; ++i)
        poly[i] = 0.0;
    _unroll_
    for (int i = 0; i != SH_DEGREE + 1; ++i)
        _unroll_
        for (int j = 0; j != SH_DEGREE + 1; ++j)
            poly[i + j] += radius_poly[i] * radius_poly[j];
    // Subtract the scaled (2 * SH_DEGREE + 2)-th power of the distance to the
    // glyph center
    float dot_sq = closest_dot_origin * closest_dot_origin;
    float binomial = 1.0;
    _unroll_
    for (int i = 0; i != SH_DEGREE + 2; ++i) {
        poly[2 * i] -= binomial * dot_sq;    
        // Update the binomial coefficient using a recurrence relation
        binomial *= float(SH_DEGREE + 1 - i) / float(i + 1);
    }
    // Find roots of the polynomial within the relevant bounds
    float roots[MAX_DEGREE + 1];
    find_roots(roots, poly, -u_max, u_max);
    // Convert them back to the original coordinate frame (i.e. ray parameters)
    _unroll_
    for (int i = 0; i != MAX_DEGREE; ++i)
        out_ray_params[i] = (roots[i] != NO_INTERSECTION)
                            ? (roots[i] * closest_dot_origin - dir_dot_origin)
                            : NO_INTERSECTION;
}


// Provides a normalized normal vector for a spherical harmonics glyph.
// \param sh_coeffs SH_COUNT spherical harmonic coefficients defining the
//        glyph. Their exact meaning is defined by eval_sh().
// \param point A point on the surface of the glyph, relative to its center.
// return A normalized surface normal pointing away from the origin.
vec3 get_sh_glyph_normal(float sh_coeffs[SH_COUNT], vec3 point) {
    float shs[SH_COUNT];
    vec3 grads[SH_COUNT];
    float length_inv = inversesqrt(dot(point, point));
    vec3 normalized = point * length_inv;
    eval_sh_grad(shs, grads, normalized);
    float value = 0.0;
    vec3 grad = vec3(0.0);
    _unroll_
    for (int i = 0; i != SH_COUNT; ++i) {
        value += sh_coeffs[i] * shs[i];
        grad += sh_coeffs[i] * grads[i];
    }
    return normalize(point - (value * length_inv) * (grad - dot(grad, normalized) * normalized));
}


// This is the glTF BRDF for dielectric materials, exactly as described here:
// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
// \param incoming The normalized incoming light direction.
// \param outgoing The normalized outgoing light direction.
// \param normal The normalized shading normal.
// \param roughness An artist friendly roughness value between 0 and 1
// \param base_color The albedo used for the Lambertian diffuse component
// return The BRDF for the given directions.
vec3 gltf_dielectric_brdf(vec3 incoming, vec3 outgoing, vec3 normal, float roughness, vec3 base_color) {
    float ni = dot(normal, incoming);
    float no = dot(normal, outgoing);
    // Early out if incoming or outgoing direction are below the horizon
    if (ni <= 0.0 || no <= 0.0)
        return vec3(0.0);
    // Save some work by not actually computing the half-vector. If the half-
    // vector were h, ih = dot(incoming, h) and
    // sqrt(nh_ih_2 / ih_2) = dot(normal, h).
    float ih_2 = dot(incoming, outgoing) * 0.5 + 0.5;
    float sum = ni + no;
    float nh_ih_2 = 0.25 * sum * sum;
    float ih = sqrt(ih_2);

    // Evaluate the GGX normal distribution function
    float roughness_2 = roughness * roughness;
    float roughness_4  = roughness_2 * roughness_2;
    float roughness_flip = 1.0 - roughness_4;
    float denominator = ih_2 - nh_ih_2 * roughness_flip;
    float ggx = (roughness_4 * M_INV_PI * ih_2) / (denominator * denominator);
    // Evaluate the "visibility" (i.e. masking-shadowing times geometry terms)
    float vi = ni + sqrt(roughness_4 + roughness_flip * ni * ni);
    float vo = no + sqrt(roughness_4 + roughness_flip * no * no);
    float v = 1.0 / (vi * vo);
    // That completes the specular BRDF
    float specular = v * ggx;

    // The diffuse BRDF is Lambertian
    vec3 diffuse = M_INV_PI * base_color;

    // Evaluate the Fresnel term using the Fresnel-Schlick approximation
    const float ior = 1.5;
    const float f0 = ((1.0 - ior) / (1.0 + ior)) * ((1.0 - ior) / (1.0 + ior));
    float ih_flip = 1.0 - ih;
    float ih_flip_2 = ih_flip * ih_flip;
    float fresnel = f0 + (1.0 - f0) * ih_flip * ih_flip_2 * ih_flip_2;

    // Mix the two components
    return mix(diffuse, vec3(specular), fresnel);
}


// Applies the non-linearity that maps linear RGB to sRGB
float linear_to_srgb(float linear) {
    return (linear <= 0.0031308) ? (12.92 * linear) : (1.055 * pow(linear, 1.0 / 2.4) - 0.055);
}

// Inverse of linear_to_srgb()
float srgb_to_linear(float non_linear) {
    return (non_linear <= 0.04045) ? ((1.0 / 12.92) * non_linear) : pow(non_linear * (1.0 / 1.055) + 0.055 / 1.055, 2.4);
}

// Turns a linear RGB color (i.e. rec. 709) into sRGB
vec3 linear_rgb_to_srgb(vec3 linear) {
    return vec3(linear_to_srgb(linear.r), linear_to_srgb(linear.g), linear_to_srgb(linear.b));
}

// Inverse of linear_rgb_to_srgb()
vec3 srgb_to_linear_rgb(vec3 srgb) {
    return vec3(srgb_to_linear(srgb.r), srgb_to_linear(srgb.g), srgb_to_linear(srgb.b));
}

// Logarithmic tonemapping operator. Input and output are linear RGB.
vec3 tonemap(vec3 linear) {
    float max_channel = max(max(1.0, linear.r), max(linear.g, linear.b));
    return linear * ((1.0 - 0.02 * log2(max_channel)) / max_channel);
}

vec3 iResolution = vec3(1920, 1080, 1.0);
float iTime = 1.0;
void mainImage(out vec4 out_color, vec2 frag_coord) {
    // Define a camera ray for a pinhole camera
    vec3 camera_pos = vec3(0.0, -5.0, 0.0);
    float aspect = float(iResolution.x) / float(iResolution.y);
    float zoom = 0.8;
    vec3 right = (aspect / zoom) * vec3(3.0, 0.0, 0.0);
    vec3 up = (1.0 / zoom) * vec3(0.0, 0.0, 3.0);
    vec3 bottom_left = -0.5 * (right + up);
    vec2 uv = frag_coord / vec2(iResolution.xy);
    vec3 ray_dir = normalize(bottom_left + uv.x * right + uv.y * up - camera_pos);
    // Rotate the camera slowly
    float pitch = -0.2 * M_PI;
    float yaw = 0.1 * M_PI * iTime;
    mat3 rot_z = mat3(cos(yaw), sin(yaw), 0.0, -sin(yaw), cos(yaw), 0.0, 0.0, 0.0, 1.0);
    mat3 rot_x = mat3(1.0, 0.0, 0.0, 0.0, cos(pitch), sin(pitch), 0.0, -sin(pitch), cos(pitch));
    camera_pos = rot_z * rot_x * camera_pos;
    ray_dir = normalize(rot_z * rot_x * ray_dir);
    // Define SH coefficients (measured up to band 8, noise beyond that)
    float sh_coeffs[SH_COUNT];
    sh_coeffs[0] = -0.2739740312099;  sh_coeffs[1] = 0.2526670396328;  sh_coeffs[2] = 1.8922271728516;  sh_coeffs[3] = 0.2878578901291;  sh_coeffs[4] = -0.5339795947075;  sh_coeffs[5] = -0.2620058953762;
#if SH_DEGREE >= 4
    sh_coeffs[6] = 0.1580424904823;  sh_coeffs[7] = 0.0329004973173;  sh_coeffs[8] = -0.1322413831949;  sh_coeffs[9] = -0.1332057565451;  sh_coeffs[10] = 1.0894461870193;  sh_coeffs[11] = -0.6319401264191;  sh_coeffs[12] = -0.0416776277125;  sh_coeffs[13] = -1.0772529840469;  sh_coeffs[14] = 0.1423762738705;
#endif
#if SH_DEGREE >= 6
    sh_coeffs[15] = 0.7941166162491;  sh_coeffs[16] = 0.7490307092667;  sh_coeffs[17] = -0.3428381681442;  sh_coeffs[18] = 0.1024847552180;  sh_coeffs[19] = -0.0219132602215;  sh_coeffs[20] = 0.0499043911695;  sh_coeffs[21] = 0.2162453681231;  sh_coeffs[22] = 0.0921059995890;  sh_coeffs[23] = -0.2611238956451;  sh_coeffs[24] = 0.2549301385880;  sh_coeffs[25] = -0.4534865319729;  sh_coeffs[26] = 0.1922748684883;  sh_coeffs[27] = -0.6200597286224;
#endif
#if SH_DEGREE >= 8
    sh_coeffs[28] = -0.0532187558711;  sh_coeffs[29] = -0.3569841980934;  sh_coeffs[30] = 0.0293972902000;  sh_coeffs[31] = -0.1977960765362;  sh_coeffs[32] = -0.1058669015765;  sh_coeffs[33] = 0.2372217923403;  sh_coeffs[34] = -0.1856198310852;  sh_coeffs[35] = -0.3373193442822;  sh_coeffs[36] = -0.0750469490886;  sh_coeffs[37] = 0.2146576642990;  sh_coeffs[38] = -0.0490148440003;  sh_coeffs[39] = 0.1288588196039;  sh_coeffs[40] = 0.3173974752426;  sh_coeffs[41] = 0.1990085393190;  sh_coeffs[42] = -0.1736343950033;  sh_coeffs[43] = -0.0482443645597;  sh_coeffs[44] = 0.1749017387629;
#endif
#if SH_DEGREE >= 10
    sh_coeffs[45] = -0.0151847425660;  sh_coeffs[46] = 0.0418366046081;  sh_coeffs[47] = 0.0863263587216;  sh_coeffs[48] = -0.0649211244490;  sh_coeffs[49] = 0.0126096132283;  sh_coeffs[50] = 0.0545089217982;  sh_coeffs[51] = -0.0275142164626;  sh_coeffs[52] = 0.0399986574832;  sh_coeffs[53] = -0.0468244261610;  sh_coeffs[54] = -0.1292105653111;  sh_coeffs[55] = -0.0786858322658;  sh_coeffs[56] = -0.0663828464882;  sh_coeffs[57] = 0.0382439706831;  sh_coeffs[58] = -0.0041550330365;  sh_coeffs[59] = -0.0502800566338;  sh_coeffs[60] = -0.0732471630735;  sh_coeffs[61] = 0.0181751900972;  sh_coeffs[62] = -0.0090119333757;  sh_coeffs[63] = -0.0604443282359;  sh_coeffs[64] = -0.1469985252752;  sh_coeffs[65] = -0.0534046899715;
#endif
#if SH_DEGREE >= 12
    sh_coeffs[66] = -0.0896672753415;  sh_coeffs[67] = -0.0130841364808;  sh_coeffs[68] = -0.0112942893801;  sh_coeffs[69] = 0.0272257498541;  sh_coeffs[70] = 0.0626717616331;  sh_coeffs[71] = -0.0222197983050;  sh_coeffs[72] = -0.0018541504308;  sh_coeffs[73] = -0.1653251944056;  sh_coeffs[74] = 0.0409697402846;  sh_coeffs[75] = 0.0749921454327;  sh_coeffs[76] = -0.0282830872616;  sh_coeffs[77] = 0.0006909458525;  sh_coeffs[78] = 0.0625599842287;  sh_coeffs[79] = 0.0812529816082;  sh_coeffs[80] = 0.0914693020772;  sh_coeffs[81] = -0.1197222726745;  sh_coeffs[82] = 0.0376277453183;  sh_coeffs[83] = -0.0832617004142;  sh_coeffs[84] = -0.0482175038043;  sh_coeffs[85] = -0.0839003635737;  sh_coeffs[86] = -0.0349423908400;  sh_coeffs[87] = 0.1204519568256;  sh_coeffs[88] = 0.0783745984003;  sh_coeffs[89] = 0.0297401205976;  sh_coeffs[90] = -0.0505947662525;
#endif
    // Perform the intersection test
    float ray_params[MAX_DEGREE];
    ray_sh_glyph_intersections(ray_params, sh_coeffs, camera_pos, ray_dir);
    // Identify the first intersection
    float first_ray_param = NO_INTERSECTION;
    _unroll_
    for (int i = 0; i != MAX_DEGREE; ++i) {
        if (ray_params[i] != NO_INTERSECTION && ray_params[i] > 0.0) {
            first_ray_param = ray_params[i];
            break;
        }
    }
    // Evaluate shading for a directional light
    vec3 color = vec3(1.0);
    if (first_ray_param != NO_INTERSECTION) {
        vec3 intersection = camera_pos + first_ray_param * ray_dir;
        vec3 normal = get_sh_glyph_normal(sh_coeffs, intersection);
        vec3 base_color = srgb_to_linear_rgb(abs(normalize(intersection)));
        const vec3 incoming = normalize(vec3(1.23, -4.56, 7.89));
        float ambient = 0.04;
        float exposure = 4.0;
        vec3 outgoing = -ray_dir;
        vec3 brdf = gltf_dielectric_brdf(incoming, outgoing, normal, 0.45, base_color);
        color = exposure * (brdf * max(0.0, dot(incoming, normal)) + base_color * ambient);
    }
    out_color = vec4(linear_rgb_to_srgb(tonemap(color)), 1.0);
}
    """

    # fmt: off
    fs_dec = compose_shader([
        fs_defs, fs_unifs, fs_vs_vars, coeffs_norm, legendre_polys,
        spherical_harmonics, sdf_map, central_diffs_normals, cast_ray,
        blinn_phong_model, common, new_code
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", decl_code=fs_dec, debug=False)

    sdf_frag_impl = """
    
    // ------------------------------------------------------------------------------------------------------------------
    vec3 pnt = vertexMCVSOutput.xyz;

    // Ray Origin
    // Camera position in world space
    vec3 ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;

    // Ray Direction
    vec3 rd = normalize(pnt - ro);

    // Light Direction
    vec3 ld = normalize(ro - pnt);

    ro += pnt - ro;

    vec3 t = castRay(ro, rd);

    vec2 frag_coord = gl_FragCoord.xy;
    vec3 camera_pos = ro; //vec3(0.0, -5.0, 0.0);
    float aspect = float(iResolution.x) / float(iResolution.y);
    float zoom = 0.8;
    vec3 right = (aspect / zoom) * vec3(3.0, 0.0, 0.0);
    vec3 up = (1.0 / zoom) * vec3(0.0, 0.0, 3.0);
    vec3 bottom_left = -0.5 * (right + up);
    vec2 uv = frag_coord / vec2(iResolution.xy);
    vec3 ray_dir = rd; //normalize(bottom_left + uv.x * right + uv.y * up - camera_pos);
    // Rotate the camera slowly
    float pitch = -0.2 * M_PI;
    float yaw = 0.1 * M_PI * iTime;
    mat3 rot_z = mat3(cos(yaw), sin(yaw), 0.0, -sin(yaw), cos(yaw), 0.0, 0.0, 0.0, 1.0);
    mat3 rot_x = mat3(1.0, 0.0, 0.0, 0.0, cos(pitch), sin(pitch), 0.0, -sin(pitch), cos(pitch));
    camera_pos = rot_z * rot_x * camera_pos;
    ray_dir = normalize(rot_z * rot_x * ray_dir);
    // Define SH coefficients (measured up to band 8, noise beyond that)
    float sh_coeffs[SH_COUNT];
    sh_coeffs[0] = -0.2739740312099;  sh_coeffs[1] = 0.2526670396328;  sh_coeffs[2] = 1.8922271728516;  sh_coeffs[3] = 0.2878578901291;  sh_coeffs[4] = -0.5339795947075;  sh_coeffs[5] = -0.2620058953762;
#if SH_DEGREE >= 4
    sh_coeffs[6] = 0.1580424904823;  sh_coeffs[7] = 0.0329004973173;  sh_coeffs[8] = -0.1322413831949;  sh_coeffs[9] = -0.1332057565451;  sh_coeffs[10] = 1.0894461870193;  sh_coeffs[11] = -0.6319401264191;  sh_coeffs[12] = -0.0416776277125;  sh_coeffs[13] = -1.0772529840469;  sh_coeffs[14] = 0.1423762738705;
#endif
#if SH_DEGREE >= 6
    sh_coeffs[15] = 0.7941166162491;  sh_coeffs[16] = 0.7490307092667;  sh_coeffs[17] = -0.3428381681442;  sh_coeffs[18] = 0.1024847552180;  sh_coeffs[19] = -0.0219132602215;  sh_coeffs[20] = 0.0499043911695;  sh_coeffs[21] = 0.2162453681231;  sh_coeffs[22] = 0.0921059995890;  sh_coeffs[23] = -0.2611238956451;  sh_coeffs[24] = 0.2549301385880;  sh_coeffs[25] = -0.4534865319729;  sh_coeffs[26] = 0.1922748684883;  sh_coeffs[27] = -0.6200597286224;
#endif
#if SH_DEGREE >= 8
    sh_coeffs[28] = -0.0532187558711;  sh_coeffs[29] = -0.3569841980934;  sh_coeffs[30] = 0.0293972902000;  sh_coeffs[31] = -0.1977960765362;  sh_coeffs[32] = -0.1058669015765;  sh_coeffs[33] = 0.2372217923403;  sh_coeffs[34] = -0.1856198310852;  sh_coeffs[35] = -0.3373193442822;  sh_coeffs[36] = -0.0750469490886;  sh_coeffs[37] = 0.2146576642990;  sh_coeffs[38] = -0.0490148440003;  sh_coeffs[39] = 0.1288588196039;  sh_coeffs[40] = 0.3173974752426;  sh_coeffs[41] = 0.1990085393190;  sh_coeffs[42] = -0.1736343950033;  sh_coeffs[43] = -0.0482443645597;  sh_coeffs[44] = 0.1749017387629;
#endif
#if SH_DEGREE >= 10
    sh_coeffs[45] = -0.0151847425660;  sh_coeffs[46] = 0.0418366046081;  sh_coeffs[47] = 0.0863263587216;  sh_coeffs[48] = -0.0649211244490;  sh_coeffs[49] = 0.0126096132283;  sh_coeffs[50] = 0.0545089217982;  sh_coeffs[51] = -0.0275142164626;  sh_coeffs[52] = 0.0399986574832;  sh_coeffs[53] = -0.0468244261610;  sh_coeffs[54] = -0.1292105653111;  sh_coeffs[55] = -0.0786858322658;  sh_coeffs[56] = -0.0663828464882;  sh_coeffs[57] = 0.0382439706831;  sh_coeffs[58] = -0.0041550330365;  sh_coeffs[59] = -0.0502800566338;  sh_coeffs[60] = -0.0732471630735;  sh_coeffs[61] = 0.0181751900972;  sh_coeffs[62] = -0.0090119333757;  sh_coeffs[63] = -0.0604443282359;  sh_coeffs[64] = -0.1469985252752;  sh_coeffs[65] = -0.0534046899715;
#endif
#if SH_DEGREE >= 12
    sh_coeffs[66] = -0.0896672753415;  sh_coeffs[67] = -0.0130841364808;  sh_coeffs[68] = -0.0112942893801;  sh_coeffs[69] = 0.0272257498541;  sh_coeffs[70] = 0.0626717616331;  sh_coeffs[71] = -0.0222197983050;  sh_coeffs[72] = -0.0018541504308;  sh_coeffs[73] = -0.1653251944056;  sh_coeffs[74] = 0.0409697402846;  sh_coeffs[75] = 0.0749921454327;  sh_coeffs[76] = -0.0282830872616;  sh_coeffs[77] = 0.0006909458525;  sh_coeffs[78] = 0.0625599842287;  sh_coeffs[79] = 0.0812529816082;  sh_coeffs[80] = 0.0914693020772;  sh_coeffs[81] = -0.1197222726745;  sh_coeffs[82] = 0.0376277453183;  sh_coeffs[83] = -0.0832617004142;  sh_coeffs[84] = -0.0482175038043;  sh_coeffs[85] = -0.0839003635737;  sh_coeffs[86] = -0.0349423908400;  sh_coeffs[87] = 0.1204519568256;  sh_coeffs[88] = 0.0783745984003;  sh_coeffs[89] = 0.0297401205976;  sh_coeffs[90] = -0.0505947662525;
#endif
    // Perform the intersection test
    float ray_params[MAX_DEGREE];
    ray_sh_glyph_intersections(ray_params, sh_coeffs, camera_pos, ray_dir);
    // Identify the first intersection
    float first_ray_param = NO_INTERSECTION;
    _unroll_
    for (int i = 0; i != MAX_DEGREE; ++i) {
        if (ray_params[i] != NO_INTERSECTION && ray_params[i] > 0.0) {
            first_ray_param = ray_params[i];
            break;
        }
    }
    // Evaluate shading for a directional light
    vec3 color = vec3(1.0);
    if (first_ray_param != NO_INTERSECTION) {
        vec3 intersection = camera_pos + first_ray_param * ray_dir;
        vec3 normal = get_sh_glyph_normal(sh_coeffs, intersection);
        vec3 base_color = srgb_to_linear_rgb(abs(normalize(intersection)));
        const vec3 incoming = normalize(vec3(1.23, -4.56, 7.89));
        float ambient = 0.04;
        float exposure = 4.0;
        vec3 outgoing = -ray_dir;
        vec3 brdf = gltf_dielectric_brdf(incoming, outgoing, normal, 0.45, base_color);
        color = exposure * (brdf * max(0.0, dot(incoming, normal)) + base_color * ambient);
    }
    vec4 out_color = vec4(linear_rgb_to_srgb(tonemap(color)), 1.0);
    fragOutput0 = out_color;
    
    """

    shader_to_actor(
        odf_actor, "fragment", impl_code=sdf_frag_impl, block="picking"
    )

    show_man.scene.add(odf_actor)

    sphere = get_sphere("repulsion724")

    sh_basis = "descoteaux07"
    # sh_basis = "tournier07"
    sh_order = 4

    sh = np.zeros((3, 1, 1, 15))
    sh[0, 0, 0, :] = coeffs[0, :]
    sh[1, 0, 0, :] = coeffs[1, :]
    sh[2, 0, 0, :] = coeffs[2, :]

    tensor_sf = sh_to_sf(
        sh, sh_order=sh_order, basis_type=sh_basis, sphere=sphere
    )

    odf_slicer_actor = actor.odf_slicer(
        tensor_sf, sphere=sphere, scale=0.5, colormap="plasma"
    )

    show_man.scene.add(odf_slicer_actor)

    show_man.start()
