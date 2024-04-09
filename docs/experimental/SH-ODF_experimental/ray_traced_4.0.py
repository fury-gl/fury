"""
Fury's simplified version of the script ray_traced_2.0.py.
 - Simplified color calculation.
 - Simplified lighting.
"""

import os

import numpy as np

from fury import actor, window
from fury.lib import FloatArray, Texture
from fury.shaders import (
    attribute_to_actor,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)
from fury.utils import numpy_to_vtk_image_data, set_polydata_tcoords


def uv_calculations(n):
    uvs = []
    for i in range(0, n):
        a = (n - (i + 1)) / n
        b = (n - i) / n
        # glyph_coord [0, a], [0, b], [1, b], [1, a]
        uvs.extend(
            [
                [0.1, a + 0.1],
                [0.1, b - 0.1],
                [0.9, b - 0.1],
                [0.9, a + 0.1],
                [0.1, a + 0.1],
                [0.1, b - 0.1],
                [0.9, b - 0.1],
                [0.9, a + 0.1],
            ]
        )
    return uvs


if __name__ == "__main__":
    show_man = window.ShowManager(size=(1280, 720))
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

    odf_actor = actor.box(centers=centers, scales=1.0)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    odf_actor_pd = odf_actor.GetMapper().GetInput()

    uv_vals = np.array(uv_calculations(3))

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
    arr = np.array(
        [
            (coeffs[i] - min[i]) * ((newmax - newmin) / (max[i] - min[i]))
            + newmin
            for i in range(coeffs.shape[0])
        ]
    )
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
    in vec2 minmax;

    out vec4 vertexMCVSOutput;
    out vec3 centerMCVSOutput;
    out vec2 minmaxVSOutput;
    out vec3 camPosMCVSOutput;
    out vec3 camRightMCVSOutput;
    out vec3 camUpMCVSOutput;
    """

    vs_impl = """
    vertexMCVSOutput = vertexMC;
    centerMCVSOutput = center;
    minmaxVSOutput = minmax;
    camPosMCVSOutput = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    camRightMCVSOutput = vec3(
        MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
    camUpMCVSOutput = vec3(
        MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
    """

    shader_to_actor(odf_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    # The index of the highest used band of the spherical harmonics basis. Must
    # be even, at least 2 and at most 12.
    def_sh_degree = "#define SH_DEGREE 4"

    # The number of spherical harmonics basis functions
    def_sh_count = "#define SH_COUNT (((SH_DEGREE + 1) * (SH_DEGREE + 2)) / 2)"

    # Degree of polynomials for which we have to find roots
    def_max_degree = "#define MAX_DEGREE (2 * SH_DEGREE + 2)"

    # If GL_EXT_control_flow_attributes is available, these defines should be
    # defined as [[unroll]] and [[loop]] to give reasonable hints to the
    # compiler. That avoids register spilling, which makes execution
    # considerably faster.
    def_gl_ext_control_flow_attributes = """
    #ifndef _unroll_
        #define _unroll_
    #endif
    #ifndef _loop_
        #define _loop_
    #endif
    """

    # When there are fewer intersections/roots than theoretically possible,
    # some array entries are set to this value
    def_no_intersection = "#define NO_INTERSECTION 3.4e38"

    # pi and its reciprocal
    def_pis = """
    #define M_PI 3.141592653589793238462643
    #define M_INV_PI 0.318309886183790671537767526745
    """

    fs_vs_vars = """
    in vec4 vertexMCVSOutput;
    in vec3 centerMCVSOutput;
    in vec2 minmaxVSOutput;
    in vec3 camPosMCVSOutput;
    in vec3 camRightMCVSOutput;
    in vec3 camUpMCVSOutput;
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

    eval_sh_2 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_2.frag")
    )

    eval_sh_4 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_4.frag")
    )

    eval_sh_6 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_6.frag")
    )

    eval_sh_8 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_8.frag")
    )

    eval_sh_10 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_10.frag")
    )

    eval_sh_12 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_12.frag")
    )

    eval_sh_grad_2 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_grad_2.frag")
    )

    eval_sh_grad_4 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_grad_4.frag")
    )

    eval_sh_grad_6 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_grad_6.frag")
    )

    eval_sh_grad_8 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_grad_8.frag")
    )

    eval_sh_grad_10 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_grad_10.frag")
    )

    eval_sh_grad_12 = import_fury_shader(
        os.path.join("rt_odfs", "descoteaux", "eval_sh_grad_12.frag")
    )

    # Searches a single root of a polynomial within a given interval.
    #   param out_root The location of the found root.
    #   param out_end_value The value of the given polynomial at end.
    #   param poly Coefficients of the polynomial for which a root should be
    #       found.
    #       Coefficient poly[i] is multiplied by x^i.
    #   param begin The beginning of an interval where the polynomial is
    #       monotonic.
    #   param end The end of said interval.
    #   param begin_value The value of the given polynomial at begin.
    #   param error_tolerance The error tolerance for the returned root
    #       location.
    #       Typically the error will be much lower but in theory it can be
    #       bigger.
    #
    #   return true if a root was found, false if no root exists.
    newton_bisection = import_fury_shader(
        os.path.join("root_finding", "newton_bisection.frag")
    )

    # Finds all roots of the given polynomial in the interval [begin, end] and
    # writes them to out_roots. Some entries will be NO_INTERSECTION but other
    # than that the array is sorted. The last entry is always NO_INTERSECTION.
    find_roots = import_fury_shader(os.path.join("rt_odfs", "find_roots.frag"))

    # Evaluates the spherical harmonics basis in bands 0, 2, ..., SH_DEGREE.
    # Conventions are as in the following paper.
    # M. Descoteaux, E. Angelino, S. Fitzgibbons, and R. Deriche. Regularized,
    # fast, and robust analytical q-ball imaging. Magnetic Resonance in
    # Medicine, 58(3), 2007. https://doi.org/10.1002/mrm.21277
    #   param out_shs Values of SH basis functions in bands 0, 2, ...,
    #       SH_DEGREE in this order.
    #   param point The point on the unit sphere where the basis should be
    #       evaluated.
    eval_sh = import_fury_shader(os.path.join("rt_odfs", "eval_sh.frag"))

    # Evaluates the gradient of each basis function given by eval_sh() and the
    # basis itself
    eval_sh_grad = import_fury_shader(
        os.path.join("rt_odfs", "eval_sh_grad.frag")
    )

    # Outputs a matrix that turns equidistant samples on the unit circle of a
    # homogeneous polynomial into coefficients of that polynomial.
    get_inv_vandermonde = import_fury_shader(
        os.path.join("rt_odfs", "get_inv_vandermonde.frag")
    )

    # Determines all intersections between a ray and a spherical harmonics
    # glyph.
    #   param out_ray_params The ray parameters at intersection points. The
    #       points themselves are at ray_origin + out_ray_params[i] * ray_dir.
    #       Some entries may be NO_INTERSECTION but other than that the array
    #       is sorted.
    #   param sh_coeffs SH_COUNT spherical harmonic coefficients defining the
    #       glyph. Their exact meaning is defined by eval_sh().
    #   param ray_origin The origin of the ray, relative to the glyph center.
    #   param ray_dir The normalized direction vector of the ray.
    ray_sh_glyph_intersections = import_fury_shader(
        os.path.join("rt_odfs", "ray_sh_glyph_intersections.frag")
    )

    # Provides a normalized normal vector for a spherical harmonics glyph.
    #   param sh_coeffs SH_COUNT spherical harmonic coefficients defining the
    #       glyph. Their exact meaning is defined by eval_sh().
    #   param point A point on the surface of the glyph, relative to its
    #       center.
    #
    #   return A normalized surface normal pointing away from the origin.
    get_sh_glyph_normal = import_fury_shader(
        os.path.join("rt_odfs", "get_sh_glyph_normal.frag")
    )

    # Applies the non-linearity that maps linear RGB to sRGB
    linear_to_srgb = import_fury_shader(
        os.path.join("rt_odfs", "linear_to_srgb.frag")
    )

    # Inverse of linear_to_srgb()
    srgb_to_linear = import_fury_shader(
        os.path.join("rt_odfs", "srgb_to_linear.frag")
    )

    # Turns a linear RGB color (i.e. rec. 709) into sRGB
    linear_rgb_to_srgb = import_fury_shader(
        os.path.join("rt_odfs", "linear_rgb_to_srgb.frag")
    )

    # Inverse of linear_rgb_to_srgb()
    srgb_to_linear_rgb = import_fury_shader(
        os.path.join("rt_odfs", "srgb_to_linear_rgb.frag")
    )

    # Logarithmic tonemapping operator. Input and output are linear RGB.
    tonemap = import_fury_shader(os.path.join("rt_odfs", "tonemap.frag"))

    # Blinn-Phong illumination model
    blinn_phong_model = import_fury_shader(
        os.path.join("lighting", "blinn_phong_model.frag")
    )

    # fmt: off
    fs_dec = compose_shader([
        def_sh_degree, def_sh_count, def_max_degree,
        def_gl_ext_control_flow_attributes, def_no_intersection, def_pis,
        fs_vs_vars, coeffs_norm, eval_sh_2, eval_sh_4, eval_sh_6, eval_sh_8,
        eval_sh_10, eval_sh_12, eval_sh_grad_2, eval_sh_grad_4, eval_sh_grad_6,
        eval_sh_grad_8, eval_sh_grad_10, eval_sh_grad_12, newton_bisection,
        find_roots, eval_sh, eval_sh_grad, get_inv_vandermonde,
        ray_sh_glyph_intersections, get_sh_glyph_normal, blinn_phong_model,
        linear_to_srgb, srgb_to_linear, linear_rgb_to_srgb, srgb_to_linear_rgb,
        tonemap
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", decl_code=fs_dec)

    point_from_vs = "vec3 pnt = vertexMCVSOutput.xyz;"

    # Ray origin is the camera position in world space
    ray_origin = "vec3 ro = camPosMCVSOutput;"

    # TODO: Check aspect for automatic scaling
    # Ray direction is the normalized difference between the fragment and the
    # camera position/ray origin
    ray_direction = "vec3 rd = normalize(pnt - ro);"

    # Light direction in a retroreflective model is the normalized difference
    # between the camera position/ray origin and the fragment
    light_direction = "vec3 ld = normalize(ro - pnt);"

    # Define SH coefficients (measured up to band 8, noise beyond that)
    sh_coeffs = """
    float i = 1 / (numCoeffs * 2);
    float sh_coeffs[SH_COUNT];
    for(int j=0; j<numCoeffs; j++){
        sh_coeffs[j] = coeffsNorm(texture(texture0, vec2(i + j / numCoeffs, tcoordVCVSOutput.y)).x);
    }
    """

    # Perform the intersection test
    intersection_test = """
    float ray_params[MAX_DEGREE];
    ray_sh_glyph_intersections(ray_params, sh_coeffs, ro - centerMCVSOutput, rd);
    """

    # Identify the first intersection
    first_intersection = """
    float first_ray_param = NO_INTERSECTION;
    _unroll_
    for (int i = 0; i != MAX_DEGREE; ++i) {
        if (ray_params[i] != NO_INTERSECTION && ray_params[i] > 0.0) {
            first_ray_param = ray_params[i];
            break;
        }
    }
    """

    # Evaluate shading for a directional light
    directional_light = """
    vec3 color = vec3(1.);
    if (first_ray_param != NO_INTERSECTION) {
        //vec3 intersection = ro + first_ray_param * rd;
        vec3 intersection = ro - centerMCVSOutput + first_ray_param * rd;
        vec3 normal = get_sh_glyph_normal(sh_coeffs, intersection);
        vec3 colorDir = srgb_to_linear_rgb(abs(normalize(intersection)));
        float attenuation = dot(ld, normal);
        color = blinnPhongIllumModel(
            //attenuation, lightColor0, diffuseColor, specularPower,
            attenuation, lightColor0, colorDir, specularPower,
            specularColor, ambientColor);
    } else {
        discard;
    }
    """

    frag_output = """
    //vec4 out_color = vec4(linear_rgb_to_srgb(tonemap(color)), 1.0);
    vec3 out_color = linear_rgb_to_srgb(tonemap(color));
    fragOutput0 = vec4(out_color, opacity);
    //fragOutput0 = vec4(color, opacity);
    """

    # fmt: off
    fs_impl = compose_shader([
        point_from_vs, ray_origin, ray_direction, light_direction, sh_coeffs,
        intersection_test, first_intersection, directional_light, frag_output
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", impl_code=fs_impl, block="picking")
    show_man.scene.add(odf_actor)

    show_man.start()
