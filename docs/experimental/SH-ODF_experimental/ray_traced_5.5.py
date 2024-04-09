"""
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


def uv_calculations(n):
    uvs = []
    for i in range(0, n):
        a = (n - (i + 1)) / n
        b = (n - i) / n
        # glyph_coord [0, a], [0, b], [1, b], [1, a]
        uvs.extend(
            [
                [0.01, a + 0.01],
                [0.01, b - 0.01],
                [0.99, b - 0.01],
                [0.99, a + 0.01],
                [0.01, a + 0.01],
                [0.01, b - 0.01],
                [0.99, b - 0.01],
                [0.99, a + 0.01],
            ]
        )
    return uvs


if __name__ == "__main__":
    show_man = window.ShowManager(size=(1280, 720))
    show_man.scene.background((1, 1, 1))

    # fmt: off
    coeffs = np.array([
        [
            -0.2739740312099, 0.2526670396328, 1.8922271728516, 0.2878578901291, -0.5339795947075,
            -0.2620058953762, 0.1580424904823, 0.0329004973173, -0.1322413831949, -0.1332057565451,
            1.0894461870193, -0.6319401264191, -0.0416776277125, -1.0772529840469,  0.1423762738705,
            0.7941166162491, 0.7490307092667, -0.3428381681442, 0.1024847552180, -0.0219132602215,
            0.0499043911695, 0.2162453681231, 0.0921059995890, -0.2611238956451, 0.2549301385880,
            -0.4534865319729, 0.1922748684883, -0.6200597286224
        ]
    ])

    """
            , -0.0532187558711, -0.3569841980934,
            0.0293972902000, -0.1977960765362, -0.1058669015765, 0.2372217923403, -0.1856198310852,
            -0.3373193442822, -0.0750469490886, 0.2146576642990, -0.0490148440003, 0.1288588196039,
            0.3173974752426, 0.1990085393190, -0.1736343950033, -0.0482443645597, 0.1749017387629,
            -0.0151847425660, 0.0418366046081, 0.0863263587216, -0.0649211244490, 0.0126096132283,
            0.0545089217982, -0.0275142164626, 0.0399986574832, -0.0468244261610, -0.1292105653111,
            -0.0786858322658, -0.0663828464882, 0.0382439706831, -0.0041550330365, -0.0502800566338,
            -0.0732471630735, 0.0181751900972, -0.0090119333757, -0.0604443282359, -0.1469985252752,
            -0.0534046899715, -0.0896672753415, -0.0130841364808, -0.0112942893801, 0.0272257498541,
            0.0626717616331, -0.0222197983050, -0.0018541504308, -0.1653251944056, 0.0409697402846,
            0.0749921454327, -0.0282830872616, 0.0006909458525, 0.0625599842287, 0.0812529816082,
            0.0914693020772, -0.1197222726745, 0.0376277453183, -0.0832617004142, -0.0482175038043,
            -0.0839003635737, -0.0349423908400,  0.1204519568256, 0.0783745984003, 0.0297401205976,
            -0.0505947662525
    """

    centers = np.array([[0, -2.5, 0]])

    odf_actor = actor.box(centers=centers, scales=4)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    odf_actor_pd = odf_actor.GetMapper().GetInput()

    # fmt: off
    uv_vals = np.array(uv_calculations(1))
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
        "numCoeffs", 28
    )

    vs_dec = """
    in vec3 center;
    in vec2 minmax;

    out vec4 vertexMCVSOutput;
    out vec3 centerMCVSOutput;
    out vec2 minmaxVSOutput;
    """

    vs_impl = """
    vertexMCVSOutput = vertexMC;
    centerMCVSOutput = center;
    minmaxVSOutput = minmax;
    vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    """

    shader_to_actor(odf_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    # The index of the highest used band of the spherical harmonics basis. Must
    # be even, at least 2 and at most 12.
    def_sh_degree = "#define SH_DEGREE 6"

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

    fs_unifs = """
    uniform mat4 MCVCMatrix;
    """

    fs_vs_vars = """
    in vec4 vertexMCVSOutput;
    in vec3 centerMCVSOutput;
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

    eval_sh_2 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_2.frag")
    )

    eval_sh_4 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_4.frag")
    )

    eval_sh_6 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_6.frag")
    )

    eval_sh_8 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_8.frag")
    )

    eval_sh_10 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_10.frag")
    )

    eval_sh_12 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_12.frag")
    )

    eval_sh_grad_2 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_grad_2.frag")
    )

    eval_sh_grad_4 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_grad_4.frag")
    )

    eval_sh_grad_6 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_grad_6.frag")
    )

    eval_sh_grad_8 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_grad_8.frag")
    )

    eval_sh_grad_10 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_grad_10.frag")
    )

    eval_sh_grad_12 = import_fury_shader(
        os.path.join("rt_odfs", "tournier", "eval_sh_grad_12.frag")
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

    # This is the glTF BRDF for dielectric materials, exactly as described
    # here:
    # https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
    #   param incoming The normalized incoming light direction.
    #   param outgoing The normalized outgoing light direction.
    #   param normal The normalized shading normal.
    #   param roughness An artist friendly roughness value between 0 and 1.
    #   param base_color The albedo used for the Lambertian diffuse component.
    #
    #   return The BRDF for the given directions.
    gltf_dielectric_brdf = import_fury_shader(
        os.path.join("rt_odfs", "gltf_dielectric_brdf.frag")
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

    # fmt: off
    fs_dec = compose_shader([
        def_sh_degree, def_sh_count, def_max_degree,
        def_gl_ext_control_flow_attributes, def_no_intersection,
        def_pis, fs_unifs, fs_vs_vars, coeffs_norm, eval_sh_2, eval_sh_4, eval_sh_6,
        eval_sh_8, eval_sh_10, eval_sh_12, eval_sh_grad_2, eval_sh_grad_4,
        eval_sh_grad_6, eval_sh_grad_8, eval_sh_grad_10, eval_sh_grad_12,
        newton_bisection, find_roots, eval_sh, eval_sh_grad,
        get_inv_vandermonde, ray_sh_glyph_intersections, get_sh_glyph_normal,
        gltf_dielectric_brdf, linear_to_srgb, srgb_to_linear,
        linear_rgb_to_srgb, srgb_to_linear_rgb, tonemap
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", decl_code=fs_dec)

    point_from_vs = "vec3 pnt = vertexMCVSOutput.xyz;"

    # Ray origin is the camera position in world space
    ray_origin = """
    vec3 ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;
    """

    # TODO: Check aspect for automatic scaling
    # Ray direction is the normalized difference between the fragment and the
    # camera position/ray origin
    ray_direction = """
    vec3 rd = normalize(pnt - ro);
    """

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
    vec3 color = vec3(0.5);
    if (first_ray_param != NO_INTERSECTION) {
        vec3 intersection = ro - centerMCVSOutput + first_ray_param * rd;
        vec3 normal = get_sh_glyph_normal(sh_coeffs, intersection);
        vec3 base_color = srgb_to_linear_rgb(abs(normalize(intersection)));
        const vec3 incoming = normalize(vec3(1.23, -4.56, 7.89));
        float ambient = 0.04;
        float exposure = 4.0;
        vec3 outgoing = -rd;
        vec3 brdf = gltf_dielectric_brdf(incoming, outgoing, normal, 0.45, base_color);
        color = exposure * (brdf * max(0.0, dot(incoming, normal)) + base_color * ambient);
    } else {
        discard;
    }
    """

    frag_output = """
    //vec4 out_color = vec4(linear_rgb_to_srgb(tonemap(color)), 1.0);
    vec3 out_color = linear_rgb_to_srgb(tonemap(color));
    fragOutput0 = vec4(out_color, opacity);
    """

    # fmt: off
    fs_impl = compose_shader([
        point_from_vs, ray_origin, ray_direction, light_direction, sh_coeffs,
        intersection_test, first_intersection, directional_light, frag_output
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", impl_code=fs_impl, block="picking")
    show_man.scene.add(odf_actor)
    #'''
    sphere = get_sphere("repulsion724")

    sh_basis = "descoteaux07"
    # sh_basis = "tournier07"
    sh_order = 6

    n = int((sh_order / 2) + 1)
    sz = 2 * n**2 - n
    sh = np.zeros((1, 1, 1, sz))
    sh[0, 0, 0, :] = coeffs[0, :sz]

    tensor_sf = sh_to_sf(
        sh, sh_order=sh_order, basis_type=sh_basis, sphere=sphere, legacy=False
    )

    odf_slicer_actor = actor.odf_slicer(
        tensor_sf, sphere=sphere, scale=1, colormap="plasma"
    )

    show_man.scene.add(odf_slicer_actor)
    #'''
    show_man.start()
