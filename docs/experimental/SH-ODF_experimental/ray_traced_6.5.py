"""
"""

import os

import numpy as np
from dipy.data.fetcher import dipy_home
from dipy.io.image import load_nifti

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
                [0.001, a + 0.001],
                [0.001, b - 0.001],
                [0.999, b - 0.001],
                [0.999, a + 0.001],
                [0.001, a + 0.001],
                [0.001, b - 0.001],
                [0.999, b - 0.001],
                [0.999, a + 0.001],
            ]
        )
    return uvs


if __name__ == "__main__":
    show_man = window.ShowManager(size=(1280, 720))

    dataset_dir = os.path.join(dipy_home, "stanford_hardi")

    coeffs, affine = load_nifti(
        os.path.join(dataset_dir, "odf_debug_sh_coeffs_9x11x28(6).nii.gz")
    )

    max_num_coeffs = coeffs.shape[-1]

    max_sh_degree = int((np.sqrt(8 * max_num_coeffs + 1) - 3) / 2)

    max_poly_degree = 2 * max_sh_degree + 2

    viz_sh_degree = 6

    valid_mask = np.abs(coeffs).max(axis=(-1)) > 0
    indices = np.nonzero(valid_mask)

    centers = np.asarray(indices).T

    x, y, z, s = coeffs.shape
    coeffs = coeffs[:, :, :].reshape((x * y * z, s))
    n_glyphs = coeffs.shape[0]

    max_val = coeffs.min(axis=1)
    total = np.sum(abs(coeffs), axis=1)
    coeffs = np.dot(np.diag(1 / total), coeffs) * 1.7

    odf_actor = actor.box(centers=centers, scales=1)
    odf_actor.GetMapper().SetVBOShiftScaleMethod(False)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    odf_actor_pd = odf_actor.GetMapper().GetInput()

    uv_vals = np.array(uv_calculations(n_glyphs))

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

    odf_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformi(
        "shDegree", viz_sh_degree
    )

    vs_dec = """
    uniform int shDegree;

    in vec3 center;
    in vec2 minmax;

    flat out int numCoeffsVSOutput;
    flat out int maxPolyDegreeVSOutput;
    out vec4 vertexMCVSOutput;
    out vec3 centerMCVSOutput;
    out vec2 minmaxVSOutput;
    out vec3 camPosMCVSOutput;
    out vec3 camRightMCVSOutput;
    out vec3 camUpMCVSOutput;
    """

    vs_impl = """
    numCoeffsVSOutput = (shDegree + 1) * (shDegree + 2) / 2;
    maxPolyDegreeVSOutput = 2 * shDegree + 2;
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
    # def_sh_degree = "#define SH_DEGREE 4"
    def_sh_degree = f"#define SH_DEGREE {max_sh_degree}"

    # The number of spherical harmonics basis functions
    # def_sh_count = "#define SH_COUNT (((SH_DEGREE + 1) * (SH_DEGREE + 2)) / 2)"
    def_sh_count = f"#define SH_COUNT {max_num_coeffs}"

    # Degree of polynomials for which we have to find roots
    # def_max_degree = "#define MAX_DEGREE (2 * SH_DEGREE + 2)"
    def_max_degree = f"#define MAX_DEGREE {max_poly_degree}"

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
    flat in int numCoeffsVSOutput;
    flat in int maxPolyDegreeVSOutput;
    in vec4 vertexMCVSOutput;
    in vec3 centerMCVSOutput;
    in vec2 minmaxVSOutput;
    in vec3 camPosMCVSOutput;
    in vec3 camRightMCVSOutput;
    in vec3 camUpMCVSOutput;
    """

    minmax_norm = import_fury_shader(os.path.join("utils", "minmax_norm.glsl"))

    sh_basis = "descoteaux"
    # sh_basis = "tournier"

    eval_sh_composed = ""
    for i in range(2, max_sh_degree + 1, 2):
        eval_sh = import_fury_shader(
            os.path.join("rt_odfs", sh_basis, "eval_sh_" + str(i) + ".frag")
        )
        eval_sh_grad = import_fury_shader(
            os.path.join(
                "rt_odfs", sh_basis, "eval_sh_grad_" + str(i) + ".frag"
            )
        )
        eval_sh_composed = compose_shader(
            [eval_sh_composed, eval_sh, eval_sh_grad]
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
    # TODO: Pass numCoeffs
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
        fs_vs_vars, minmax_norm, eval_sh_composed, newton_bisection,
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
    float i = 1 / (numCoeffsVSOutput * 2);
    float sh_coeffs[SH_COUNT];
    for(int j=0; j < numCoeffsVSOutput; j++){
        sh_coeffs[j] = rescale(
            texture(
                texture0,
                vec2(i + j / numCoeffsVSOutput, tcoordVCVSOutput.y)
            ).x, 0, 1, minmaxVSOutput.x, minmaxVSOutput.y
        );
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
    //for (int i = 0; i != MAX_DEGREE; ++i) {
    for (int i = 0; i != maxPolyDegreeVSOutput; ++i) {
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
