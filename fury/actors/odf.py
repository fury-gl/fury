import os

import numpy as np

from fury import actor
from fury.lib import FloatArray, Texture
from fury.shaders import (
    attribute_to_actor,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)
from fury.utils import (
    numpy_to_vtk_image_data,
    set_polydata_tcoords,
    minmax_norm
)
from fury.texture.utils import uv_calculations


def sh_odf(centers, coeffs, degree, sh_basis, scales, opacity):
    """
    Visualize one or many ODFs with different features.

    Parameters
    ----------
    centers : ndarray(N, 3)
        ODFs positions.
    coeffs : ndarray
        2D ODFs array in SH coefficients.
    sh_basis: str, optional
        Type of basis (descoteaux, tournier)
        'descoteaux' for the default ``descoteaux07`` DYPY basis.
        'tournier' for the default ``tournier07` DYPY basis.
    degree: int, optional
        Index of the highest used band of the spherical harmonics basis. Must
        be even, at least 2 and at most 12.
    scales : float or ndarray (N, )
        ODFs size.
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    box_actor: Actor

    """
    odf_actor = actor.box(centers=centers, scales=scales)
    odf_actor.GetMapper().SetVBOShiftScaleMethod(False)
    odf_actor.GetProperty().SetOpacity(opacity)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    odf_actor_pd = odf_actor.GetMapper().GetInput()
    
    n_glyphs = coeffs.shape[0]
    # Coordinates to locate the data of each glyph in the texture.
    uv_vals = np.array(uv_calculations(n_glyphs))
    num_pnts = uv_vals.shape[0]
    
    # Definition of texture coordinates to be associated with the actor.
    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pnts)
    [t_coords.SetTuple(i, uv_vals[i]) for i in range(num_pnts)]

    set_polydata_tcoords(odf_actor_pd, t_coords)
    
    # The coefficient data is stored in a texture to be passed to the shaders.

    # Data is normalized to a range of 0 to 1.
    arr = minmax_norm(coeffs)
    # Data is turned into values within the RGB color range, and then converted
    # into a vtk image data.
    arr *= 255
    grid = numpy_to_vtk_image_data(arr.astype(np.uint8))

    # Vtk image data is associated to a texture.
    texture = Texture()
    texture.SetInputDataObject(grid)
    texture.Update()

    # Texture is associated with the actor
    odf_actor.GetProperty().SetTexture("texture0", texture)

    max_num_coeffs = coeffs.shape[-1]
    max_sh_degree = int((np.sqrt(8 * max_num_coeffs + 1) - 3) / 2)
    max_poly_degree = 2 * max_sh_degree + 2
    viz_sh_degree = max_sh_degree
    
    # The number of coefficients is associated to the order of the SH
    odf_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf(
        "shDegree", viz_sh_degree
    )

    # Start of shader implementation

    vs_dec = \
        """
        uniform float shDegree;
        
        in vec3 center;
        in vec2 minmax;

        flat out float numCoeffsVSOutput;
        flat out float maxPolyDegreeVSOutput;
        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out vec2 minmaxVSOutput;
        out vec3 camPosMCVSOutput;
        out vec3 camRightMCVSOutput;
        out vec3 camUpMCVSOutput;
        """

    vs_impl = \
        """
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
    def_sh_degree = f"#define SH_DEGREE {max_sh_degree}"

    # The number of spherical harmonics basis functions
    def_sh_count = f"#define SH_COUNT {max_num_coeffs}"

    # Degree of polynomials for which we have to find roots
    def_max_degree = f"#define MAX_DEGREE {max_poly_degree}"

    # If GL_EXT_control_flow_attributes is available, these defines should be
    # defined as [[unroll]] and [[loop]] to give reasonable hints to the
    # compiler. That avoids register spilling, which makes execution
    # considerably faster.
    def_gl_ext_control_flow_attributes = \
        """
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
    def_pis = \
        """
        #define M_PI 3.141592653589793238462643
        #define M_INV_PI 0.318309886183790671537767526745
        """

    fs_vs_vars = \
        """
        flat in float numCoeffsVSOutput;
        flat in float maxPolyDegreeVSOutput;
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in vec2 minmaxVSOutput;
        in vec3 camPosMCVSOutput;
        in vec3 camRightMCVSOutput;
        in vec3 camUpMCVSOutput;
        """

    coeffs_norm = import_fury_shader(os.path.join("utils", "minmax_norm.glsl"))

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
        os.path.join("utils", "newton_bisection.frag")
    )

    # Finds all roots of the given polynomial in the interval [begin, end] and
    # writes them to out_roots. Some entries will be NO_INTERSECTION but other
    # than that the array is sorted. The last entry is always NO_INTERSECTION.
    find_roots = import_fury_shader(os.path.join("utils", "find_roots.frag"))

    # Evaluates the spherical harmonics basis in bands 0, 2, ..., SH_DEGREE.
    # Conventions are as in the following paper.
    # M. Descoteaux, E. Angelino, S. Fitzgibbons, and R. Deriche. Regularized,
    # fast, and robust analytical q-ball imaging. Magnetic Resonance in
    # Medicine, 58(3), 2007. https://doi.org/10.1002/mrm.21277
    #   param outSH Values of SH basis functions in bands 0, 2, ...,
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
        os.path.join("lighting", "linear_to_srgb.frag")
    )

    # Inverse of linear_to_srgb()
    srgb_to_linear = import_fury_shader(
        os.path.join("lighting", "srgb_to_linear.frag")
    )

    # Turns a linear RGB color (i.e. rec. 709) into sRGB
    linear_rgb_to_srgb = import_fury_shader(
        os.path.join("lighting", "linear_rgb_to_srgb.frag")
    )

    # Inverse of linear_rgb_to_srgb()
    srgb_to_linear_rgb = import_fury_shader(
        os.path.join("lighting", "srgb_to_linear_rgb.frag")
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
        fs_vs_vars, coeffs_norm, eval_sh_composed, newton_bisection, find_roots,
        eval_sh, eval_sh_grad, get_inv_vandermonde, ray_sh_glyph_intersections,
        get_sh_glyph_normal, blinn_phong_model, linear_to_srgb, srgb_to_linear,
        linear_rgb_to_srgb, srgb_to_linear_rgb, tonemap
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", decl_code=fs_dec)

    point_from_vs = "vec3 pnt = vertexMCVSOutput.xyz;"

    # Ray origin is the camera position in world space
    ray_origin = "vec3 ro = camPosMCVSOutput;"

    # Ray direction is the normalized difference between the fragment and the
    # camera position/ray origin
    ray_direction = "vec3 rd = normalize(pnt - ro);"

    # Light direction in a retroreflective model is the normalized difference
    # between the camera position/ray origin and the fragment
    light_direction = "vec3 ld = normalize(ro - pnt);"

    # Define SH coefficients (measured up to band 8, noise beyond that)
    sh_coeffs = \
        """
        float i = 1 / (numCoeffsVSOutput * 2);
        float sh_coeffs[SH_COUNT];
        for(int j=0; j < numCoeffsVSOutput; j++){
            sh_coeffs[j] = rescale(
                texture(
                    texture0,
                    vec2(i + j / numCoeffsVSOutput, tcoordVCVSOutput.y)).x,
                    0, 1, minmaxVSOutput.x, minmaxVSOutput.y
            );
        }
        """

    # Perform the intersection test
    intersection_test = \
        """
        float ray_params[MAX_DEGREE];
        rayGlyphIntersections(
            ray_params, sh_coeffs, ro - centerMCVSOutput, rd, int(shDegree),
            int(numCoeffsVSOutput), int(maxPolyDegreeVSOutput), M_PI,
            NO_INTERSECTION
        );
        """

    # Identify the first intersection
    first_intersection = \
        """
        float first_ray_param = NO_INTERSECTION;
        _unroll_
        for (int i = 0; i != MAX_DEGREE; ++i) {
        //for (int i = 0; i != maxPolyDegreeVSOutput; ++i) {
            if (ray_params[i] != NO_INTERSECTION && ray_params[i] > 0.0) {
                first_ray_param = ray_params[i];
                break;
            }
        }
        """

    # Evaluate shading for a directional light
    directional_light = \
        """
        vec3 color = vec3(1.);
        if (first_ray_param != NO_INTERSECTION) {
            vec3 intersection = ro - centerMCVSOutput + first_ray_param * rd;
            vec3 normal = getShGlyphNormal(sh_coeffs, intersection, int(shDegree), int(numCoeffsVSOutput));
            vec3 colorDir = srgbToLinearRgb(abs(normalize(intersection)));
            float attenuation = ld.z;//dot(ld, normal);
            color = blinnPhongIllumModel(
                attenuation, lightColor0, colorDir, specularPower,
                specularColor, ambientColor);
        } else {
            discard;
        }
        """

    frag_output = \
        """
        vec3 out_color = linearRgbToSrgb(tonemap(color));
        fragOutput0 = vec4(out_color, opacity);
        """

    fs_impl = compose_shader([
        point_from_vs, ray_origin, ray_direction, light_direction, sh_coeffs,
        intersection_test, first_intersection, directional_light, frag_output
    ])

    shader_to_actor(odf_actor, "fragment", impl_code=fs_impl, block="picking")

    return odf_actor
