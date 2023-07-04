import os

from dipy.core.gradients import gradient_table
from dipy.reconst import dti

import dipy.denoise.noise_estimate as ne

import numpy as np

from fury import actor
from fury.shaders import (attribute_to_actor, import_fury_shader,
                          shader_to_actor, compose_shader)


def uncertainty_cone(data, bvals, bvecs, scales, opacity):
    """
    Visualize the cones of uncertainty for DTI.

    Parameters
    ----------
    data : 3D or 4D ndarray
        Diffusion data.
    bvals : array, (N,) or None
        Array containing the b-values.
    bvecs : array, (N, 3) or None
        Array containing the b-vectors.
    scales : float or ndarray (N, )
        Cones of uncertainty size.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    box_actor: Actor

    """
    angles, centers, axes = main_dir_uncertainty(data, bvals, bvecs)
    colors = np.array([107, 107, 107])

    if centers.ndim != 2:
        centers = np.array([centers])
        axes = np.array([axes])
        colors = np.array([colors])

    x, y, z = axes.shape

    if not isinstance(scales, np.ndarray):
        scales = np.array(scales)
    if scales.size == 1:
        scales = np.repeat(scales, x)

    return double_cone(centers, axes, angles, colors, scales, opacity)


def double_cone(centers, axes, angles, colors, scales, opacity):
    """
    Visualize one or many Double Cones with different features.

    Parameters
    ----------
    centers : ndarray(N, 3)
        Cone positions.
    axes : ndarray (3, 3) or (N, 3, 3)
        Axes of the cone.
    angles : float or ndarray (N, )
        Angles of the cone.
    colors : ndarray (N, 3) or tuple (3,)
        R, G and B should be in the range [0, 1].
    scales : float or ndarray (N, )
        Cone size.
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    box_actor: Actor

    """

    box_actor = actor.box(centers, colors=colors, scales=scales)
    box_actor.GetMapper().SetVBOShiftScaleMethod(False)
    box_actor.GetProperty().SetOpacity(opacity)

    # Number of vertices that make up the box
    n_verts = 8

    big_centers = np.repeat(centers, n_verts, axis=0)
    attribute_to_actor(box_actor, big_centers, 'center')

    big_scales = np.repeat(scales, n_verts, axis=0)
    attribute_to_actor(box_actor, big_scales, 'scale')

    evec1 = np.array([item[0] for item in axes])
    evec2 = np.array([item[1] for item in axes])
    evec3 = np.array([item[2] for item in axes])

    big_vectors_1 = np.repeat(evec1, n_verts, axis=0)
    attribute_to_actor(box_actor, big_vectors_1, 'evec1')
    big_vectors_2 = np.repeat(evec2, n_verts, axis=0)
    attribute_to_actor(box_actor, big_vectors_2, 'evec2')
    big_vectors_3 = np.repeat(evec3, n_verts, axis=0)
    attribute_to_actor(box_actor, big_vectors_3, 'evec3')

    big_angles = np.repeat(np.array(angles, dtype=float), n_verts, axis=0)
    attribute_to_actor(box_actor, big_angles, 'angle')

    # Start of shader implementation

    vs_dec = \
        """
        in vec3 center;
        in float scale;
        in vec3 evec1;
        in vec3 evec2;
        in vec3 evec3;
        in float angle;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out float scaleVSOutput;
        out mat3 rotationMatrix;
        out float angleVSOutput;
        """

    # Variables assignment
    v_assign = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        angleVSOutput = angle;
        """

    # Rotation matrix
    rot_matrix = \
        """
        mat3 R = mat3(normalize(evec1), normalize(evec2), normalize(evec3));
        float a = radians(90);
        mat3 rot = mat3(cos(a),-sin(a), 0,
                        sin(a), cos(a), 0, 
                            0 ,      0, 1);
        rotationMatrix = transpose(R) * rot;
        """

    vs_impl = compose_shader([v_assign, rot_matrix])

    shader_to_actor(box_actor, 'vertex', decl_code=vs_dec,
                    impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;
        in mat3 rotationMatrix;
        in float angleVSOutput;

        uniform mat4 MCVCMatrix;
        """

    # Importing the cone SDF
    sd_cone = import_fury_shader(os.path.join('sdf', 'sd_cone.frag'))

    # Importing the union operation SDF
    sd_union = import_fury_shader(os.path.join('sdf', 'sd_union.frag'))

    # SDF definition
    sdf_map = \
        """
        float map(in vec3 position)
        {
            vec3 p = (position - centerMCVSOutput)/scaleVSOutput
                *rotationMatrix;
            float angle = clamp(angleVSOutput, 0, 6.283);
            vec2 a = vec2(sin(angle), cos(angle));
            float h = .5 * a.y;
            return opUnion(sdCone(p,a,h), sdCone(-p,a,h)) * scaleVSOutput;
        }
        """

    # Importing central differences function for computing surface normals
    central_diffs_normal = import_fury_shader(os.path.join(
        'sdf', 'central_diffs.frag'))

    # Importing raymarching function
    cast_ray = import_fury_shader(os.path.join(
        'ray_marching', 'cast_ray.frag'))

    # Importing Blinn-Phong model for lighting
    blinn_phong_model = import_fury_shader(os.path.join(
        'lighting', 'blinn_phong_model.frag'))

    # Full fragment shader declaration
    fs_dec = compose_shader([fs_vars_dec, sd_cone, sd_union, sdf_map,
                             central_diffs_normal, cast_ray,
                             blinn_phong_model])

    shader_to_actor(box_actor, 'fragment', decl_code=fs_dec)

    # Vertex in Model Coordinates.
    point = "vec3 point = vertexMCVSOutput.xyz;"

    # Camera position in world space
    ray_origin = "vec3 ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;"

    ray_direction = "vec3 rd = normalize(point - ro);"

    light_direction = "vec3 ld = normalize(ro - point);"

    ray_origin_update = "ro += point - ro;"

    # Total distance traversed along the ray
    distance = "float t = castRay(ro, rd);"

    # Fragment shader output definition
    # If surface is detected, color is assigned, otherwise, nothing is painted
    frag_output_def = \
        """
        if(t < 20)
        {
            vec3 pos = ro + t * rd;
            vec3 normal = centralDiffsNormals(pos, .0001);
            // Light Attenuation
            float la = dot(ld, normal);
            vec3 color = blinnPhongIllumModel(la, lightColor0, 
                diffuseColor, specularPower, specularColor, ambientColor);
            fragOutput0 = vec4(color, opacity);
        }
        else
        {
            discard;
        }
        """

    # Full fragment shader implementation
    sdf_frag_impl = compose_shader([point, ray_origin, ray_direction,
                                    light_direction, ray_origin_update,
                                    distance, frag_output_def])

    shader_to_actor(box_actor, 'fragment', impl_code=sdf_frag_impl,
                    block='light')

    return box_actor


def main_dir_uncertainty(data, bvals, bvecs):
    """
    Calculates the angle of the cone of uncertainty that represents the
    perturbation of the main eigenvector of the diffusion tensor matrix.
    Additionally, it gives information needed for the cone visualization.

    Parameters
    ----------
    data : 3D or 4D ndarray
        Diffusion data.
    bvals : array, (N,) or None
        Array containing the b-values.
    bvecs : array, (N, 3) or None
        Array containing the b-vectors.

    Returns
    -------
    angles, centers, evecs

    Notes
    -----
    The uncertainty calculation is based on first-order matrix perturbation
    analysis described in [1]_. The idea is to estimate the variance of the
    main eigenvector which corresponds to the main direction of diffusion,
    directly from estimated D and its estimated covariance matrix \Delta D (see
    [2]_, equation 4). The subtended angle, $\Delta\theta_i$, between the ith
    perturbed eigenvector of D, $\varepsilon_i+\Delta\varepsilon_i$, and the
    estimated eigenvector $\varepsilon_i$, measures the angular deviation of
    the fiber direction, $\Delta\theta_i$:

    .. math::
        \theta=\tan^{-1}(\|\Delta\varepsilon_1\|)

    Giving way to a graphical construct for displaying both the main
    eigenvector of D and its associated uncertainty, with the so-called
    "uncertainty cone".

    References
    ----------
    .. [1] Basser, P. J. (1997). Quantifying errors in fiber direction and
    diffusion tensor field maps resulting from MR noise. In 5th Scientific
    Meeting of the ISMRM (Vol. 1740).

    .. [2] Chang, L. C., Koay, C. G., Pierpaoli, C., & Basser, P. J. (2007).
    Variance of estimated DTI‐derived parameters via first‐order perturbation
    methods. Magnetic Resonance in Medicine: An Official Journal of the
    International Society for Magnetic Resonance in Medicine, 57(1), 141-149.
    """

    gtab = gradient_table(bvals, bvecs)

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    fevals = tenfit.evals
    fevecs = tenfit.evecs

    tensor_vals = dti.lower_triangular(tenfit.quadratic_form)
    dti_params = dti.eig_from_lo_tri(tensor_vals)

    fsignal = dti.tensor_prediction(dti_params, gtab, 1.0)
    b_matrix = dti.design_matrix(gtab)

    valid_mask = np.abs(fevecs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    evecs = fevecs[indices]
    evals = fevals[indices]
    signal = fsignal[indices]

    # Uncertainty calculations

    sigma = ne.estimate_sigma(data)  # standard deviation of the noise

    # Angles for cone of uncertainty
    angles = np.ones(evecs.shape[0])
    for i in range(evecs.shape[0]):
        sigma_e = np.diag(signal[i] / sigma ** 2)
        k = np.dot(np.transpose(b_matrix), sigma_e)
        sigma_ = np.dot(k, b_matrix)

        dd = np.diag(sigma_)
        delta_DD = dti.from_lower_triangular(
            np.array([dd[0], dd[3], dd[1], dd[4], dd[5], dd[2]]))

        # perturbation matrix of tensor D
        try:
            delta_D = np.linalg.inv(delta_DD)
        except:
            delta_D = delta_DD

        D_ = evecs
        eigen_vals = evals[i]

        e1, e2, e3 = np.array(D_[i, :, 0]), np.array(D_[i, :, 1]), \
                     np.array(D_[i, :, 2])
        lambda1, lambda2, lambda3 = eigen_vals[0], eigen_vals[1], eigen_vals[2]

        if (lambda1 > lambda2 and lambda1 > lambda3):
            # The perturbation of the eigenvector associated with the largest
            # eigenvalue is given by
            a = np.dot(np.outer(np.dot(e1, delta_D), np.transpose(e2)) /
                       (lambda1 - lambda2), e2)
            b = np.dot(np.outer(np.dot(e1, delta_D), np.transpose(e3)) /
                       (lambda1 - lambda3), e3)
            delta_e1 = a + b

            # The angle \theta between the perturbed principal eigenvector of D
            theta = np.arctan(np.linalg.norm(delta_e1))
            angles[i] = theta
        else:
            theta = 1.39626

    centers = np.asarray(indices).T
    return angles, centers, evecs
