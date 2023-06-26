import os

from dipy.core.gradients import gradient_table
from dipy.reconst import dti

import dipy.denoise.noise_estimate as ne

import numpy as np

from fury import actor
from fury.shaders import (attribute_to_actor, import_fury_shader,
                          shader_to_actor, compose_shader)


def uncertainty_cone(data, bvals, bvecs, scales, opacity):
    angles, centers, axes, lengths = main_dir_uncertainty(data, bvals, bvecs)
    colors = np.array([107, 107, 107])
    return double_cone(centers, axes, lengths, angles, colors, scales, opacity)


def double_cone(centers, axes, lengths, angles, colors, scales, opacity):
    """
    Visualize one or many Double Cones with different features.

    Parameters
    ----------
    centers : ndarray(N, 3)
    axes : ndarray (3, 3) or (N, 3, 3)
    lengths : ndarray (3, ) or (N, 3)
    angles : float or ndarray (N, )
    colors : ndarray (N,3) or tuple (3,), optional
    scales : float or ndarray (N, ), optional
    opacity : float, optional

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

    big_values = np.repeat(np.array(lengths, dtype=float), n_verts, axis=0)
    attribute_to_actor(box_actor, big_values, 'evals')

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
        in vec3 evals;
        in vec3 evec1;
        in vec3 evec2;
        in vec3 evec3;
        in float angle;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out float scaleVSOutput;
        out vec3 evalsVSOutput;
        out mat3 rotationMatrix;
        out float angleVSOutput;
        """

    vs_impl = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        mat3 R = mat3(normalize(evec1), normalize(evec2), normalize(evec3));
        float a = radians(90);
        mat3 rot = mat3(cos(a),-sin(a),0,
                        sin(a),cos(a), 0, 
                        0,     0,      1);
        rotationMatrix = transpose(R) * rot;
        angleVSOutput = angle;
        """

    shader_to_actor(box_actor, 'vertex', decl_code=vs_dec,
                    impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;
        in vec3 evalsVSOutput;
        in mat3 rotationMatrix;
        in float angleVSOutput;

        uniform mat4 MCVCMatrix;
        """

    sd_sphere = import_fury_shader(os.path.join('sdf', 'sd_sphere.frag'))

    sdf_map = \
        """
        float opUnion( float d1, float d2 ) { return min(d1,d2); }

        float sdCone( vec3 p, vec2 c, float h )
        {
            // c is the sin/cos of the angle, h is height
            // Alternatively pass q instead of (c,h),
            // which is the point at the base in 2D
            vec2 q = h*vec2(c.x/c.y,-1.0);

            vec2 w = vec2( length(p.xz), p.y );
            vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
            vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
            float k = sign( q.y );
            float d = min(dot( a, a ),dot(b, b));
            float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
            return sqrt(d)*sign(s);
        }

        float sdDoubleCone( vec3 p, vec2 c, float h )
        {
            return opUnion(sdCone(p,c,h),sdCone(-p,c,h));
        }

        float map(in vec3 position)
        {
            float a = clamp(angleVSOutput, 0, 6.283);
            //float a = angleVSOutput;
            vec2 angle = vec2(sin(a), cos(a));
            return sdDoubleCone((position - centerMCVSOutput)/scaleVSOutput
                *rotationMatrix, angle, .5*angle.y) * scaleVSOutput;

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
    fs_dec = compose_shader([fs_vars_dec, sdf_map,
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
    gtab = gradient_table(bvals, bvecs)

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    evals = tenfit.evals
    evecs = tenfit.evecs

    tensor_vals = dti.lower_triangular(tenfit.quadratic_form)
    dti_params = dti.eig_from_lo_tri(tensor_vals)

    signalOI = dti.tensor_prediction(dti_params, gtab, 1.0)
    b_matrix = dti.design_matrix(gtab)

    valid_mask = np.abs(evecs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    dofs_vecs = evecs[indices]
    dofs_vals = evals[indices]
    signal = signalOI[indices]

    # Uncertainty calculations ------------------------------------------------

    sigma = ne.estimate_sigma(data)  # standard deviation of the noise

    # Angles for cone of uncertainty ------------------------------------------

    angles = np.ones(dofs_vecs.shape[0])
    for i in range(dofs_vecs.shape[0]):
        sigma_e = np.diag(signal[i] / sigma ** 2)
        k = np.dot(np.transpose(b_matrix), sigma_e)
        sigma_ = np.dot(k, b_matrix)

        dd = np.diag(sigma_)
        delta_DD = dti.from_lower_triangular(
            np.array([dd[0], dd[3], dd[1], dd[4], dd[5], dd[2]]))

        try:
            delta_D = np.linalg.inv(delta_DD)
        except:
            delta_D = delta_DD

        D_ = dofs_vecs
        eigen_vals = dofs_vals[i]

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
            theta = 0.0872665

    centers = np.asarray(indices).T
    return angles, centers, dofs_vecs, dofs_vals
