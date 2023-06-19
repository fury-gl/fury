import os

import numpy as np

from fury import actor
from fury.shaders import (attribute_to_actor, import_fury_shader,
                          shader_to_actor, compose_shader)


def tensor_ellipsoid(centers, axes, lengths, colors, scales, opacity):
    """
    Visualize one or many Tensor Ellipsoids with different features.

    Parameters
    ----------
    centers : ndarray(N, 3)
        Tensor ellipsoid positions
    axes : ndarray (3, 3) or (N, 3, 3)
        Axes of the tensor ellipsoid
    lengths : ndarray (3, ) or (N, 3)
        Axes lengths
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : float or ndarray (N, ), optional
        Tensor ellipsoid size, default(1)
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If a value is given, each dot will have the same opacity otherwise
        opacity is set to 1 by default, or is defined by Alpha parameter
        in colors if given.

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

    # Start of shader implementation

    vs_dec = \
        """
        in vec3 center;
        in float scale;
        in vec3 evals;
        in vec3 evec1;
        in vec3 evec2;
        in vec3 evec3;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out float scaleVSOutput;
        out vec3 evalsVSOutput;
        out mat3 tensorMatrix;
        """

    # Variables assignment
    v_assign = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        """

    # Normalization
    n_evals = "evalsVSOutput = evals/(max(evals.x, max(evals.y, evals.z)));"
        
    # Values constraint to avoid incorrect visualizations
    evals = "evalsVSOutput = clamp(evalsVSOutput,0.05,1);"
        
    # Scaling matrix
    sc_matrix = \
        """
        mat3 S = mat3(1/evalsVSOutput.x, 0.0, 0.0,
                      0.0, 1/evalsVSOutput.y, 0.0,
                      0.0, 0.0, 1/evalsVSOutput.z);
        """

    # Rotation matrix
    rot_matrix = "mat3 R = mat3(evec1, evec2, evec3);"
        
    # Tensor matrix
    t_matrix = "tensorMatrix = inverse(R) * S * R;"

    vs_impl = compose_shader([v_assign, n_evals, evals, sc_matrix, rot_matrix,
                              t_matrix])

    # Adding shader implementation to actor
    shader_to_actor(box_actor, 'vertex', decl_code=vs_dec, impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;
        in vec3 evalsVSOutput;
        in mat3 tensorMatrix;

        uniform mat4 MCVCMatrix;
        """

    # Importing the sphere SDF
    sd_sphere = import_fury_shader(os.path.join('sdf', 'sd_sphere.frag'))

    # SDF definition
    sdf_map = \
        """
        float map(in vec3 position)
        {
            /*
            As the scaling is not a rigid body transformation, we multiply
            by a factor to compensate for distortion and not overestimate
            the distance.
            */
            float scFactor = min(evalsVSOutput.x, min(evalsVSOutput.y,
                                     evalsVSOutput.z));
                                     
            /*
            The approximation of distance is calculated by stretching the
            space such that the ellipsoid becomes a sphere (multiplying by
            the transformation matrix) and then computing the distance to
            a sphere in that space (using the sphere SDF).
            */
            return sdSphere(tensorMatrix * (position - centerMCVSOutput),
                scaleVSOutput*0.48) * scFactor;
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
    fs_dec = compose_shader([fs_vars_dec, sd_sphere, sdf_map,
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
