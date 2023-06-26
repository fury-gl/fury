import os

from dipy.core.gradients import gradient_table
from dipy.reconst import dti

import dipy.denoise.noise_estimate as ne

import numpy as np

from fury import actor
from fury.shaders import (attribute_to_actor, import_fury_shader,
                          shader_to_actor, compose_shader)


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
        out vec2 angleVSOutput;
        """

    # Variables assignment
    v_assign = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        """

    # Rotation matrix
    rot_matrix = \
        """
        mat3 R = mat3(normalize(evec1), normalize(evec2), normalize(evec3));
        float a = radians(90);
        mat3 rot = mat3(cos(a),-sin(a),0,
                        sin(a),cos(a), 0, 
                        0,     0,      1);
        rotationMatrix = transpose(R) * rot;
        float ang = clamp(angle, 0, 6.283);
        angleVSOutput = vec2(sin(ang), cos(ang));
        """

    vs_impl = compose_shader([v_assign, rot_matrix])

    # Adding shader implementation to actor
    shader_to_actor(box_actor, 'vertex', decl_code=vs_dec, impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;
        in vec3 evalsVSOutput;
        in mat3 rotationMatrix;
        in vec2 angleVSOutput;

        uniform mat4 MCVCMatrix;
        """

    # Importing the cone SDF
    sd_cone = import_fury_shader(os.path.join('sdf', 'sd_cone.frag'))

    # Importing the union SDF operation
    sd_union = import_fury_shader(os.path.join('sdf', 'sd_union.frag'))

    # SDF definition
    sdf_map = \
        """
            float sdDoubleCone( vec3 p, vec2 c, float h )
            {
                return opUnion(sdCone(p,c,h),sdCone(-p,c,h));
            }

            float map(in vec3 position)
            {
                vec3 p = (position - centerMCVSOutput) /
                                     scaleVSOutput * rotationMatrix;
                vec2 a = angleVSOutput;
                float h = .5 * a.y;
                //return opUnion(sdCone(p,a,h),sdCone(-p,a,h)) * scaleVSOutput;
                return sdDoubleCone((position - centerMCVSOutput)/scaleVSOutput
                    *rotationMatrix, a, h) * scaleVSOutput;
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


