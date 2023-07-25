"""
This spript includes the basic implementation of Spherical Harmonics
"""

import numpy as np
import os

from fury import actor, window
from fury.shaders import (attribute_to_actor, compose_shader,
                          import_fury_shader, shader_to_actor)


class Sphere:
    vertices = None
    faces = None


if __name__ == '__main__':
    centers = np.array([[0, 0, 0], [-10, 0, 0], [10, 0, 0]])
    vecs = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    vals = np.array([1.0, 1.0, 1.0, 1.0])

    box_sd_stg_actor = actor.box(centers=centers, directions=vecs,
                                 colors=colors, scales=1.0)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_centers, 'center')

    big_directions = np.repeat(vecs, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_directions, 'direction')

    big_scales = np.repeat(vals, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_scales, 'scale')

    vs_dec = \
        """
        in vec3 center;
        in vec3 direction;
        in float scale;
    
        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out vec3 directionVSOutput;
        out float scaleVSOutput;
        """

    vs_impl = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        directionVSOutput = direction;
        scaleVSOutput = scale;
        """

    shader_to_actor(box_sd_stg_actor, 'vertex', decl_code=vs_dec,
                    impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in vec3 directionVSOutput;
        in float scaleVSOutput;
    
        uniform mat4 MCVCMatrix;
        """

    vec_to_vec_rot_mat = import_fury_shader(os.path.join(
        'utils', 'vec_to_vec_rot_mat.glsl'))

    sd_cylinder = import_fury_shader(os.path.join('sdf', 'sd_sphere.frag'))

    sdf_map = \
        """
        float map(in vec3 position)
        {
            return sdSphere((position - centerMCVSOutput)/scaleVSOutput, .5)
                            *scaleVSOutput;
        }
        """

    central_diffs_normal = import_fury_shader(os.path.join(
        'sdf', 'central_diffs.frag'))

    cast_ray = import_fury_shader(os.path.join(
        'ray_marching', 'cast_ray.frag'))

    blinn_phong_model = import_fury_shader(os.path.join(
        'lighting', 'blinn_phong_model.frag'))

    fs_dec = compose_shader([fs_vars_dec, vec_to_vec_rot_mat, sd_cylinder,
                             sdf_map, central_diffs_normal, cast_ray,
                             blinn_phong_model])

    shader_to_actor(box_sd_stg_actor, 'fragment', decl_code=fs_dec)

    sdf_frag_impl = \
        """
        vec3 pnt = vertexMCVSOutput.xyz;
    
        // Ray Origin
        // Camera position in world space
        vec3 ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;
    
        // Ray Direction
        vec3 rd = normalize(pnt - ro);
    
        // Light Direction
        vec3 ld = normalize(ro - pnt);
    
        ro += pnt - ro;
    
        float t = castRay(ro, rd);
    
        if(t < 20)
        {
            vec3 pos = ro + t * rd;
            vec3 normal = centralDiffsNormals(pos, .0001);
            fragOutput0 = vec4(vertexColorVSOutput.xyz, 1.0);
        }
        else
        {
            discard;
        }
        """

    shader_to_actor(box_sd_stg_actor, 'fragment', impl_code=sdf_frag_impl,
                    block='light')

    # Scene setup
    scene = window.Scene()

    scene.add(box_sd_stg_actor)

    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene, reset_camera=False)