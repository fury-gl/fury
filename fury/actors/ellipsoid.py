import os

import numpy as np

from fury import actor
from fury.lib import Actor
from fury.shaders import (attribute_to_actor, import_fury_shader,
                          shader_to_actor, compose_shader)


class EllipsoidActor(Actor):
    """
    VTK actor for visualizing Ellipsoids.

    Parameters
    ----------
    centers : ndarray(N, 3)
        Ellipsoid positions
    axes : ndarray (3, 3) or (N, 3, 3)
        Axes of the ellipsoid
    lengths : ndarray (3, ) or (N, 3)
        Axes lengths
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N, ), optional
        Ellipsoid size, default(1)
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If a value is given, each dot will have the same opacity otherwise
        opacity is set to 1 by default, or is defined by Alpha parameter
        in colors if given.

    """
    def __init__(self, centers, axes, lengths, colors, scales, opacity):
        self.centers = centers
        self.axes = axes
        self.lengths = lengths
        self.colors = colors
        self.scales = scales
        self.opacity = opacity
        self.SetMapper(actor.box(self.centers, colors=self.colors,
                                 scales=scales).GetMapper())
        self.GetMapper().SetVBOShiftScaleMethod(False)
        self.GetProperty().SetOpacity(self.opacity)

        big_centers = np.repeat(self.centers, 8, axis=0)
        attribute_to_actor(self, big_centers, 'center')

        big_scales = np.repeat(self.scales, 8, axis=0)
        attribute_to_actor(self, big_scales, 'scale')

        big_values = np.repeat(np.array(self.lengths, dtype=float), 8, axis=0)
        attribute_to_actor(self, big_values, 'evals')

        evec1 = np.array([item[0] for item in self.axes])
        evec2 = np.array([item[1] for item in self.axes])
        evec3 = np.array([item[2] for item in self.axes])

        big_vectors_1 = np.repeat(evec1, 8, axis=0)
        attribute_to_actor(self, big_vectors_1, 'evec1')
        big_vectors_2 = np.repeat(evec2, 8, axis=0)
        attribute_to_actor(self, big_vectors_2, 'evec2')
        big_vectors_3 = np.repeat(evec3, 8, axis=0)
        attribute_to_actor(self, big_vectors_3, 'evec3')

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

        vs_impl = \
            """
            vertexMCVSOutput = vertexMC;
            centerMCVSOutput = center;
            scaleVSOutput = scale;
            evalsVSOutput = evals/(max(evals.x, max(evals.y, evals.z)));
            evalsVSOutput = clamp(evalsVSOutput,0.05,1);
            mat3 T = mat3(1/evalsVSOutput.x, 0.0, 0.0,
                          0.0, 1/evalsVSOutput.y, 0.0,
                          0.0, 0.0, 1/evalsVSOutput.z);
            mat3 R = mat3(evec1, evec2, evec3);
            tensorMatrix = inverse(R) * T * R;
            """

        shader_to_actor(self, 'vertex', decl_code=vs_dec,
                        impl_code=vs_impl)

        fs_vars_dec = \
            """
            in vec4 vertexMCVSOutput;
            in vec3 centerMCVSOutput;
            in float scaleVSOutput;
            in vec3 evalsVSOutput;
            in mat3 tensorMatrix;

            uniform mat4 MCVCMatrix;
            """

        sd_sphere = import_fury_shader(os.path.join('sdf', 'sd_sphere.frag'))

        sdf_map = \
            """
            float map(in vec3 position)
            {
                return sdSphere(tensorMatrix * (position - centerMCVSOutput), 
                    scaleVSOutput*0.48) * min(evalsVSOutput.x,
                    min(evalsVSOutput.y, evalsVSOutput.z));
            }
            """

        central_diffs_normal = import_fury_shader(os.path.join(
            'sdf', 'central_diffs.frag'))

        cast_ray = import_fury_shader(os.path.join(
            'ray_marching', 'cast_ray.frag'))

        blinn_phong_model = import_fury_shader(os.path.join(
            'lighting', 'blinn_phong_model.frag'))

        fs_dec = compose_shader([fs_vars_dec, sd_sphere, sdf_map,
                                 central_diffs_normal, cast_ray,
                                 blinn_phong_model])

        shader_to_actor(self, 'fragment', decl_code=fs_dec)

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

        shader_to_actor(self, 'fragment', impl_code=sdf_frag_impl,
                        block='light')