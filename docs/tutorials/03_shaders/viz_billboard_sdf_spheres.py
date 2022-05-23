from fury import actor, window
from fury.shaders import compose_shader, import_fury_shader

import numpy as np
import os


centers = np.array([[0, 0, 0], [-6, -6, -6], [8, 8, 8], [8.5, 9.5, 9.5],
                    [10, -10, 10], [-13, 13, -13], [-17, -17, 17]])
colors = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0],
                   [0, 1, 0], [0, 1, 1]])
scales = np.array([6, 1.2, 1, .2, .7, 3, 2])

scene = window.Scene()

spheres_actor = actor.sphere(centers, colors, radii=scales, phi=8, theta=8,
                             use_primitive=False)

scene.add(spheres_actor)

interactive = True
if interactive:
    window.show(scene)

spheres_actor.GetProperty().SetRepresentationToWireframe()

if interactive:
    window.show(scene)

scene.clear()
spheres_actor = actor.sphere(centers, colors, radii=scales, phi=16, theta=16,
                             use_primitive=False)
spheres_actor.GetProperty().SetRepresentationToWireframe()
scene.add(spheres_actor)

if interactive:
    window.show(scene)

scene.clear()
billboards_actor = actor.billboard(centers, colors=colors, scales=scales)
billboards_actor.GetProperty().SetRepresentationToWireframe()
scene.add(billboards_actor)

if interactive:
    window.show(scene)

scene.clear()
sd_sphere = import_fury_shader(os.path.join('sdf', 'sd_sphere.frag'))
sphere_radius = 'float sphereRadius = 1;'
sdf_impl = \
"""
if (sdSphere(point, sphereRadius) < 0)
    fragOutput0 = vec4(color, opacity);
else
    discard;
"""
fs_impl = compose_shader([sphere_radius, sdf_impl])
spheres_actor = actor.billboard(centers, colors=colors, scales=scales,
                                fs_dec=sd_sphere, fs_impl=fs_impl)
scene.add(spheres_actor)

if interactive:
    window.show(scene)

scene.clear()
central_diffs_normal = import_fury_shader(os.path.join('sdf',
                                                       'central_diffs.frag'))
sd_sphere_normal = \
"""
float map(vec3 p)
{
    return sdSphere(p, 1);
}
"""
fs_dec = compose_shader([sd_sphere, sd_sphere_normal, central_diffs_normal])
illum_impl = \
"""
// SDF evaluation
float dist = sdSphere(point, sphereRadius);

if (dist > 0)
    discard;

// Absolute value of the distance
float absDist = abs(dist);

// Normal of a point on the surface of the sphere
vec3 normal = centralDiffsNormals(vec3(point.xy, absDist), .0001);

// Calculate the diffuse factor and diffuse color
df = max(0, normal.z);
diffuse = df * diffuseColor * lightColor0;

// Calculate the specular factor and specular color
sf = pow(df, specularPower);
specular = sf * specularColor * lightColor0;

// Using Blinn-Phong model
fragOutput0 = vec4(ambientColor + diffuse + specular, opacity);
"""
fs_impl = compose_shader([sphere_radius, illum_impl])
spheres_actor = actor.billboard(centers, colors=colors, scales=scales,
                                fs_dec=fs_dec, fs_impl=fs_impl)
scene.add(spheres_actor)

if interactive:
    window.show(scene)
