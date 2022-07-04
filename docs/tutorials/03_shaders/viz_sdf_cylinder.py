"""
===============================================================================
Make a Cylinder using polygons vs SDF
===============================================================================
This tutorial is intended to show two ways of primitives creation with the use
of polygons, and SDFs. We will use cylinders as an example since they have a
simpler polygonal representation. Hence, it allows us to see better the
difference between using one or the other method.

For the cylinder representation with polygons, we will use cylinder actor
implementation on FURY, and for the visualization using SDFs, we will
implement shader code to create the cylinder and use a box actor to put our
implementation inside.

We start by importing the necessary modules:
"""

from fury import actor, window
from fury.shaders import compose_shader, shader_to_actor, attribute_to_actor

import numpy as np

###############################################################################
# Cylinder using polygons
# ================
# Polygons-based modeling, use smaller components namely triangles or polygons
# to represent 3D objects. Each polygon is defined by the position of its
# vertices and its connecting edges. In order to get a better representation
# of an object, it may be necessary to increase the number of polygons in the
# model, which is translated into the use of more space to store data and more
# rendering time to display the object.
#
# Now we define some properties of our actors, use them to create a set of
# cylinders, and add them to the scene.

centers = np.array([[-3, 0, -2], [-3, -3, -2], [0, 0, 2], [0, -3, 2],
                    [3, 0, 0], [3, -3, 0]])
dirs = np.array([[0, 1, 1], [0, 1, 1], [0, .5, .8], [0, .5, 1], [0, .8, .2],
                 [0, .3, 1]])
colors = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0],
                   [1, 0, 1]])

###############################################################################
# In order to see how cylinders are made, we set different resolutions (number
# of sides used to define the bases of the cylinder) to see how it changes the
# surface of the primitive.

cylinders_8 = actor.cylinder(centers[:2], dirs[:2], colors[:2], radius=.4,
                             heights=1.5, capped=True, resolution=8)
cylinders_16 = actor.cylinder(centers[2:4], dirs[2:4], colors[2:4], radius=.4,
                              heights=1.5, capped=True, resolution=16)
cylinders_32 = actor.cylinder(centers[4:6], dirs[4:6], colors[4:6], radius=.4,
                              heights=1.5, capped=True, resolution=32)

###############################################################################
# Next, we set up a new scene to add and visualize the actors created.

scene = window.Scene()

scene.add(cylinders_8)
scene.add(cylinders_16)
scene.add(cylinders_32)

interactive = True

if interactive:
    window.show(scene)

window.record(scene, size=(600, 600), out_path='viz_poly_cylinder.png')

###############################################################################
# Visualize the surface geometry representation for the object.

cylinders_8.GetProperty().SetRepresentationToWireframe()
cylinders_16.GetProperty().SetRepresentationToWireframe()
cylinders_32.GetProperty().SetRepresentationToWireframe()

if interactive:
    window.show(scene)

window.record(scene, size=(600, 600), out_path='viz_poly_cylinder_geom.png')

scene.clear()

###############################################################################
# Cylinder using SDF
# ================
# Signed Distance Functions (SDFs) are mathematical functions that determines
# the distance from a point in space to a surface. We will use the ray marching
# algorithm to render the SDF primitive using shaders. Raymarching is a
# technique where you step along a ray in order to find intersections with
# solid geometry. Objects in the scene are defined by SDF, and because we
# don’t use polygonal meshes it is possible to define perfectly smooth
# surfaces and allows a faster rendering in comparison to polygon-based
# modeling.

###############################################################################
# Now we create cylinders using box actor and SDF implementation on shaders.
# For this, we first create a box actor and associate the centers and
# directions data to each of the 8 vertices that make up the box.

box_actor = actor.box(centers=centers, directions=dirs, colors=colors,
                      scales=2)

box_actor.GetProperty().SetRepresentationToWireframe()
scene.add(box_actor)

if interactive:
    window.show(scene)

window.record(scene, size=(600, 600), out_path='viz_sdf_cylinder_box.png')

scene.clear()

rep_directions = np.repeat(dirs, 8, axis=0)
rep_centers = np.repeat(centers, 8, axis=0)

attribute_to_actor(box_actor, rep_centers, 'center')
attribute_to_actor(box_actor, rep_directions, 'direction')

###############################################################################
# We split the shader code into 4 different variables which correspond to
# vertex and fragment shader declaration and implementation.
#
# Vertex shaders perform basic processing of each individual vertex.

sdf_cylinder_vert_dec = \
    '''
    
    /* SDF vertex shader declaration */
    
    //VTK::ValuePass::Dec
    
    in vec3 center;
    in vec3 direction;
    
    out vec4 vertexMCVSOutput;
    out vec3 centerWCVSOutput;
    out vec3 directionVSOutput;
    
    '''

sdf_cylinder_vert_impl = \
    '''
    
    /* SDF vertex shader implementation */
    
    //VTK::ValuePass::Impl
    
    vertexMCVSOutput = vertexMC;
    centerWCVSOutput = center;
    directionVSOutput = direction;
    
    '''

###############################################################################
# Fragment shaders run on each of the pixels that the object occupies on the
# screen.  For each sample of the pixels covered by a primitive, a ‘fragment’
# is generated.

sdf_cylinder_frag_dec = \
    '''
    /* SDF fragment shader declaration */
    
    //VTK::ValuePass::Dec
    in vec4 vertexMCVSOutput;
    
    in vec3 centerWCVSOutput;
    in vec3 directionVSOutput;
    
    uniform mat4 MCVCMatrix;
    
    
    // A rotation matrix is used to transform our position vectors
    // around a given 3D axis
    mat4 rotationAxisAngle( vec3 v, float angle )
    {
        float s = sin(angle);
        float c = cos(angle);
        float ic = 1.0 - c;
    
        return mat4( v.x*v.x*ic + c,     v.y*v.x*ic - s*v.z, v.z*v.x*ic + s*v.y, 0.0,
                     v.x*v.y*ic + s*v.z, v.y*v.y*ic + c,     v.z*v.y*ic - s*v.x, 0.0,
                     v.x*v.z*ic - s*v.y, v.y*v.z*ic + s*v.x, v.z*v.z*ic + c,     0.0,
                     0.0,                0.0,                0.0,                1.0 );
    }
    
    
    // SDF for the cylinder
    float sdCylinder( vec3 p, float h, float r )
    {
        vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
        return min(max(d.x,d.y),0.0) + length(max(d,0.0));
    }
    
    
    // This is used on calculations for surface normals of the cylinder
    float map( in vec3 position )
    {
        mat4 rot = rotationAxisAngle( normalize(directionVSOutput), 90);
    
        // this allows us to accommodate more than one object in the world space
        vec3 pos = (rot*vec4(position - centerWCVSOutput, 0.0)).xyz;
    
        // distance to the cylinder
        return sdCylinder(pos, 0.6, 0.8);
    }
    
    
    // We need surface normals when doing lighting of the scene
    vec3 calculateNormal( in vec3 position )
    {
        vec2 e = vec2(0.001, 0.0);
        return normalize( vec3( map(position + e.xyy) - map(position - e.xyy),
                                map(position + e.yxy) - map(position - e.yxy),
                                map(position + e.yyx) - map(position - e.yyx)));
    }
    
    
    // Ray Marching
    float castRay( in vec3 ro, vec3 rd )
    {
        float t = 0.0;
        for(int i=0; i < 4000; i++){
            vec3 position = ro + t * rd;
            float  h = map(position);
            t += h;
    
            if ( t > 20.0 || h < 0.001) break;
        }
        return t;
    }
    '''

sdf_cylinder_frag_impl = \
    '''
    /* SDF fragment shader implementation */
    
    //VKT::Light::Impl
    
    vec3 point = vertexMCVSOutput.xyz;
    
    // ray origin
    vec4 ro = -MCVCMatrix[3] * MCVCMatrix;  // camera position in world space
    
    vec3 col = vertexColorVSOutput.rgb;
    
    // ray direction
    vec3 rd = normalize(point - ro.xyz);
    
    ro += vec4((point - ro.xyz),0.0);
    
    // light direction
    vec3 ld = vec3(1.0, 1.0, 1.0);
    
    float t = castRay(ro.xyz, rd);
    
    if(t < 20.0)
    {
        vec3 position = ro.xyz + t * rd;
        vec3 norm = calculateNormal(position);
        float light = dot(ld, norm);
    
        fragOutput0 = vec4(col * light, 1.0);
    }
    else{
        discard;
    }
    '''

###############################################################################
# Finally, we add shader code implementation to the box_actor. We use
# shader_to_actor to apply our implementation to the shader creation process,
# this function joins our code to the shader template that FURY has by default.

shader_to_actor(box_actor, "vertex", impl_code=sdf_cylinder_vert_impl,
                decl_code=sdf_cylinder_vert_dec)
shader_to_actor(box_actor, "fragment", decl_code=sdf_cylinder_frag_dec)
shader_to_actor(box_actor, "fragment", impl_code=sdf_cylinder_frag_impl,
                block="light")

box_actor.GetProperty().SetRepresentationToSurface()
scene.add(box_actor)

if interactive:
    window.show(scene)

window.record(scene, size=(600, 600), out_path='viz_sdf_cylinder.png')
