from fury import actor, window, primitive, utils
import vtk
from vtk.util import numpy_support
import numpy as np


#Create  a Scene
scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
#import a box primitive
vertices, faces = primitive.prim_box()
#create a actor
box_actor = utils.get_actor_from_primitive(vertices, faces)
center = ([5, 0, 0])

box_actor.SetPosition(center)

#create a Mapper
mapper = box_actor.GetMapper()
#set attribute for shader
vtk_center = numpy_support.numpy_to_vtk(center)
vtk_center.SetName("center")
box_actor.GetMapper().GetInput().GetPointData().AddArray(vtk_center)

mapper.AddShaderReplacement(
	vtk.vtkShader.Vertex,
	"//VTK::ValuePass::Dec",
	True,
	"""
	//VTK::ValuePass::Dec
	in vec3 center;
	out vec3 centeredVertexMC;

	""",
	False
)

mapper.AddShaderReplacement(
	vtk.vtkShader.Vertex,
	"//VTK::ValuePass::Impl",
	True,
	"""
	//VTK::ValuePass::Impl
	centeredVertexMC = vertexMC.xyz - center;
	
	""",
	False
)

mapper.AddShaderReplacement(
	vtk.vtkShader.Fragment,
	"//VTK::ValuePass::Dec",
	True,
	"""
	//VTK::ValuePass::Dec
	in vec3 centeredVertexMC;

	uniform mat4 MCVCMatrix;

    float sdTorus(vec3 p, vec2 t)
    {
    	vec2 q = vec2(length(p.xz) - t.x, p.y);
    	return length(q) - t.y;
    }

    float map( in vec3 position)
    {
    	float d1 = sdTorus(position, vec2(0.4, 0.1));
    	return d1;
    }

    vec3 calculateNormal( in vec3 position )
    {
    	vec2 e = vec2(0.001, 0.0);
    	return normalize( vec3( map(position + e.xyy) - map(position - e.xyy),
    							map(position + e.yxy) - map(position - e.yxy),
    							map(position + e.yyx) - map(position - e.yyx)
    						)
    					);

    }

    float castRay(in vec3 ro, vec3 rd)
    {
    	float t = 0.0;
    	for(int i=0; i<400; i++){

    		vec3 position = ro + t * rd;
    		vec3 norm = calculateNormal(position);

    		float  h = map(position);
    		if(h<0.001) break;

    		t += h;
    		if ( t > 20.0) break;
    	}
    	return t;
    }

	""",
	False
)

mapper.AddShaderReplacement(
	vtk.vtkShader.Fragment,
	"//VTK::Light::Impl",
	True,
	"""
	//VTK::Light::Impl

	vec3 point  = centeredVertexMC;
	
	vec3 uu = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]); // camera right
    vec3 vv = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]); //  camera up
    vec3 ww = vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]); // camera direction
    
    //ray origin
    vec4 ro = -MCVCMatrix[3] * MCVCMatrix;  // camera position in world space

    //ray direction
    vec3 rd = normalize(point - ro.xyz);

    float t = castRay(ro.xyz, rd);
    
    if(t < 20.0)
    {
    	vec3 pos = ro.xyz + t * rd;
    	vec3 norm = calculateNormal(pos);
    	fragOutput0 = vec4(norm, 1.0);

    }
    else{
    	fragOutput0 = vec4(0, 0, 0, 0.3);
    }



	""",
	False
)



scene.add(box_actor)
scene.add(actor.axes())
window.show(scene, size=(1920, 1200))

