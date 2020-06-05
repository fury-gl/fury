/* SDF fragment shader declaration */

//VTK::ValuePass::Dec
in vec3 centeredVertexMC;
uniform float prim;

uniform mat4 MCVCMatrix;
uniform mat4 MCWCMatrix;
uniform mat3 WCVCMatrix;


float sdSphere( vec3 p, float s )
{
    return length(p)-s;
}

float sdTorus(vec3 p, vec2 t)
{
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float map( in vec3 position)
{
	float d1;
		if(prim==1.0){
			d1 = sdSphere(position, 0.25);
    	}
    	else if(prim==2.0){
    	
    		d1 = sdTorus(position, vec2(0.4, 0.1));
    	}
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
    	
    	float  h = map(position);
    	if(h<0.001) break;

    	t += h;
    	if ( t > 20.0) break;
    }
    return t;
}
