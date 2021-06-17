/* SDF fragment shader declaration */

//VTK::ValuePass::Dec
in vec4 vertexMCVSOutput;

in vec3 centerWCVSOutput;
flat in int primitiveVSOutput;
in float scaleVSOutput;
in vec3 directionVSOutput;

uniform mat4 MCVCMatrix;
uniform mat4 MCWCMatrix;
uniform mat4 WCVCMatrix;
uniform mat4 WCMCMatrix;
uniform mat4 VCMCMatrix;


float sdSE( vec3 pos, vec3 radius, vec2 exponent )
{
    vec3 e = vec3(vec2(1.0)/exponent.xy, exponent.x/exponent.y);
    vec3 g = 2.0*e; 
    vec3 invr = vec3(1.0)/radius;
    vec3 p = pos;
    vec3 A = p*invr ; 
    vec3 B = pow(A*A, e.xxy); 
    float E = B.x + B.y;
    float F = pow(E, e.z);
    float P = F + B.z;

    float K = pow(P, exponent.y) - 0.0;
    return(K);
}


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


mat4 translate( float x, float y, float z )
{
    return mat4( 1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 x,   y,   z,   1.0 );
}


float sdEllipsoid( vec3 p, vec3 r )
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}


float sdSphere( vec3 p, float s )
{
    return length(p)-s;
}


float sdTorus( vec3 p, vec2 t )
{
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}


float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
	vec3 ab = vec3(b-a);
    vec3 ap = vec3(p-a);
    
    
    float t = dot(ab, ap) / dot(ab, ab);
    t = clamp(t, 0., 1.);
    
    vec3 c = a + t*ab;
    
    return length(p-c)-r;
}


float map( in vec3 position )
{

    mat4 rot = rotationAxisAngle( normalize(directionVSOutput), 90.0 );
    mat4 tra = translate( 0.0, 1.0, 0.0 );
    mat4 txi = tra * rot; 

    vec3 pos = (txi*vec4(position  - centerWCVSOutput, 0.0)).xyz;
	
    float d1;
	
    if(primitiveVSOutput==1){
		d1 = sdSphere((pos)/scaleVSOutput, 0.25)*scaleVSOutput;
    }
    
    else if(primitiveVSOutput==2){
    	d1 = sdTorus((pos)/scaleVSOutput, vec2(0.4, 0.1))*scaleVSOutput;
    }
    
    else if(primitiveVSOutput==3){
        d1 = sdEllipsoid((pos)/scaleVSOutput, vec3(0.1, 0.1, 0.3))*scaleVSOutput;
    }
    
    else if(primitiveVSOutput==4){
        d1 = sdCapsule((pos)/scaleVSOutput, vec3(0.0, -0.3, 0.2), vec3(0.0, -0.1, 0.2), 0.2)*scaleVSOutput;
    }
    
    else if(primitiveVSOutput==5){
        d1 = sdSE((pos)/scaleVSOutput, vec3(50.0 / 3.0, 50.0 / 3.0, 50.0 / 2.0), vec2(2.0))*scaleVSOutput;
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
