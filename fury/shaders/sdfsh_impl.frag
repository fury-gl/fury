/* SDF fragment shader implementation */

//VKT::Light::Impl

vec3 point = vertexMCVSOutput.xyz;

//ray origin
vec4 ro = -MCVCMatrix[3] * MCVCMatrix;  // camera position in world space

vec3 col = vertexColorVSOutput.rgb;

//ray direction
vec3 rd = normalize(point - ro.xyz);


vec3 t = castRay(ro.xyz, rd);
    
if(t.y > -0.05)
{
    vec3 mater = 0.5*mix( vec3(1.0,0.6,0.15), vec3(0.2,0.4,0.5), t.y ); 	

    fragOutput0 = vec4( mater , 1.0);
    //fragOutput0 = vec4( norm, 1.0);
    	
}
else{
    fragOutput0 = vec4(0, 0, 0, 0.3);
    //discard;
}
