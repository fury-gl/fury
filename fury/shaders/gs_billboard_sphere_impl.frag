

//float d = distance(point, center);

vec4 ro = -MCVCMatrix[3] * MCVCMatrix;
vec3 rd = normalize(point - ro.xyz);

// light direction is the front vector of the camera.
vec3 lightDir = normalize(vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]));
//ro += vec4((point - ro.xyz),0.0);
float d = sphIntersect(ro.xyz, rd, vec4(center, scale));

if(d!= -1){
    vec3 intersection = ro.xyz + rd * d;
    vec3 normal = normalize(intersection - center);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 ambientColor = ambientIntensity * vertexColorGSOutput.rgb;
    vec3 diffuseColor = diffuseIntensity * vertexColorGSOutput.rgb * diff ;
    float opacity = opacityUniform * vertexColorGSOutput.a;
    fragOutput0 = vec4(ambientColor + diffuseColor, opacity);
    vec4 dep = MCDCMatrix * vec4(intersection, 1);
    gl_FragDepth = (dep.z / dep.w + 1.0) / 2.0;
    return;
}
else discard;