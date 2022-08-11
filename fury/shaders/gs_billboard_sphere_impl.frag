

vec4 ro = -MCVCMatrix[3] * MCVCMatrix;
vec3 rd = normalize(pointGSOutput - ro.xyz);

// light direction is the front vector of the camera.
vec3 lightDir = normalize(vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]));
//ro += vec4((point - ro.xyz),0.0);
float d = sphIntersect(ro.xyz, rd, centerGSOutput, scaleGSOutput);

//float fDist = length(vec2(dFdx(d), dFdy(d)));
//float alpha = (( -0.5 + d) / max(fDist, 0.0001) - 0.5);

if(d >= 0){
//if(alpha >= 0){
//    alpha = clamp(alpha, 0, 1);
    vec3 intersection = ro.xyz + rd * d;
    vec3 normal = normalize(intersection - centerGSOutput);
    vec3 specularColor = specularIntensity * specularColorUniform;
    float specularPower = specularPowerUniform;
    vec3 ambientColor = ambientIntensity * vertexColorGSOutput.rgb;
    vec3 diffuseColor = diffuseIntensity * vertexColorGSOutput.rgb ;

    float opacity = opacityUniform * vertexColorGSOutput.a;
    float lightAttenuation =  max(dot(normal, lightDir), 0.0);
    // lightColor0, specularPowerUniform, specularColorUniform are not provided for points and lines bu VTK.
    vec3 light = vec3(1, 1, 1);
    vec3 color = blinnPhongIllumModel(
                lightAttenuation, light, diffuseColor,
                specularPower, specularColor, ambientColor);

    fragOutput0 = vec4(color, opacity);
    vec4 dep = MCDCMatrix * vec4(intersection, 1);
    gl_FragDepth = (dep.z / dep.w + 1.0) / 2.0;
    return;
}
else discard;
