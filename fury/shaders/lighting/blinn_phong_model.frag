vec3 blinnPhongIllumModel(float la, vec3 lc, vec3 dc, float sp, vec3 sc, vec3 ac)
{
    // Calculate the diffuse factor and adjust the diffuse color
    float df = max(0, la);
    dc *= df * lc;

    // Calculate the specular factor and adjust the specular color
    float sf = pow(df, sp);
    sc *= sf * lc;

    // Blinn-Phong illumination model
    return vec3(ac + dc + sc);
}
