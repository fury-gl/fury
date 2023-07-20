vec3 color_mapping(float intensity, sampler2D colormapTexture){
    return texture(colormapTexture, vec2(intensity,0)).rgb;
}