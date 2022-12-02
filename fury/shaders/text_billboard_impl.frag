vec4 rgba = vec4(color, vtkOpacity);

if (texture(charactersTexture, UV).r < 1.0) discard;

fragOutput0 = rgba;