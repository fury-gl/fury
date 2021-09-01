vec4 rgba = vec4(color, vtkOpacity);


// threshold parameter cames from the SDF bitmap and is used to determine
// if the pixel is inside or outside the glyph.
const float threshold = 0.5;
float dist = texture2D(charactersTexture, UV).r;