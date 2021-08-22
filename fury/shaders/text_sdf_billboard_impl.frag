vec4 rgba = vec4(color, .8);

// threshold parameter cames from the SDF bitmap and is used to determine
// if the pixel is inside or outside the glyph.
const float threshold = 0.5;
float dist = texture2D(charactersTexture, UV).r;

// size of the halo around the glyph
float borderWidth = 0.05;
float alpha = smoothstep(threshold - borderWidth, threshold + borderWidth, dist);

rgba = vec4(rgba.rgb, alpha * rgba.a);
  

fragOutput0 = rgba;