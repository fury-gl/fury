// size of the halo around the glyph
float alpha = smoothstep(threshold - borderWidth, threshold + borderWidth, dist);

rgba = vec4(rgba.rgb, alpha * rgba.a);


fragOutput0 = rgba;
  