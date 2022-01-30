
float d = dist - threshold;
vec4 b = vec4(1.0, 1.0, 1.0, 1.0);
if (d > borderWidth/2.0)
{
    rgba = 1.0*rgba;
}
else if (d >= -borderWidth/2.0  && d <= borderWidth/2.0)
{
    //if inside shape but within range of border, lerp from border to fill colour
    float t = -d / borderWidth;
    t = t * t;
    rgba = mix(borderColor, rgba, t);
}
else if (d<-borderWidth/2.0)
{
    //if outside shape but within range of border, lerp from border to background colour
    float t = -d / borderWidth;
    t = t * t;
    
    float alpha = smoothstep(threshold - borderWidth, threshold + borderWidth, dist);
    rgba = mix(borderColor, rgba, t);
    rgba = vec4(rgba.rgb, alpha * rgba.a);

}

fragOutput0 = rgba;
  