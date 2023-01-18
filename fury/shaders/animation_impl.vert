
// vertexVCVSOutput = MCVCMatrix * vertexMC;

vec3 f_pos = vec3(0., 0., 0.);
if (is_interpolatable(position_k))
    f_pos = interp(position_k, time);

vec3 f_scale = vec3(1., 1., 1.);
if (is_interpolatable(scale_k))
    f_scale = interp(scale_k, time);

if (is_interpolatable(color_k))
    vertexColorVSOutput = vec4(interp(color_k, time), 1);
else
    vertexColorVSOutput = scalarColor;

if (is_interpolatable(opacity_k))
    vertexColorVSOutput.a = interp(opacity_k, time).x;
else
    vertexColorVSOutput = scalarColor;

gl_Position = MCDCMatrix * transformation(f_pos, f_scale) * vertexMC ;