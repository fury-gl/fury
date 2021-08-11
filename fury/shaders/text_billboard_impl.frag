// float s = 1;
// float edgeWidth = 0.0;
// edgeWidth = edgeWidth/2.;
// float minSdf = 0.5/2.0;
// vec2 d = abs(point.xy) - vec2(s, s);
// float sdf = -length(max(d,0.0)) - min(max(d.x,d.y),0.0);

// if (sdf<0.0) discard;

vec4 rgba = vec4(  color, 1);
// if (edgeWidthNew > 0.0){
//     if (sdf < edgeWidthNew) {
//         rgba = vec4(edgeColor, edgeOpacity);
//     }
// }

    //vertexMC.xyz - center

rgba = texture(charactersTexture, UV)*rgba;
//rgba.w = 0.0;
fragOutput0 = rgba;