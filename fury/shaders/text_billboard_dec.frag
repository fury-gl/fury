in vec2 UV;

// vec3 getDistFunc(vec2 p, float s, float edgeWidth){
//     //square sdf func
//     edgeWidth = edgeWidth/2.;
//     float minSdf = 0.5/2.0;
//     vec2 d = abs(p) - vec2(s, s);
//     float sdf = -length(max(d,0.0)) - min(max(d.x,d.y),0.0);

//     vec3 result = vec3(sdf, minSdf, edgeWidth);
//     return result ;
// }