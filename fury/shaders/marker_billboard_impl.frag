float len = length(point);
float radius = 1.;
float s = 0.5;

vec3 result = getDistFunc(point.xy, s, edgeWidth, vMarker);
float sdf = result.x;
float minSdf = result.y;
float edgeWidthNew = result.z;

if (sdf<0.0) discard;

vec4 rgba = vec4(  color, markerOpacity );
if (edgeWidthNew > 0.0){
   if (sdf < edgeWidthNew)  rgba  = vec4(edgeColor, edgeOpacity);
}

fragOutput0 = rgba;
