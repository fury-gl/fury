// camera right
vec3 uu = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
//  camera up
vec3 vv = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
// camera direction
vec3 ww = vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]);

// create view ray
vec3 rd = normalize( point.x*-uu + point.y*-vv + 7*ww);

float len = length(point);
float radius = 1.;
if(len > radius){
    discard;
}

vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
vec3 direction = normalize(vec3(1., 1., 1.));
float ddf = max(0, dot(direction, normalizedPoint));
float ssf = pow(ddf, 24);
fragOutput0 = vec4(max(color*0.5+ddf * color, ssf * vec3(1)), 1);
