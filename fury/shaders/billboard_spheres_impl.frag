
float len = length(point);
float radius = 1.;
if(len > radius){
    discard;
}

float d = sqrt(1. - len);
vec3 normalizedPoint = normalize(vec3(point.xy, d));
vec3 direction = normalize(vec3(1., 1., 1.));
float ddf = max(0, dot(direction, normalizedPoint));
float ssf = pow(ddf, 24);

float far = gl_DepthRange.far;
float near = gl_DepthRange.near;

vec3 cameraZ = vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]);
vec4 positionDisplaced = vec4(centerVertexMCVSOutput.xyz
                              +cameraZ*sqrt(1.0 - len*len),1.0);
vec4 positionDisplacedDC = (MCDCMatrix*positionDisplaced);
float depth = positionDisplacedDC.z/positionDisplacedDC.w;
depth = ((far-near) * (depth) + near + far) / 2.0;
gl_FragDepth = depth;

fragOutput0 = vec4(max(color*0.5+ddf * color, ssf * vec3(1)), 1);
