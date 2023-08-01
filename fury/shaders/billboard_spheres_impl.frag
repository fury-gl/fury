
/* Calculating the distance between the fragment to the center
   of the billboard */
float len = length(point);
float radius = 1.;
/* Discarding any fragment not present in a circle of unit radius */
if(len > radius){
    discard;
}

/* Calculating the 3D distance d from the center */
float d = sqrt(1. - len*len);

/* Calculating the normal as if we had a sphere of radius len*/
vec3 normalizedPoint = normalize(vec3(point.xy, d));

/* Defining a fixed light direction */
vec3 direction = normalize(vec3(1., 1., 1.));

/* Calculating diffuse */
float ddf = max(0, dot(direction, normalizedPoint));

/* Calculating specular */
float ssf = pow(ddf, 24);

/* Obtaining the two clipping planes for depth buffer */
float far = gl_DepthRange.far;
float near = gl_DepthRange.near;

/* Getting camera Z vector */
vec3 cameraZ = vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]);

/* Get the displaced position based on camera z by adding d
   in this direction */
vec4 positionDisplaced = vec4(centerVertexMCVSOutput.xyz
                              +cameraZ*d,1.0);

/* Projecting the displacement to the viewport */
vec4 positionDisplacedDC = (MCDCMatrix*positionDisplaced);

/* Applying perspective transformation to z */
float depth = positionDisplacedDC.z/positionDisplacedDC.w;

/* Interpolating the z of the displacement between far and near planes */
depth = ((far-near) * (depth) + near + far) / 2.0;

/* Writing the final depth to depth buffer */
gl_FragDepth = depth;

/* Calculating colors based on a fixed light */
fragOutput0 = vec4(max(color*0.5+ddf * color, ssf * vec3(1)), 1);
