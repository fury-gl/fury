vec3 normalizedDirection = direction / length(direction);
vec3 normalizedVelocity = velocity / length(velocity);

mat3 RFollowers = vecToVecRotMat(normalizedVelocity, normalizedDirection);

vec3 vertex = relativePosition * RFollowers + center;

gl_Position = MCDCMatrix * vec4(vertex, 1);
