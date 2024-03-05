// This assumes the center of the normal distribution is the center of the screen
#define E 2.7182818
float kde(vec3 point, float sigma){
    return exp(-1.0*length(point)/sigma);
}


// This requires a center to be passed
// float kde(vec3 point, vec3 center, float sigma){
//     return exp(-1.0*length(center - point)/sigma);
// }