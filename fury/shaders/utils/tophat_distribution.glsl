// This assumes the center of the normal distribution is the center of the screen
float kde(vec3 point, float sigma){
    float norm = (1.0/sigma*2.0);
    return norm*int(length(point) < sigma);
}


// This requires a center to be passed
// float kde(vec3 point, vec3 center, float sigma){
//     return 1.0*int(length(center - point) < sigma);
// }