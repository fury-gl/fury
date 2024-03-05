// This assumes the center of the normal distribution is the center of the screen
float kde(vec3 point, float sigma){
    float norm = (1.0/sigma);
    return norm*(1.0 - length(point)/sigma)*int(length(point) < sigma);
}


// This requires a center to be passed
// float kde(vec3 point, vec3 center, float sigma){
//     return (1.0 - length(center - point)/sigma)*int(length(center - point) < sigma);
// }