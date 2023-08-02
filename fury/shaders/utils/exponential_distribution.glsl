// This assumes the center of the normal distribution is the center of the screen
float kde(vec3 point, float sigma){
    return exp(-1.0*length(point)/sigma);
}