// This assumes the center of the normal distribution is the center of the screen
float kde(vec3 point, float sigma){
    return int(length(point) < sigma);
}