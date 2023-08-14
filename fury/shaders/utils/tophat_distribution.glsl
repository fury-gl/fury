// This assumes the center of the normal distribution is the center of the screen
float kde(vec3 point, float bandwidth){
    return int(length(point) < bandwidth);
}