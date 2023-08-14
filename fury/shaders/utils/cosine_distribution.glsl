// This assumes the center of the normal distribution is the center of the screen
float kde(vec3 point, float bandwidth){
    return cos(PI*length(point)/(2*bandwidth))*int(length(point) < bandwidth);
}