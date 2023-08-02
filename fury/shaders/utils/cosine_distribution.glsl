// This assumes the center of the normal distribution is the center of the screen
float kde(vec3 point, float sigma){
    return cos(PI*length(point)/(2*sigma))*int(length(point) < sigma);
}