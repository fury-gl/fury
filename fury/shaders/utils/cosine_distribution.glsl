// This assumes the center of the normal distribution is the center of the screen
#define PI 3.1415926
float kde(vec3 point, float sigma){
    float norm = (PI/(4.0*sigma)); 
    return norm*cos(PI*length(point)/(2*sigma))*int(length(point) < sigma);
}


// This requires a center to be passed
// float kde(vec3 point, vec3 center, float sigma){
//     return cos(PI*length(center - point)/(2*sigma))*int(length(center - point) < sigma);
// }