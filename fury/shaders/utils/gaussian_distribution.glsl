// This assumes the center of the normal distribution is the center of the screen
#define PI 3.1415926
float kde(vec3 point, float sigma){
    float norm = (1/(sigma*sqrt(2.0*PI)));
    return norm*exp(-1.0*pow(length(point), 2.0)/(2.0*sigma*sigma) );
}


// This requires a center to be passed
// float kde(vec3 point, vec3 center, float sigma){
//     return (1/(sigma*sqrt(2.0*PI)))*exp(-1.0*pow(length(center - point), 2.0)/(2.0*sigma*sigma) );
// }