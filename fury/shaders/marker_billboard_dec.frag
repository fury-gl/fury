in float vMarker;
uniform float markerOpacity;
uniform float edgeOpacity;
uniform float edgeWidth;
uniform vec3 edgeColor;
uniform mat4 MCDCMatrix;
uniform mat4 MCVCMatrix;

float ndot(vec2 a, vec2 b ) {
    return a.x*b.x - a.y*b.y;
}
/* Refs for sdf functions
   https://github.com/rougier/python-opengl
   https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.html
   https://andrewhungblog.wordpress.com/2018/07/28/shader-art-tutorial-hexagonal-grids/
*/
vec3 getDistFunc(vec2 p, float s, float edgeWidth, float marker){
    float  sdf = 0;
    float minSdf = 0;

    if (marker == 0.){
        edgeWidth = edgeWidth/2.;
        minSdf = 0.5;
        sdf = -length(p) + s;
     }else if  (marker == 1.){
        edgeWidth = edgeWidth/2.;
        minSdf = 0.5/2.0;
        vec2 d = abs(p) - vec2(s, s);
        sdf = -length(max(d,0.0)) - min(max(d.x,d.y),0.0);
     }else if  (marker == 2.){
        edgeWidth = edgeWidth/4.;
        minSdf = 0.5/2.0;
        vec2 b  = vec2(s, s/2.0);
        vec2 q = abs(p);
        float h = clamp((-2.0*ndot(q,b)+ndot(b,b))/dot(b,b),-1.0,1.0);
        float d = length( q - 0.5*b*vec2(1.0-h,1.0+h) );
        sdf = -d * sign( q.x*b.y + q.y*b.x - b.x*b.y );
      }else if  (marker == 3.){
        float l = s/1.5;
        minSdf = 1000.0;
        float k = sqrt(3.0);
        p.x = abs(p.x) - l;
        p.y = p.y + l/k;
        if( p.x+k*p.y>0.0 ) p = vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
        p.x -= clamp( p.x, -2.0*l, 0.0 );
        sdf = length(p)*sign(p.y);
     }else if  (marker == 4.){
        edgeWidth = edgeWidth/4.;
        minSdf = 0.5/2.0;
        float r = s/2.0;
       /*https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.html*/
        const vec3 k = vec3(0.809016994,0.587785252,0.726542528);
        p.x = abs(p.x);
        p -= 2.0*min(dot(vec2(-k.x,k.y),p),0.0)*vec2(-k.x,k.y);
        p -= 2.0*min(dot(vec2( k.x,k.y),p),0.0)*vec2( k.x,k.y);
        p -= vec2(clamp(p.x,-r*k.z,r*k.z),r);
        sdf = -length(p)*sign(p.y);
    }else if  (marker == 5.){
        edgeWidth = edgeWidth/4.;
        minSdf = 0.5/2.0;
        float r = s/2.0;
       /*https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.html*/
        const vec3 k = vec3(-0.866025404,0.5,0.577350269);
        p = abs(p);
        p -= 2.0*min(dot(k.xy,p),0.0)*k.xy;
        p -= vec2(clamp(p.x, -k.z*r, k.z*r), r);
        sdf = -length(p)*sign(p.y);
     }else if  (marker == 6.){
        minSdf = 0.5/2.0;
        edgeWidth = edgeWidth/4.;
        float r = s/2.0;
       /*https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.html*/
        const vec4 k = vec4(-0.5,0.8660254038,0.5773502692,1.7320508076);
        p = abs(p);
        p -= 2.0*min(dot(k.xy,p),0.0)*k.xy;
        p -= 2.0*min(dot(k.yx,p),0.0)*k.yx;
        p -= vec2(clamp(p.x,r*k.z,r*k.w),r);
        sdf = -length(p)*sign(p.y);
     }else if  (marker == 7.){
        edgeWidth = edgeWidth/8.;
        minSdf = 0.5/4.0;
        float r = s/4.0;
        float w = 0.5;
        p = abs(p);
        sdf = -length(p-min(p.x+p.y,w)*0.5) + r;
      }else{
        edgeWidth = edgeWidth/4.;
        minSdf = 0.5/2.0;
        float r = s/15.0; //corner radius
        vec2 b = vec2(s/1.0, s/3.0); //base , size
        //vec2 b = vec2(r, r);
        p = abs(p); p = (p.y>p.x) ? p.yx : p.xy;
        vec2  q = p - b;
        float k = max(q.y,q.x);
        vec2  w = (k>0.0) ? q : vec2(b.y-p.x,-k);
        sdf = -sign(k)*length(max(w,0.0)) - r;
      }
    vec3 result = vec3(sdf, minSdf, edgeWidth);
    return result ;
}
