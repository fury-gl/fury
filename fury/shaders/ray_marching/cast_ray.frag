float castRay(in vec3 ro, vec3 rd)
{
    float t = 0.0;
    for(int i=0; i < 4000; i++)
    {
        vec3 position = ro + t * rd;
        float h = map(position);
        t += h;
        if ( t > 20.0 || h < 0.001) break;
    }
    return t;
}
