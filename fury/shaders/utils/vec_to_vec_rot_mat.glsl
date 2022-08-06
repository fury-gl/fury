mat4 vec2VecRotMat(vec3 u, vec3 v)
{
    /*
    Given the 3D unit vectors u, v, return the 4x4 rotation matrix R
    that aligns u with v.
    */

    // Cross product is the first step to find R
    vec3 w = cross(u, v);
    float wn = length(w);

    // Check that cross product is OK and vectors u, v are not collinear
    // (norm(w)>0.0)
    if(isnan(wn) || wn < 0.0)
    {
        float normUV = length(u - v);
        // This is the case of two antipodal vectors:
        // ** former checking assumed norm(u) == norm(v)
        if(normUV > length(u))
            return mat4(-1);
        return mat4(1);
    }

    // if everything ok, normalize w
    w = w / wn;

    // vp is in plane of u,v,  perpendicular to u
    vec3 vp = (v - dot(u, v) * u);
    vp = vp / length(vp);

    // (u vp w) is an orthonormal basis
    mat3 Pt = mat3(u, vp, w);
    mat3 P = transpose(Pt);

    float cosa = clamp(dot(u, v), -1, 1);
    float sina = sqrt(1 - pow(cosa, 2));

    mat3 R = mat3(mat2(cosa, sina, -sina, cosa));
    mat3 Rp = Pt * (R * P);

    // make sure that you don't return any Nans
    bool anyNanCheckRp0 = any(isnan(Rp[0]));
    bool anyNanCheckRp1 = any(isnan(Rp[1]));
    bool anyNanCheckRp2 = any(isnan(Rp[2]));
    if(anyNanCheckRp0 || anyNanCheckRp1 || anyNanCheckRp2)
        return mat4(1);
    return mat4(Rp);
}
