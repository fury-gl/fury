void rayGlyphIntersections(out float outRayParams[MAX_DEGREE], float shCoeffs[SH_COUNT],
    vec3 rayOri, vec3 rayDir, int shDegree, int numCoeffs, int maxPolyDegree, float pi,
    float noIntersection)
{
    // Determine the direction from the glyph center to the closest point on
    // the ray
    float dotDirOri = dot(rayDir, rayOri);
    vec3 closestDir = normalize(rayOri - dotDirOri * rayDir);
    // Evaluate the SH polynomial at SH_DEGREE + 1 points. That is enough to
    // know its value everywhere along the ray.
    float shValues[SH_DEGREE + 1];
    _unroll_
    for (int i = 0; i != shDegree + 1; ++i) {
        vec3 point = cos(float(i) * (pi / float(shDegree + 1))) * rayDir
                   + sin(float(i) * (pi / float(shDegree + 1))) * closestDir;
        float shs[SH_COUNT];
        eval_sh(shs, point);
        shValues[i] = 0.0;
        _unroll_
        for (int j = 0; j != numCoeffs; ++j)
            shValues[i] += shCoeffs[j] * shs[j];
    }
    // Compute coefficients of the SH polynomial along the ray in the
    // coordinate frame given by rayDir and closestDir
    float radius_poly[SH_DEGREE + 1];
    float inv_vander[(SH_DEGREE + 1) * (SH_DEGREE + 1)];
    get_inv_vandermonde(inv_vander);
    _unroll_
    for (int i = 0; i != shDegree + 1; ++i) {
        radius_poly[i] = 0.0;
        _unroll_
        for (int j = 0; j != shDegree + 1; ++j)
            radius_poly[i] += inv_vander[i * (shDegree + 1) + j] * shValues[j];
    }
    // Compute a bounding circle around the glyph in the relevant plane
    float radius_max = 0.0;
    _unroll_
    for (int i = 0; i != shDegree + 1; ++i) {
        float bound = sqrt(
            pow(float(i), float(i)) * pow(float(shDegree - i), float(shDegree - i)) /
            pow(float(shDegree), float(shDegree))
        );
        // Workaround for buggy compilers where 0^0 is 0
        bound = (i == 0 || i == shDegree) ? 1.0 : bound;
        radius_max += bound * abs(radius_poly[i]);
    }
    // Figure out the interval, where (if at all) the ray intersects the circle
    float closest_dot_origin = dot(closestDir, rayOri);
    if (radius_max < abs(closest_dot_origin)) {
        _unroll_
        for (int i = 0; i != maxPolyDegree; ++i)
            outRayParams[i] = noIntersection;
        return;
    }
    float radius_over_dot = radius_max / closest_dot_origin;
    float u_max = sqrt(radius_over_dot * radius_over_dot - 1.0);
    // Take the square of radius_poly
    float poly[MAX_DEGREE + 1];
    _unroll_
    for (int i = 0; i != maxPolyDegree + 1; ++i)
        poly[i] = 0.0;
    _unroll_
    for (int i = 0; i != shDegree + 1; ++i)
        _unroll_
        for (int j = 0; j != shDegree + 1; ++j)
            poly[i + j] += radius_poly[i] * radius_poly[j];
    // Subtract the scaled (2 * SH_DEGREE + 2)-th power of the distance to the
    // glyph center
    float dot_sq = closest_dot_origin * closest_dot_origin;
    float binomial = 1.0;
    _unroll_
    for (int i = 0; i != shDegree + 2; ++i) {
        poly[2 * i] -= binomial * dot_sq;
        // Update the binomial coefficient using a recurrence relation
        binomial *= float(shDegree + 1 - i) / float(i + 1);
    }
    // Find roots of the polynomial within the relevant bounds
    float roots[MAX_DEGREE + 1];
    find_roots(roots, poly, -u_max, u_max);
    // Convert them back to the original coordinate frame (i.e. ray parameters)
    _unroll_
    for (int i = 0; i != maxPolyDegree; ++i)
        outRayParams[i] = (roots[i] != noIntersection)
                            ? (roots[i] * closest_dot_origin - dotDirOri)
                            : noIntersection;
}
