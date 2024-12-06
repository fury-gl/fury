void findRoots(out float outRoots[MAX_DEGREE + 1], float poly[MAX_DEGREE + 1], float begin, float end, int maxPolyDegree, float noIntersection) {
    float tolerance = (end - begin) * 1.0e-4;
    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float derivative[MAX_DEGREE + 1];
    derivative[0] = poly[maxPolyDegree - 2];
    derivative[1] = float(maxPolyDegree - 1) * poly[maxPolyDegree - 1];
    derivative[2] = (0.5 * float((maxPolyDegree - 1) * maxPolyDegree)) * poly[maxPolyDegree - 0];
    _unroll_
    for (int i = 3; i != maxPolyDegree + 1; ++i)
        derivative[i] = 0.0;
    // Compute its two roots using the quadratic formula
    float discriminant = derivative[1] * derivative[1] - 4.0 * derivative[0] * derivative[2];
    if (discriminant >= 0.0) {
        float sqrtDiscriminant = sqrt(discriminant);
        float scaledRoot = derivative[1] + ((derivative[1] > 0.0) ? sqrtDiscriminant : (-sqrtDiscriminant));
        float root0 = clamp(-2.0 * derivative[0] / scaledRoot, begin, end);
        float root1 = clamp(-0.5 * scaledRoot / derivative[2], begin, end);
        outRoots[maxPolyDegree - 2] = min(root0, root1);
        outRoots[maxPolyDegree - 1] = max(root0, root1);
    }
    else {
        // Indicate that the cubic derivative has a single root
        outRoots[maxPolyDegree - 2] = begin;
        outRoots[maxPolyDegree - 1] = begin;
    }
    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched roots are set to begin
    outRoots[maxPolyDegree] = end;
    _unroll_
    for (int i = 0; i != maxPolyDegree - 2; ++i)
        outRoots[i] = begin;
    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree MAX_DEGREE and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.
    _loop_
    for (int degree = 3; degree != maxPolyDegree + 1; ++degree) {
        // Take the integral of the previous derivative (scaled such that the
        // constant coefficient can still be copied directly from poly)
        float prevDerivativeOrder = float(maxPolyDegree + 1 - degree);
        _unroll_
        for (int i = maxPolyDegree; i != 0; --i)
            derivative[i] = derivative[i - 1] * (prevDerivativeOrder * (1.0 / float(i)));
        // Copy the constant coefficient without causing spilling. This part
        // would be harder if the derivative were not scaled the way it is.
        _unroll_
        for (int i = 0; i != maxPolyDegree - 2; ++i)
            derivative[0] = (degree == maxPolyDegree - i) ? poly[i] : derivative[0];
        // Determine the value of this derivative at begin
        float beginValue = derivative[maxPolyDegree];
        _unroll_
        for (int i = maxPolyDegree - 1; i != -1; --i)
            beginValue = beginValue * begin + derivative[i];
        // Iterate over the intervals where roots may be found
        _unroll_
        for (int i = 0; i != maxPolyDegree; ++i) {
            if (i < maxPolyDegree - degree)
                continue;
            float currentBegin = outRoots[i];
            float currentEnd = outRoots[i + 1];
            // Try to find a root
            float root;
            if (newtonBisection(root, beginValue, derivative, currentBegin, currentEnd, beginValue, tolerance, maxPolyDegree))
                outRoots[i] = root;
            else if (degree < maxPolyDegree)
                // Create an empty interval for the next iteration
                outRoots[i] = outRoots[i - 1];
            else
                outRoots[i] = noIntersection;
        }
    }
    // We no longer need this array entry
    outRoots[maxPolyDegree] = noIntersection;
}
