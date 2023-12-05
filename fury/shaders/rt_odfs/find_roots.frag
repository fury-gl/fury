void find_roots(out float out_roots[MAX_DEGREE + 1], float poly[MAX_DEGREE + 1], float begin, float end) {
    float tolerance = (end - begin) * 1.0e-4;
    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float derivative[MAX_DEGREE + 1];
    derivative[0] = poly[MAX_DEGREE - 2];
    derivative[1] = float(MAX_DEGREE - 1) * poly[MAX_DEGREE - 1];
    derivative[2] = (0.5 * float((MAX_DEGREE - 1) * MAX_DEGREE)) * poly[MAX_DEGREE - 0];
    _unroll_
    for (int i = 3; i != MAX_DEGREE + 1; ++i)
        derivative[i] = 0.0;
    // Compute its two roots using the quadratic formula
    float discriminant = derivative[1] * derivative[1] - 4.0 * derivative[0] * derivative[2];
    if (discriminant >= 0.0) {
        float sqrt_discriminant = sqrt(discriminant);
        float scaled_root = derivative[1] + ((derivative[1] > 0.0) ? sqrt_discriminant : (-sqrt_discriminant));
        float root_0 = clamp(-2.0 * derivative[0] / scaled_root, begin, end);
        float root_1 = clamp(-0.5 * scaled_root / derivative[2], begin, end);
        out_roots[MAX_DEGREE - 2] = min(root_0, root_1);
        out_roots[MAX_DEGREE - 1] = max(root_0, root_1);
    }
    else {
        // Indicate that the cubic derivative has a single root
        out_roots[MAX_DEGREE - 2] = begin;
        out_roots[MAX_DEGREE - 1] = begin;
    }
    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched roots are set to begin
    out_roots[MAX_DEGREE] = end;
    _unroll_
    for (int i = 0; i != MAX_DEGREE - 2; ++i)
        out_roots[i] = begin;
    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree MAX_DEGREE and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.
    _loop_
    for (int degree = 3; degree != MAX_DEGREE + 1; ++degree) {
        // Take the integral of the previous derivative (scaled such that the
        // constant coefficient can still be copied directly from poly)
        float prev_derivative_order = float(MAX_DEGREE + 1 - degree);
        _unroll_
        for (int i = MAX_DEGREE; i != 0; --i)
            derivative[i] = derivative[i - 1] * (prev_derivative_order * (1.0 / float(i)));
        // Copy the constant coefficient without causing spilling. This part
        // would be harder if the derivative were not scaled the way it is.
        _unroll_
        for (int i = 0; i != MAX_DEGREE - 2; ++i)
            derivative[0] = (degree == MAX_DEGREE - i) ? poly[i] : derivative[0];
        // Determine the value of this derivative at begin
        float begin_value = derivative[MAX_DEGREE];
        _unroll_
        for (int i = MAX_DEGREE - 1; i != -1; --i)
            begin_value = begin_value * begin + derivative[i];
        // Iterate over the intervals where roots may be found
        _unroll_
        for (int i = 0; i != MAX_DEGREE; ++i) {
            if (i < MAX_DEGREE - degree)
                continue;
            float current_begin = out_roots[i];
            float current_end = out_roots[i + 1];
            // Try to find a root
            float root;
            if (newton_bisection(root, begin_value, derivative, current_begin, current_end, begin_value, tolerance))
                out_roots[i] = root;
            else if (degree < MAX_DEGREE)
                // Create an empty interval for the next iteration
                out_roots[i] = out_roots[i - 1];
            else
                out_roots[i] = NO_INTERSECTION;
        }
    }
    // We no longer need this array entry
    out_roots[MAX_DEGREE] = NO_INTERSECTION;
}
