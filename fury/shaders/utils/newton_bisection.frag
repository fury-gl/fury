bool newton_bisection(out float out_root, out float out_end_value,
    float poly[MAX_DEGREE + 1], float begin, float end,
    float begin_value, float error_tolerance)
{
    if (begin == end) {
        out_end_value = begin_value;
        return false;
    }
    // Evaluate the polynomial at the end of the interval
    out_end_value = poly[MAX_DEGREE];
    _unroll_
    for (int i = MAX_DEGREE - 1; i != -1; --i)
        out_end_value = out_end_value * end + poly[i];
    // If the values at both ends have the same non-zero sign, there is no root
    if (begin_value * out_end_value > 0.0)
        return false;
    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);
    _loop_
    for (int i = 0; i != 90; ++i) {
        // Evaluate the polynomial and its derivative
        float value = poly[MAX_DEGREE] * current + poly[MAX_DEGREE - 1];
        float derivative = poly[MAX_DEGREE];
        _unroll_
        for (int j = MAX_DEGREE - 2; j != -1; --j) {
            derivative = derivative * current + value;
            value = value * current + poly[j];
        }
        // Shorten the interval
        bool right = begin_value * value > 0.0;
        begin = right ? current : begin;
        end = right ? end : current;
        // Apply Newton's method
        float guess = current - value / derivative;
        // Pick a guess
        float middle = 0.5 * (begin + end);
        float next = (guess >= begin && guess <= end) ? guess : middle;
        // Move along or terminate
        bool done = abs(next - current) < error_tolerance;
        current = next;
        if (done)
            break;
    }
    out_root = current;
    return true;
}
