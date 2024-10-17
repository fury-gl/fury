bool newtonBisection(out float outRoot, out float outEndValue,
    float poly[MAX_DEGREE + 1], float begin, float end,
    float beginValue, float errorTolerance, int maxPolyDegree)
{
    if (begin == end) {
        outEndValue = beginValue;
        return false;
    }
    // Evaluate the polynomial at the end of the interval
    outEndValue = poly[maxPolyDegree];
    _unroll_
    for (int i = maxPolyDegree - 1; i != -1; --i)
        outEndValue = outEndValue * end + poly[i];
    // If the values at both ends have the same non-zero sign, there is no root
    if (beginValue * outEndValue > 0.0)
        return false;
    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);
    _loop_
    for (int i = 0; i != 90; ++i) {
        // Evaluate the polynomial and its derivative
        float value = poly[maxPolyDegree] * current + poly[maxPolyDegree - 1];
        float derivative = poly[maxPolyDegree];
        _unroll_
        for (int j = maxPolyDegree - 2; j != -1; --j) {
            derivative = derivative * current + value;
            value = value * current + poly[j];
        }
        // Shorten the interval
        bool right = beginValue * value > 0.0;
        begin = right ? current : begin;
        end = right ? end : current;
        // Apply Newton's method
        float guess = current - value / derivative;
        // Pick a guess
        float middle = 0.5 * (begin + end);
        float next = (guess >= begin && guess <= end) ? guess : middle;
        // Move along or terminate
        bool done = abs(next - current) < errorTolerance;
        current = next;
        if (done)
            break;
    }
    outRoot = current;
    return true;
}
