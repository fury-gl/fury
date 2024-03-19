def uv_calculations(n):
    uvs = []
    for i in range(0, n):
        a = (n - (i + 1)) / n
        b = (n - i) / n
        uvs.extend(
            [
                [0.001, a + 0.001],
                [0.001, b - 0.001],
                [0.999, b - 0.001],
                [0.999, a + 0.001],
                [0.001, a + 0.001],
                [0.001, b - 0.001],
                [0.999, b - 0.001],
                [0.999, a + 0.001],
            ]
        )
    return uvs