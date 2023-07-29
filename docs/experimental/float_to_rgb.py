import numpy as np
i = 255 + 255*256 + 255*256**2 + 255*256**3
i = 0.000000001
j = 0.000000001
f = int(i*256*256*256*256 - 1)
g = int(j*256*256*256*256 - 1)
print("f:", f)

def converter(n):
    f = int(n*256*256*256*256 - 1)
    c = np.zeros(4, dtype = float)
    c[0] = f % 256
    c[1] = float((f % 256**2 - c[0]) // 256)
    c[2] = float((f % 256**3 - c[1] - c[0]) // 256**2)
    c[3] = float((f % 256**4 - c[2] - c[1] - c[0]) // 256**3)

    return c/255

def de_converter(h):
    return (255*(h[0] + h[1]*256 + h[2]*256**2 + h[3]*256**3) + 1.0)/(256*256*256*256)

c = converter(i)
d = converter(j)
print(f, g)
print(i)
print(np.array(c))
de = de_converter(c + d)
print(int(de*256*256*256*256 - 1))