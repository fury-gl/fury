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
# print(f, g)
# print(i)
# print(np.array(c))
de = de_converter(c + d)
# print(int(de*256*256*256*256 - 1))



def gaussian_kernel(n, sigma = 1.0):
    x0 = np.arange(0.0, 1.0, 1/n)
    y0 = np.arange(0.0, 1.0, 1/n)
    x, y = np.meshgrid(x0, y0)
    center = np.array([x0[n // 2], y0[n // 2]])
    mesh = np.stack((x, y), 2)
    center = np.repeat(center, x.shape[0]*y.shape[0]).reshape(x.shape[0], y.shape[0], 2)
    kernel = np.exp((-1.0*np.linalg.norm(center - mesh, axis = 2)**2)/(2*sigma**2))
    string = f"const float gauss_kernel[{x.shape[0]*y.shape[0]}] = "
    kernel = kernel/np.sum(kernel)
    flat = str(kernel.flatten()).split(" ")
    copy_flat = flat.copy()
    taken = 0
    for i in range(len(flat)):
        if flat[i] == ' ' or flat[i] == '': 
            copy_flat.pop(i - taken)
            taken += 1
    if "[" in copy_flat[0]:
        copy_flat[0] = copy_flat[0][1:]
    else:
        copy_flat.pop(0)

    if "]" in copy_flat[-1]:
        copy_flat[-1] = copy_flat[-1][:-1]
    else:
        copy_flat.pop(-1)

    if '' == copy_flat[0]:
        copy_flat.pop(0)

    if '' == copy_flat[-1]:
        copy_flat.pop(-1)
    
    # copy_flat.pop(-1)
    print(copy_flat)

    string += "{" + ", ".join(copy_flat) + "};"
    return string

print(gaussian_kernel(13, 3.0))