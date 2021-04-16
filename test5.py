from fury import window, actor
import numpy as np


def dashed_line(start_pos, end_pos, colors=np.array([[1, 0, 0]]),
                num_lines=10, line_fraction=0.5):
    N = start_pos.shape[0]
    arr = np.empty((num_lines * N, 2, 3))

    for start_pos, end_pos, x in zip(start_pos, end_pos, range(0, N)):
        dp = (end_pos - start_pos) / num_lines
        p = start_pos.copy()
        count = 0
        while count < num_lines:
            arr[x * num_lines + count] = \
                np.array([p,  p + (dp * line_fraction)])
            count = count + 1
            p = p + dp
    c = actor.line(arr, np.repeat(colors, num_lines, axis=0))
    return c


scene = window.Scene()

# start_pos = np.array([0, 0, 0])
start_pos = np.random.rand(3, 3)
# end_pos = np.array([1, 1, 1])
end_pos = np.random.rand(3, 3)

num_lines = 20
colors = np.random.rand(3, 3)
line_fraction = 0.5

c = dashed_line(start_pos, end_pos, colors, num_lines, line_fraction)
scene.add(c)
window.show(scene)
