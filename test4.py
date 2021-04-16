from fury import window, actor
import numpy as np

def dotted_line(start_pos, end_pos, colors=np.array([[1, 0, 0]]),
                num_points=10, radius=0.01, units='dots'):
    N = start_pos.shape[0]
    arr = np.empty((num_points * N, 3))

    for start_pos, end_pos, x in zip(start_pos, end_pos, range(0, N)):
        dp = (end_pos - start_pos) / (num_points-1)
        p = start_pos.copy()
        count = 0
        while count < num_points:
            arr[x * num_points + count] = np.array([p])
            count = count + 1
            p = p + dp
    if units == 'dots':
        # c = actor.dots(points=arr, color=np.repeat(colors, num_points, axis=0), dot_size=radius)
        c = actor.dots(points=arr, color=colors, dot_size=radius)
    elif units == 'points':
        c = actor.point(points=arr, colors=np.repeat(colors, num_points, axis=0), point_radius=radius)
    return c


scene = window.Scene()

# start_pos = np.array([0, 0, 0])
# end_pos = np.array([1, 1, 1])

start_pos = np.random.rand(3, 3)
end_pos = np.random.rand(3, 3)

num_points = 10
colors = np.random.rand(3, 3)
# colors = [1, 0, 0]
radius = .005

# c = dotted_line(start_pos, end_pos, colors, num_points, radius)
c = dotted_line(start_pos, end_pos, colors, num_points, radius, units='points')
d = actor.dots(end_pos, color=[0, 1, 0])
scene.add(c)
scene.add(d)
window.show(scene)
