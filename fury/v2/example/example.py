from fury.v2.window import ShowManager
from fury.v2.actor import sphere, points
# import pygfx as gfx

show_m = ShowManager()

# cube = gfx.Mesh(
#     gfx.box_geometry(100, 100, 100),
#     gfx.MeshPhongMaterial(color="red", pick_write=True),
# )

s = sphere(15, color=(1, 0, 1, 1), position=(100, 100, 100))

point_cloud = points(3,
                     point_positions=[(5, -5, 5), (-5, 5, 5), (5, 5, -5)],
                     colors=[(1, 1, 1, 1), (1, 1, 0, 1), (1, 0, 0, 1)])

# show_m.scene.add(cube)
show_m.scene.add(s)
show_m.scene.add(point_cloud)


# def screenshot(event):
#     show_m.snapshot('snapshot.png')


# cube.add_event_handler(screenshot, "click")

if __name__ == "__main__":
    show_m.render()
    show_m.start()
