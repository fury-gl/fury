from fury.v2.window import ShowManager
from fury.v2.ui import Panel2D
from fury.v2.actor import sphere
import pygfx as gfx


show_m = ShowManager()

s = sphere(15, color=(1, 0, 1, 1), position=(25, 25, 25))
show_m.scene.add(s)
# show_m.scene.add(cube1)

panel = Panel2D((200, 50))
panel.add_to_scene(show_m.scene)
panel.register_events(panel.obj)


def clicked(event):
    print(show_m.scene.get_bounding_box())
    print("I am clicked on external")


# panel.obj.add_event_handler(clicked, 'pointer_down')
show_m.scene.add(panel.obj)
panel.obj.handle_event(gfx.PointerEvent(x=10, y=10, type="pointer_down"))


# geo = gfx.plane_geometry(200, 50)
# mat = gfx.MeshPhongMaterial(color=(255, 255, 0, 1), pick_write=True)
# obj = gfx.Mesh(geo, mat)
# obj.add_event_handler(clicked, 'pointer_down')
# show_m.scene.add(obj)


if __name__ == "__main__":
    show_m.render()
    show_m.start()
