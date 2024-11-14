from fury.v2.window import display, record
from fury.v2.actor import sphere

sphere_actor0 = sphere(15, color=(1, 0, 0, 1), position=(15, 0, 0))
sphere_actor1 = sphere(15, color=(0, 1, 0, 1), position=(0, 15, 0))
sphere_actor2 = sphere(15, color=(0, 0, 1, 1), position=(0, 0, 15))

interactive = False

if __name__ == "__main__":
    if interactive:
        display(actors=(sphere_actor0, sphere_actor1, sphere_actor2))
    else:
        record(
            actors=(sphere_actor0, sphere_actor1, sphere_actor2), fname="display.png"
        )
