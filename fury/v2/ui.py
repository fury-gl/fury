from abc import abstractmethod
import pygfx as gfx


class UI:

    def __init__(self):
        self.mouse_state = 'released'
        self.on_mouse_button_pressed = lambda event: None
        self.on_mouse_button_dragged = lambda event: None
        self.on_mouse_button_released = lambda event: None

    def add_to_scene(self, scene):
        scene.add(*self._get_objects())

    @abstractmethod
    def _get_objects(self):
        pass

    def register_events(self, obj):
        obj.add_event_handler(
            self.pointer_down, 'pointer_down'
        )
        obj.add_event_handler(
            self.pointer_move, 'pointer_move'
        )
        obj.add_event_handler(
            self.pointer_up, 'pointer_up'
        )
        print(obj._event_handlers)

    def pointer_down(self, event: gfx.Event):
        print("I clicked")
        self.mouse_state = 'pressing'
        self.on_mouse_button_pressed(event)
        event.cancel()

    def pointer_move(self, event):
        pressing_or_dragging = (
            self.mouse_state == "pressing" or self.mouse_state == "dragging"
        )

        if pressing_or_dragging:
            self.left_button_state = "dragging"
            self.on_mouse_button_dragged(event)

    def pointer_up(self, event):
        print("I am not up")
        self.mouse_state = 'released'
        self.on_mouse_button_released(event)


class Panel2D(UI):

    def __init__(
            self,
            size,
            color=(255, 255, 0, 0.6),
            position=(0, 0, 0)
    ):
        super().__init__()
        geo = gfx.plane_geometry(size[0], size[1])
        mat = gfx.MeshPhongMaterial(color=color, pick_write=True)
        self.obj = gfx.Mesh(geo, mat)
        self.register_events(self.obj)
        # self.obj.add_event_handler(self._clicked, 'pointer_down')
        # self._group = gfx.Group()
        # self._group.material.pick_write = True
        # self._group.add(obj)
        self.obj.local.position = position

        self._drag_offset = (0, 0)
        self.on_mouse_button_pressed = self._clicked
        self.on_mouse_button_dragged = self._dragged

        self.size = size

    def _get_objects(self):
        return [self.obj]

    def _dragged(self, event):
        new_position = (event.x - self._drag_offset[0],
                        self._drag_offset[1] - event.y, 0)
        self.obj.local.position = new_position

    def _clicked(self, event):
        print("clicked (", event.x, ",", event.y, ")")
        off_x = event.x - self.obj.local.position[0]
        off_y = event.y - self.obj.local.position[1]

        self._drag_offset = (off_x, off_y)
