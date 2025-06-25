"""UI core module that describe UI abstract class."""

import abc

import numpy as np

from fury.actor import disk
from fury.decorators import warn_on_args_to_kwargs
from fury.geometry import (
    create_mesh,
)
from fury.lib import (
    KeyboardEvent,
    Mesh,
    PointerEvent,
    plane_geometry,
)
from fury.material import (
    _create_mesh_material,
)
from fury.primitive import prim_disk
from fury.ui import UIContext
from fury.ui.helpers import Anchor

# from fury.interactor import CustomInteractorStyle
# from fury.io import load_image
# from fury.lib import (
#     Actor2D,
#     CellArray,
#     DiskSource,
#     FloatArray,
#     Points,
#     PolyData,
#     PolyDataMapper2D,
#     Polygon,
#     Property2D,
#     TextActor,
#     Texture,
#     TexturedActor2D,
# )
# from fury.utils import set_input


class UI(object, metaclass=abc.ABCMeta):
    """An umbrella class for all UI elements.

    While adding UI elements to the scene, we go over all the sub-elements
    that come with it and add those to the scene automatically.

    Attributes
    ----------
    position : (float, float)
        Absolute coordinates (x, y) of the lower-left corner of this
        UI component.
    center : (float, float)
        Absolute coordinates (x, y) of the center of this UI component.
    size : (int, int)
        Width and height in pixels of this UI component.
    on_left_mouse_button_pressed: function
        Callback function for when the left mouse button is pressed.
    on_left_mouse_button_released: function
        Callback function for when the left mouse button is released.
    on_left_mouse_button_clicked: function
        Callback function for when clicked using the left mouse button
        (i.e. pressed -> released).
    on_left_mouse_double_clicked: function
        Callback function for when left mouse button is double clicked
        (i.e pressed -> released -> pressed -> released).
    on_left_mouse_button_dragged: function
        Callback function for when dragging using the left mouse button.
    on_right_mouse_button_pressed: function
        Callback function for when the right mouse button is pressed.
    on_right_mouse_button_released: function
        Callback function for when the right mouse button is released.
    on_right_mouse_button_clicked: function
        Callback function for when clicking using the right mouse button
        (i.e. pressed -> released).
    on_right_mouse_double_clicked: function
        Callback function for when right mouse button is double clicked
        (i.e pressed -> released -> pressed -> released).
    on_right_mouse_button_dragged: function
        Callback function for when dragging using the right mouse button.
    on_middle_mouse_button_pressed: function
        Callback function for when the middle mouse button is pressed.
    on_middle_mouse_button_released: function
        Callback function for when the middle mouse button is released.
    on_middle_mouse_button_clicked: function
        Callback function for when clicking using the middle mouse button
        (i.e. pressed -> released).
    on_middle_mouse_double_clicked: function
        Callback function for when middle mouse button is double clicked
        (i.e pressed -> released -> pressed -> released).
    on_middle_mouse_button_dragged: function
        Callback function for when dragging using the middle mouse button.
    on_key_press: function
        Callback function for when a keyboard key is pressed.

    """

    def __init__(self, *, position=(0, 0), x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP):
        """Init scene.

        Parameters
        ----------
        position : (float, float)
            Absolute pixel coordinates `(x, y)` which, in combination with
            `x_anchor` and `y_anchor`, define the initial placement of this
            UI component.
        x_anchor : str, optional
            Defines the horizontal anchor point for `position`. Can be "LEFT",
            "CENTER", or "RIGHT". Defaults to "LEFT".
        y_anchor : str, optional
            Defines the vertical anchor point for `position`. Can be "BOTTOM",
            "CENTER", or "TOP". Defaults to "BOTTOM".

        """
        self._position = np.array([0, 0])
        self._childrens = []
        self._actors = []

        self._setup()  # Setup needed actors and sub UI components.
        self.set_position(position, x_anchor, y_anchor)

        self.left_button_state = "released"
        self.right_button_state = "released"
        self.middle_button_state = "released"

        self.on_left_mouse_button_pressed = lambda event: None
        self.on_left_mouse_button_dragged = lambda event: None
        self.on_left_mouse_button_released = lambda event: None
        self.on_left_mouse_button_clicked = lambda event: None
        self.on_left_mouse_double_clicked = lambda event: None
        self.on_right_mouse_button_pressed = lambda event: None
        self.on_right_mouse_button_released = lambda event: None
        self.on_right_mouse_button_clicked = lambda event: None
        self.on_right_mouse_double_clicked = lambda event: None
        self.on_right_mouse_button_dragged = lambda event: None
        self.on_middle_mouse_button_pressed = lambda event: None
        self.on_middle_mouse_button_released = lambda event: None
        self.on_middle_mouse_button_clicked = lambda event: None
        self.on_middle_mouse_double_clicked = lambda event: None
        self.on_middle_mouse_button_dragged = lambda event: None
        self.on_key_press = lambda event: None

    @abc.abstractmethod
    def _setup(self):
        """Set up this UI component.

        This is where you should create all your needed actors and sub UI
        components.

        """
        msg = "Subclasses of UI must implement `_setup(self)`."
        raise NotImplementedError(msg)

    @property
    def actors(self):
        """Actors composing this UI component."""
        return self._actors

    @property
    def childrens(self):
        """Childrens composing this UI component."""
        return self._childrens

    def perform_position_validation(self, x_anchor, y_anchor):
        if not hasattr(self, "size"):
            msg = "Subclasses of UI must implement property `size`."
            raise NotImplementedError(msg)

        if x_anchor not in [Anchor.LEFT, Anchor.CENTER, Anchor.RIGHT]:
            raise ValueError(
                f"x_anchor should be one of these {', '.join([Anchor.LEFT, Anchor.CENTER, Anchor.RIGHT])} but received {x_anchor}"
            )

        if y_anchor not in [Anchor.TOP, Anchor.CENTER, Anchor.BOTTOM]:
            raise ValueError(
                f"y_anchor should be one of these {', '.join([Anchor.TOP, Anchor.CENTER, Anchor.BOTTOM])} but received {y_anchor}"
            )

    def set_position(
        self, coords, x_anchor: str = Anchor.LEFT, y_anchor: str = Anchor.BOTTOM
    ):
        """Position this UI component according to the specified anchor.

        Parameters
        ----------
        coords : (float, float)
            Absolute pixel coordinates (x, y). These coordinates
            are interpreted based on `x_anchor` and `y_anchor`.
        x_anchor : str, optional
            Defines the horizontal anchor point for `coords`. Can be "LEFT",
            "CENTER", or "RIGHT". Case-insensitive. Defaults to "LEFT".
        y_anchor : str, optional
            Defines the vertical anchor point for `coords`. Can be "BOTTOM",
            "CENTER", or "TOP". Case-insensitive. Defaults to "BOTTOM".

        """
        self.perform_position_validation(x_anchor=x_anchor, y_anchor=y_anchor)

        self._position = np.array(coords)
        self._anchors = [x_anchor.upper(), y_anchor.upper()]
        self._update_actors_position()

    def get_position(self, x_anchor: str = Anchor.LEFT, y_anchor: str = Anchor.TOP):
        """Get the position of this UI component according to the specified anchor.

        Parameters
        ----------
        x_anchor : str, optional
            Defines the horizontal anchor point for the returned coordinates.
            Can be "LEFT", "CENTER", or "RIGHT".
            Defaults to "LEFT".
        y_anchor : str, optional
            Defines the vertical anchor point for the returned coordinates.
            Can be "BOTTOM", "CENTER", or "TOP".
            Defaults to "BOTTOM".

        Returns
        -------
        (float, float)
            The (x, y) pixel coordinates of the specified anchor point.

        """
        ANCHOR_TO_MULTIPLIER = {
            Anchor.LEFT: 0.0,
            Anchor.RIGHT: 1.0,
            Anchor.TOP: 0.0 if UIContext.get_is_v2_ui() else 1.0,
            Anchor.BOTTOM: 1.0 if UIContext.get_is_v2_ui() else 0.0,
            Anchor.CENTER: 0.5,
        }

        self.perform_position_validation(x_anchor=x_anchor, y_anchor=y_anchor)

        return np.array(
            [
                self._position[0]
                + self.size[0]
                * (
                    ANCHOR_TO_MULTIPLIER[x_anchor.upper()]
                    - ANCHOR_TO_MULTIPLIER[self._anchors[0].upper()]
                ),
                self._position[1]
                + self.size[1]
                * (
                    ANCHOR_TO_MULTIPLIER[y_anchor.upper()]
                    - ANCHOR_TO_MULTIPLIER[self._anchors[1].upper()]
                ),
            ]
        )

    @abc.abstractmethod
    def _update_actors_position(self):
        """Update the position of the internal actors."""
        msg = "Subclasses of UI must implement `_set_actors_position(self, coords)`."
        raise NotImplementedError(msg)

    @property
    def size(self):
        return np.asarray(self._get_size(), dtype=int)

    @abc.abstractmethod
    def _get_size(self):
        msg = "Subclasses of UI must implement property `size`."
        raise NotImplementedError(msg)

    def set_visibility(self, visibility: bool):
        """Set visibility of this UI component."""
        for actor in self.actors:
            # actor.SetVisibility(visibility)
            actor.visible = visibility

    def handle_events(self, actor: Mesh):
        actor.add_event_handler(self.mouse_button_down_callback, "pointer_down")
        actor.add_event_handler(self.mouse_button_up_callback, "pointer_up")
        actor.add_event_handler(self.mouse_move_callback, "pointer_move")
        # actor.add_event_handler(self.mouse_button_up_callback, "pointer_enter")
        # actor.add_event_handler(self.mouse_button_down_callback, "pointer_leave")
        # actor.add_event_handler(self.mouse_button_up_callback, "click")
        # actor.add_event_handler(self.mouse_button_down_callback, "double_click")
        # actor.add_event_handler(self.mouse_button_up_callback, "wheel")

        # actor.add_event_handler(self.mouse_button_down_callback, "key_down")
        actor.add_event_handler(self.key_press_callback, "key_up")

        # self.add_callback(
        #     actor, "LeftButtonPressEvent", self.left_button_click_callback
        # )
        # self.add_callback(
        #     actor, "LeftButtonReleaseEvent", self.left_button_release_callback
        # )
        # self.add_callback(
        #     actor, "RightButtonPressEvent", self.right_button_click_callback
        # )
        # self.add_callback(
        #     actor, "RightButtonReleaseEvent", self.right_button_release_callback
        # )
        # self.add_callback(
        #     actor, "MiddleButtonPressEvent", self.middle_button_click_callback
        # )
        # self.add_callback(
        #     actor, "MiddleButtonReleaseEvent", self.middle_button_release_callback
        # )
        # self.add_callback(actor, "MouseMoveEvent", self.mouse_move_callback)
        # self.add_callback(actor, "KeyPressEvent", self.key_press_callback)

    def mouse_button_down_callback(self, event: PointerEvent):
        if event.button == 1:
            self.left_button_click_callback(event)
        elif event.button == 2:
            self.right_button_click_callback(event)
        elif event.button == 3:
            self.middle_button_click_callback(event)
        event.cancel()

    def mouse_button_up_callback(self, event: PointerEvent):
        if event.button == 1:
            self.left_button_release_callback(event)
        elif event.button == 2:
            self.right_button_release_callback(event)
        elif event.button == 3:
            self.middle_button_release_callback(event)
        event.cancel()

    def left_button_click_callback(self, event: PointerEvent):
        self.left_button_state = "pressing"
        self.on_left_mouse_button_pressed(event)

    def left_button_release_callback(self, event: PointerEvent):
        if self.left_button_state == "pressing":
            self.on_left_mouse_button_clicked(event)
        self.left_button_state = "released"
        self.on_left_mouse_button_released(event)

    def right_button_click_callback(self, event: PointerEvent):
        self.right_button_state = "pressing"
        self.on_right_mouse_button_pressed(event)

    def right_button_release_callback(self, event: PointerEvent):
        if self.right_button_state == "pressing":
            self.on_right_mouse_button_clicked(event)
        self.right_button_state = "released"
        self.on_right_mouse_button_released(event)

    def middle_button_click_callback(self, event: PointerEvent):
        self.middle_button_state = "pressing"
        self.on_middle_mouse_button_pressed(event)

    def middle_button_release_callback(self, event: PointerEvent):
        if self.middle_button_state == "pressing":
            self.on_middle_mouse_button_clicked(event)
        self.middle_button_state = "released"
        self.on_middle_mouse_button_released(event)

    def mouse_move_callback(self, event: PointerEvent):
        left_pressing_or_dragging = (
            self.left_button_state == "pressing" or self.left_button_state == "dragging"
        )

        right_pressing_or_dragging = (
            self.right_button_state == "pressing"
            or self.right_button_state == "dragging"
        )

        middle_pressing_or_dragging = (
            self.middle_button_state == "pressing"
            or self.middle_button_state == "dragging"
        )

        if left_pressing_or_dragging:
            self.left_button_state = "dragging"
            self.on_left_mouse_button_dragged(event)
        elif right_pressing_or_dragging:
            self.right_button_state = "dragging"
            self.on_right_mouse_button_dragged(event)
        elif middle_pressing_or_dragging:
            self.middle_button_state = "dragging"
            self.on_middle_mouse_button_dragged(event)

    def key_press_callback(self, event: KeyboardEvent):
        self.on_key_press(event)


class Rectangle2D(UI):
    """A 2D rectangle sub-classed from UI."""

    @warn_on_args_to_kwargs()
    def __init__(self, *, size=(0, 0), position=(0, 0), color=(1, 1, 1), opacity=1.0):
        """Initialize a rectangle.

        Parameters
        ----------
        size : (int, int)
            The size of the rectangle (width, height) in pixels.
        position : (float, float)
            Coordinates (x, y) of the lower-left corner of the rectangle.
        color : (float, float, float)
            Must take values in [0, 1].
        opacity : float
            Must take values in [0, 1].

        """
        super(Rectangle2D, self).__init__(
            position=position,
            x_anchor=Anchor.LEFT,
            y_anchor=Anchor.TOP if UIContext.get_is_v2_ui() else Anchor.BOTTOM,
        )
        self.color = color
        self.opacity = opacity
        self.resize(size)

    def _setup(self):
        """Set up this UI component.

        Creating the polygon actor used internally.
        """
        # # Setup four points
        # size = (1, 1)
        # self._points = Points()
        # self._points.InsertNextPoint(0, 0, 0)
        # self._points.InsertNextPoint(size[0], 0, 0)
        # self._points.InsertNextPoint(size[0], size[1], 0)
        # self._points.InsertNextPoint(0, size[1], 0)

        # # Create the polygon
        # polygon = Polygon()
        # polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
        # polygon.GetPointIds().SetId(0, 0)
        # polygon.GetPointIds().SetId(1, 1)
        # polygon.GetPointIds().SetId(2, 2)
        # polygon.GetPointIds().SetId(3, 3)

        # # Add the polygon to a list of polygons
        # polygons = CellArray()
        # polygons.InsertNextCell(polygon)

        # # Create a PolyData
        # self._polygonPolyData = PolyData()
        # self._polygonPolyData.SetPoints(self._points)
        # self._polygonPolyData.SetPolys(polygons)

        # # Create a mapper and actor
        # mapper = PolyDataMapper2D()
        # mapper = set_input(mapper, self._polygonPolyData)

        # self.actor = Actor2D()
        # self.actor.SetMapper(mapper)

        # # Add default events listener to the VTK actor.
        # self.handle_events(self.actor)

        geo = plane_geometry(width=1, height=1)
        mat = _create_mesh_material(
            material="basic", enable_picking=True, flat_shading=True
        )
        self.actor = create_mesh(geometry=geo, material=mat)

        self._actors.append(self.actor)
        self.handle_events(self.actor)

    def _get_size(self):
        # # Get 2D coordinates of two opposed corners of the rectangle.
        # lower_left_corner = np.array(self._points.GetPoint(0)[:2])
        # upper_right_corner = np.array(self._points.GetPoint(2)[:2])
        # size = abs(upper_right_corner - lower_left_corner)
        # return size
        bounds = self.actor.get_bounding_box()
        minx, miny, minz = bounds[0]
        maxx, maxy, maxz = bounds[1]
        return [maxx - minx, maxy - miny]

    @property
    def width(self):
        return self._get_size()[0]

    @width.setter
    def width(self, width):
        self.resize((width, self.height))

    @property
    def height(self):
        return self._get_size()[1]

    @height.setter
    def height(self, height):
        self.resize((self.width, height))

    def resize(self, size):
        """Set the button size.

        Parameters
        ----------
        size : (float, float)
            Button size (width, height) in pixels.

        """
        # self._points.SetPoint(0, 0, 0, 0.0)
        # self._points.SetPoint(1, size[0], 0, 0.0)
        # self._points.SetPoint(2, size[0], size[1], 0.0)
        # self._points.SetPoint(3, 0, size[1], 0.0)
        # self._polygonPolyData.SetPoints(self._points)
        # mapper = PolyDataMapper2D()
        # mapper = set_input(mapper, self._polygonPolyData)

        # self.actor.SetMapper(mapper)
        self.actor.geometry = plane_geometry(width=size[0], height=size[1])
        self._update_actors_position()

    def _update_actors_position(self):
        """Set the position of the internal actor."""
        position = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)
        canvas_size = UIContext.get_canvas_size()

        self.actor.local.x = position[0]
        self.actor.local.y = (
            canvas_size[1] - position[1] if UIContext.get_is_v2_ui() else position[1]
        )

    @property
    def color(self):
        """Get the rectangle's color."""
        # color = self.actor.GetProperty().GetColor()
        # return np.asarray(color)
        return self.actor.material.color

    @color.setter
    def color(self, color):
        """Set the rectangle's color.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].

        """
        # self.actor.GetProperty().SetColor(*color)
        self.actor.material.color = np.array([*color, 1.0])

    @property
    def opacity(self):
        """Get the rectangle's opacity."""
        # return self.actor.GetProperty().GetOpacity()
        return self.actor.material.opacity

    @opacity.setter
    def opacity(self, opacity):
        """Set the rectangle's opacity.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].

        """
        # self.actor.GetProperty().SetOpacity(opacity)
        self.actor.material.opacity = opacity


class Disk2D(UI):
    """A 2D disk UI component."""

    @warn_on_args_to_kwargs()
    def __init__(
        self,
        outer_radius,
        *,
        inner_radius=0,
        center=(0, 0),
        color=(1, 1, 1),
        opacity=1.0,
    ):
        """Initialize a 2D Disk.

        Parameters
        ----------
        outer_radius : int
            Outer radius of the disk.
        inner_radius : int, optional
            Inner radius of the disk. A value > 0, makes a ring.
        center : (float, float), optional
            Coordinates (x, y) of the center of the disk.
        color : (float, float, float), optional
            Must take values in [0, 1].
        opacity : float, optional
            Must take values in [0, 1].

        """
        self.actor = None
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius

        super(Disk2D, self).__init__(
            position=center, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER
        )

        self.color = color
        self.opacity = opacity

    def _setup(self):
        """Setup this UI component.

        Creating the disk actor used internally.

        """
        # # Setting up disk actor.
        # self._disk = DiskSource()
        # self._disk.SetRadialResolution(10)
        # self._disk.SetCircumferentialResolution(50)
        # self._disk.Update()

        # # Mapper
        # mapper = PolyDataMapper2D()
        # mapper = set_input(mapper, self._disk.GetOutputPort())

        # # Actor
        # self.actor = Actor2D()
        # self.actor.SetMapper(mapper)

        # # Add default events listener to the VTK actor.
        # self.handle_events(self.actor)

        self.actor = disk(
            centers=np.zeros((1, 3)), radii=self.outer_radius, material="basic"
        )

        self._actors.append(self.actor)
        self.handle_events(self.actor)

    def _get_size(self):
        diameter = 2 * self.outer_radius
        size = (diameter, diameter)
        return size

    def _update_actors_position(self):
        """Set the position of the internal actor."""
        position = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)
        canvas_size = UIContext.get_canvas_size()

        self.actor.local.x = position[0]
        self.actor.local.y = (
            canvas_size[1] - position[1] if UIContext.get_is_v2_ui() else position[1]
        )

    @property
    def color(self):
        """Get the color of this UI component."""
        # color = self.actor.GetProperty().GetColor()
        # return np.asarray(color)
        return self.actor.material.color

    @color.setter
    def color(self, color):
        """Set the color of this UI component.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].

        """
        # self.actor.GetProperty().SetColor(*color)
        self.actor.material.color = np.array([*color, 1.0])

    @property
    def opacity(self):
        """Get the opacity of this UI component."""
        # return self.actor.GetProperty().GetOpacity()
        return self.actor.material.opacity

    @opacity.setter
    def opacity(self, opacity):
        """Set the opacity of this UI component.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].

        """
        # self.actor.GetProperty().SetOpacity(opacity)
        self.actor.material.opacity = opacity

    @property
    def inner_radius(self):
        # return self._disk.GetInnerRadius()
        pass

    @inner_radius.setter
    def inner_radius(self, radius):
        # self._disk.SetInnerRadius(radius)
        # self._disk.Update()
        pass

    @property
    def outer_radius(self):
        # return self._disk.GetOuterRadius()
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, radius):
        # self._disk.SetOuterRadius(radius)
        # self._disk.Update()
        if self.actor:
            self.actor.geometry = prim_disk(radius=radius)
        self._outer_radius = radius


# class TextBlock2D(UI):
#     """Wrap over the default vtkTextActor and helps setting the text.

#     Contains member functions for text formatting.

#     Attributes
#     ----------
#     actor : :class:`vtkTextActor`
#         The text actor.
#     message : str
#         The initial text while building the actor.
#     position : (float, float)
#         (x, y) in pixels.
#     color : (float, float, float)
#         RGB: Values must be between 0-1.
#     bg_color : (float, float, float)
#         RGB: Values must be between 0-1.
#     font_size : int
#         Size of the text font.
#     font_family : str
#         Currently only supports Arial.
#     justification : str
#         left, right or center.
#     vertical_justification : str
#         bottom, middle or top.
#     bold : bool
#         Makes text bold.
#     italic : bool
#         Makes text italicised.
#     shadow : bool
#         Adds text shadow.
#     size : (int, int)
#         Size (width, height) in pixels of the text bounding box.
#     auto_font_scale : bool
#         Automatically scale font according to the text bounding box.
#     dynamic_bbox : bool
#         Automatically resize the bounding box according to the content.

#     """

#     @warn_on_args_to_kwargs()
#     def __init__(
#         self,
#         *,
#         text="Text Block",
#         font_size=18,
#         font_family="Arial",
#         justification="left",
#         vertical_justification="bottom",
#         bold=False,
#         italic=False,
#         shadow=False,
#         size=None,
#         color=(1, 1, 1),
#         bg_color=None,
#         position=(0, 0),
#         auto_font_scale=False,
#         dynamic_bbox=False,
#     ):
#         """Init class instance.

#         Parameters
#         ----------
#         text : str
#             The initial text while building the actor.
#         position : (float, float)
#             (x, y) in pixels.
#         color : (float, float, float)
#             RGB: Values must be between 0-1.
#         bg_color : (float, float, float)
#             RGB: Values must be between 0-1.
#         font_size : int
#             Size of the text font.
#         font_family : str
#             Currently only supports Arial.
#         justification : str
#             left, right or center.
#         vertical_justification : str
#             bottom, middle or top.
#         bold : bool
#             Makes text bold.
#         italic : bool
#             Makes text italicised.
#         shadow : bool
#             Adds text shadow.
#         size : (int, int)
#             Size (width, height) in pixels of the text bounding box.
#         auto_font_scale : bool, optional
#             Automatically scale font according to the text bounding box.
#         dynamic_bbox : bool, optional
#             Automatically resize the bounding box according to the content.

#         """
#         self.boundingbox = [0, 0, 0, 0]
#         super(TextBlock2D, self).__init__(position=position)
#         self.scene = None
#         self.have_bg = bool(bg_color)
#         self.color = color
#         self.background_color = bg_color
#         self.font_family = font_family
#         self._justification = justification
#         self.bold = bold
#         self.italic = italic
#         self.shadow = shadow
#         self._vertical_justification = vertical_justification
#         self._dynamic_bbox = dynamic_bbox
#         self.auto_font_scale = auto_font_scale
#         self.message = text
#         self.font_size = font_size
#         if size is not None:
#             self.resize(size)
#         elif not self.dynamic_bbox:
#             # raise ValueError("TextBlock size is required as it is not dynamic.")
#             self.resize((0, 0))

#     def _setup(self):
#         self.actor = TextActor()
#         self.actor.GetPosition2Coordinate().SetCoordinateSystemToViewport()
#         self.background = Rectangle2D()
#         self.handle_events(self.actor)

#     def resize(self, size):
#         """Resize TextBlock2D.

#         Parameters
#         ----------
#         size : (int, int)
#             Text bounding box size(width, height) in pixels.

#         """
#         self.update_bounding_box(size=size)

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         return [self.actor] + self.background.actors

#     def _add_to_scene(self, scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : scene

#         """
#         scene.add(self.background, self.actor)

#     @property
#     def message(self):
#         """Get message from the text.

#         Returns
#         -------
#         str
#             The current text message.

#         """
#         return self.actor.GetInput()

#     @message.setter
#     def message(self, text):
#         """Set the text message.

#         Parameters
#         ----------
#         text : str
#             The message to be set.

#         """
#         self.actor.SetInput(text)
#         if self.dynamic_bbox:
#             self.update_bounding_box()

#     @property
#     def font_size(self):
#         """Get text font size.

#         Returns
#         -------
#         int
#             Text font size.

#         """
#         return self.actor.GetTextProperty().GetFontSize()

#     @font_size.setter
#     def font_size(self, size):
#         """Set font size.

#         Parameters
#         ----------
#         size : int
#             Text font size.

#         """
#         if not self.auto_font_scale:
#             self.actor.SetTextScaleModeToNone()
#             self.actor.GetTextProperty().SetFontSize(size)

#         if self.dynamic_bbox:
#             self.update_bounding_box()

#     @property
#     def font_family(self):
#         """Get font family.

#         Returns
#         -------
#         str
#             Text font family.

#         """
#         return self.actor.GetTextProperty().GetFontFamilyAsString()

#     @font_family.setter
#     def font_family(self, family="Arial"):
#         """Set font family.

#         Currently Arial and Courier are supported.

#         Parameters
#         ----------
#         family : str
#             The font family.

#         """
#         if family == "Arial":
#             self.actor.GetTextProperty().SetFontFamilyToArial()
#         elif family == "Courier":
#             self.actor.GetTextProperty().SetFontFamilyToCourier()
#         else:
#             raise ValueError("Font not supported yet: {}.".format(family))

#     @property
#     def justification(self):
#         """Get text justification.

#         Returns
#         -------
#         str
#             Text justification.

#         """
#         return self._justification

#     @justification.setter
#     def justification(self, justification):
#         """Justify text.

#         Parameters
#         ----------
#         justification : str
#             Possible values are left, right, center.

#         """
#         self._justification = justification
#         self.update_alignment()

#     @property
#     def vertical_justification(self):
#         """Get text vertical justification.

#         Returns
#         -------
#         str
#             Text vertical justification.

#         """
#         return self._vertical_justification

#     @vertical_justification.setter
#     def vertical_justification(self, vertical_justification):
#         """Justify text vertically.

#         Parameters
#         ----------
#         vertical_justification : str
#             Possible values are bottom, middle, top.

#         """
#         self._vertical_justification = vertical_justification
#         self.update_alignment()

#     @property
#     def bold(self):
#         """Return whether the text is bold.

#         Returns
#         -------
#         bool
#             Text is bold if True.

#         """
#         return self.actor.GetTextProperty().GetBold()

#     @bold.setter
#     def bold(self, flag):
#         """Bold/un-bold text.

#         Parameters
#         ----------
#         flag : bool
#             Sets text bold if True.

#         """
#         self.actor.GetTextProperty().SetBold(flag)

#     @property
#     def italic(self):
#         """Return whether the text is italicised.

#         Returns
#         -------
#         bool
#             Text is italicised if True.

#         """
#         return self.actor.GetTextProperty().GetItalic()

#     @italic.setter
#     def italic(self, flag):
#         """Italicise/un-italicise text.

#         Parameters
#         ----------
#         flag : bool
#             Italicises text if True.

#         """
#         self.actor.GetTextProperty().SetItalic(flag)

#     @property
#     def shadow(self):
#         """Return whether the text has shadow.

#         Returns
#         -------
#         bool
#             Text is shadowed if True.

#         """
#         return self.actor.GetTextProperty().GetShadow()

#     @shadow.setter
#     def shadow(self, flag):
#         """Add/remove text shadow.

#         Parameters
#         ----------
#         flag : bool
#             Shadows text if True.

#         """
#         self.actor.GetTextProperty().SetShadow(flag)

#     @property
#     def color(self):
#         """Get text color.

#         Returns
#         -------
#         (float, float, float)
#             Returns text color in RGB.

#         """
#         return self.actor.GetTextProperty().GetColor()

#     @color.setter
#     def color(self, color=(1, 0, 0)):
#         """Set text color.

#         Parameters
#         ----------
#         color : (float, float, float)
#             RGB: Values must be between 0-1.

#         """
#         self.actor.GetTextProperty().SetColor(*color)

#     @property
#     def background_color(self):
#         """Get background color.

#         Returns
#         -------
#         (float, float, float) or None
#             If None, there no background color.
#             Otherwise, background color in RGB.

#         """
#         if not self.have_bg:
#             return None

#         return self.background.color

#     @background_color.setter
#     def background_color(self, color):
#         """Set text color.

#         Parameters
#         ----------
#         color : (float, float, float) or None
#             If None, remove background.
#             Otherwise, RGB values (must be between 0-1).

#         """
#         if color is None:
#             # Remove background.
#             self.have_bg = False
#             self.background.set_visibility(False)

#         else:
#             self.have_bg = True
#             self.background.set_visibility(True)
#             self.background.color = color

#     @property
#     def auto_font_scale(self):
#         """Return whether text font is automatically scaled.

#         Returns
#         -------
#         bool
#             Text is auto_font_scaled if True.

#         """
#         return self._auto_font_scale

#     @auto_font_scale.setter
#     def auto_font_scale(self, flag):
#         """Add/remove text auto_font_scale.

#         Parameters
#         ----------
#         flag : bool
#             Automatically scales the text font if True.

#         """
#         self._auto_font_scale = flag
#         if flag:
#             self.actor.SetTextScaleModeToProp()
#             self._justification = "left"
#             self.update_bounding_box(size=self.size)
#         else:
#             self.actor.SetTextScaleModeToNone()

#     @property
#     def dynamic_bbox(self):
#         """Automatically resize the bounding box according to the content.

#         Returns
#         -------
#         bool
#             Bounding box is dynamic if True.

#         """
#         return self._dynamic_bbox

#     @dynamic_bbox.setter
#     def dynamic_bbox(self, flag):
#         """Add/remove dynamic_bbox.

#         Parameters
#         ----------
#         flag : bool
#             The text bounding box is dynamic if True.

#         """
#         self._dynamic_bbox = flag
#         if flag:
#             self.update_bounding_box()

#     def update_alignment(self):
#         """Update Text Alignment."""
#         text_property = self.actor.GetTextProperty()
#         updated_text_position = [0, 0]

#         if self.justification.lower() == "left":
#             text_property.SetJustificationToLeft()
#             updated_text_position[0] = self.boundingbox[0]
#         elif self.justification.lower() == "center":
#             text_property.SetJustificationToCentered()
#             updated_text_position[0] = (
#                 self.boundingbox[0] + (self.boundingbox[2] - self.boundingbox[0]) // 2
#             )
#         elif self.justification.lower() == "right":
#             text_property.SetJustificationToRight()
#             updated_text_position[0] = self.boundingbox[2]
#         else:
#             msg = "Text can only be justified left, right and center."
#             raise ValueError(msg)

#         if self.vertical_justification.lower() == "bottom":
#             text_property.SetVerticalJustificationToBottom()
#             updated_text_position[1] = self.boundingbox[1]
#         elif self.vertical_justification.lower() == "middle":
#             text_property.SetVerticalJustificationToCentered()
#             updated_text_position[1] = (
#                 self.boundingbox[1] + (self.boundingbox[3] - self.boundingbox[1]) // 2
#             )
#         elif self.vertical_justification.lower() == "top":
#             text_property.SetVerticalJustificationToTop()
#             updated_text_position[1] = self.boundingbox[3]
#         else:
#             msg = "Vertical justification must be: bottom, middle or top."
#             raise ValueError(msg)

#         self.actor.SetPosition(updated_text_position)

#     def cal_size_from_message(self):
#         """Calculate size of background according to the message it contains."""
#         lines = self.message.split("\n")
#         max_length = max(len(line) for line in lines)
#         return [max_length * self.font_size, len(lines) * self.font_size]

#     @warn_on_args_to_kwargs()
#     def update_bounding_box(self, *, size=None):
#         """Update Text Bounding Box.

#         Parameters
#         ----------
#         size : (int, int) or None
#             If None, calculates bounding box.
#             Otherwise, uses the given size.

#         """
#         if size is None:
#             size = self.cal_size_from_message()

#         self.boundingbox = [
#             self.position[0],
#             self.position[1],
#             self.position[0] + size[0],
#             self.position[1] + size[1],
#         ]
#         self.background.resize(size)

#         if self.auto_font_scale:
#             self.actor.SetPosition2(
#                 self.boundingbox[2] - self.boundingbox[0],
#                 self.boundingbox[3] - self.boundingbox[1],
#             )
#         else:
#             self.update_alignment()

#     def _set_position(self, position):
#         """Set text actor position.

#         Parameters
#         ----------
#         position : (float, float)
#             The new position. (x, y) in pixels.

#         """
#         self.actor.SetPosition(*position)
#         self.background.position = position

#     def _get_size(self):
#         bb_size = (
#             self.boundingbox[2] - self.boundingbox[0],
#             self.boundingbox[3] - self.boundingbox[1],
#         )
#         if self.dynamic_bbox or self.auto_font_scale or sum(bb_size):
#             return bb_size
#         return self.cal_size_from_message()


# class Button2D(UI):
#     """A 2D overlay button and is of type vtkTexturedActor2D.

#     Currently supports::

#         - Multiple icons.
#         - Switching between icons.

#     """

#     @warn_on_args_to_kwargs()
#     def __init__(self, icon_fnames, *, position=(0, 0), size=(30, 30)):
#         """Init class instance.

#         Parameters
#         ----------
#         icon_fnames : List(string, string)
#             ((iconname, filename), (iconname, filename), ....)
#         position : (float, float), optional
#             Absolute coordinates (x, y) of the lower-left corner of the button.
#         size : (int, int), optional
#             Width and height in pixels of the button.

#         """
#         super(Button2D, self).__init__(position=position)

#         self.icon_extents = {}
#         self.icons = self._build_icons(icon_fnames)
#         self.icon_names = [icon[0] for icon in self.icons]
#         self.current_icon_id = 0
#         self.current_icon_name = self.icon_names[self.current_icon_id]
#         self.set_icon(self.icons[self.current_icon_id][1])
#         self.resize(size)

#     def _get_size(self):
#         lower_left_corner = self.texture_points.GetPoint(0)
#         upper_right_corner = self.texture_points.GetPoint(2)
#         size = np.array(upper_right_corner) - np.array(lower_left_corner)
#         return abs(size[:2])

#     def _build_icons(self, icon_fnames):
#         """Convert file names to ImageData.

#         A pre-processing step to prevent re-read of file names during every
#         state change.

#         Parameters
#         ----------
#         icon_fnames : List(string, string)
#             ((iconname, filename), (iconname, filename), ....)

#         Returns
#         -------
#         icons : List
#             A list of corresponding ImageData.

#         """
#         icons = []
#         for icon_name, icon_fname in icon_fnames:
#             icons.append((icon_name, load_image(icon_fname, as_vtktype=True)))

#         return icons

#     def _setup(self):
#         """Set up this UI component.

#         Creating the button actor used internally.

#         """
#         # This is highly inspired by
#         # https://github.com/Kitware/VTK/blob/c3ec2495b183e3327820e927af7f8f90d34c3474/Interaction/Widgets/vtkBalloonRepresentation.cxx#L47

#         self.texture_polydata = PolyData()
#         self.texture_points = Points()
#         self.texture_points.SetNumberOfPoints(4)

#         polys = CellArray()
#         polys.InsertNextCell(4)
#         polys.InsertCellPoint(0)
#         polys.InsertCellPoint(1)
#         polys.InsertCellPoint(2)
#         polys.InsertCellPoint(3)
#         self.texture_polydata.SetPolys(polys)

#         tc = FloatArray()
#         tc.SetNumberOfComponents(2)
#         tc.SetNumberOfTuples(4)
#         tc.InsertComponent(0, 0, 0.0)
#         tc.InsertComponent(0, 1, 0.0)
#         tc.InsertComponent(1, 0, 1.0)
#         tc.InsertComponent(1, 1, 0.0)
#         tc.InsertComponent(2, 0, 1.0)
#         tc.InsertComponent(2, 1, 1.0)
#         tc.InsertComponent(3, 0, 0.0)
#         tc.InsertComponent(3, 1, 1.0)
#         self.texture_polydata.GetPointData().SetTCoords(tc)

#         texture_mapper = PolyDataMapper2D()
#         texture_mapper = set_input(texture_mapper, self.texture_polydata)

#         button = TexturedActor2D()
#         button.SetMapper(texture_mapper)

#         self.texture = Texture()
#         button.SetTexture(self.texture)

#         button_property = Property2D()
#         button_property.SetOpacity(1.0)
#         button.SetProperty(button_property)
#         self.actor = button

#         # Add default events listener to the VTK actor.
#         self.handle_events(self.actor)

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         return [self.actor]

#     def _add_to_scene(self, scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : scene

#         """
#         scene.add(self.actor)

#     def resize(self, size):
#         """Resize the button.

#         Parameters
#         ----------
#         size : (float, float)
#             Button size (width, height) in pixels.

#         """
#         # Update actor.
#         self.texture_points.SetPoint(0, 0, 0, 0.0)
#         self.texture_points.SetPoint(1, size[0], 0, 0.0)
#         self.texture_points.SetPoint(2, size[0], size[1], 0.0)
#         self.texture_points.SetPoint(3, 0, size[1], 0.0)
#         self.texture_polydata.SetPoints(self.texture_points)

#     def _set_position(self, coords):
#         """Set the lower-left corner position of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         self.actor.SetPosition(*coords)

#     @property
#     def color(self):
#         """Get the button's color."""
#         color = self.actor.GetProperty().GetColor()
#         return np.asarray(color)

#     @color.setter
#     def color(self, color):
#         """Set the button's color.

#         Parameters
#         ----------
#         color : (float, float, float)
#             RGB. Must take values in [0, 1].

#         """
#         self.actor.GetProperty().SetColor(*color)

#     def scale(self, factor):
#         """Scale the button.

#         Parameters
#         ----------
#         factor : (float, float)
#             Scaling factor (width, height) in pixels.

#         """
#         self.resize(self.size * factor)

#     def set_icon_by_name(self, icon_name):
#         """Set the button icon using its name.

#         Parameters
#         ----------
#         icon_name : str

#         """
#         icon_id = self.icon_names.index(icon_name)
#         self.set_icon(self.icons[icon_id][1])

#     def set_icon(self, icon):
#         """Modify the icon used by the vtkTexturedActor2D.

#         Parameters
#         ----------
#         icon : imageData

#         """
#         self.texture = set_input(self.texture, icon)

#     def next_icon_id(self):
#         """Set the next icon ID while cycling through icons."""
#         self.current_icon_id += 1
#         if self.current_icon_id == len(self.icons):
#             self.current_icon_id = 0
#         self.current_icon_name = self.icon_names[self.current_icon_id]

#     def next_icon(self):
#         """Increment the state of the Button.

#         Also changes the icon.
#         """
#         self.next_icon_id()
#         self.set_icon(self.icons[self.current_icon_id][1])
