"""UI core module that describe UI abstract class."""

import abc

import numpy as np

from fury.actor import Text, create_mesh
from fury.decorators import warn_on_args_to_kwargs
from fury.geometry import buffer_to_geometry
from fury.lib import (
    EventType,
    plane_geometry,
)
from fury.material import (
    _create_mesh_material,
)
from fury.primitive import prim_ring
from fury.ui import UIContext
from fury.ui.helpers import UI_Z_RANGE, Anchor, get_anchor_to_multiplier


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

    Parameters
    ----------
    position : (float, float)
        Absolute pixel coordinates `(x, y)` which, in combination with
        `x_anchor` and `y_anchor`, define the initial placement of this
        UI component.
    x_anchor : str, optional
        Define the horizontal anchor point for `position`. Can be "LEFT",
        "CENTER", or "RIGHT".
    y_anchor : str, optional
        Define the vertical anchor point for `position`. Can be "BOTTOM",
        "CENTER", or "TOP".
    z_order : int, optional
        The initial Z-order of the UI component.
    """

    def __init__(
        self, *, position=(0, 0), x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP, z_order=0
    ):
        """Init scene.

        Parameters
        ----------
        position : (float, float)
            Absolute pixel coordinates `(x, y)` which, in combination with
            `x_anchor` and `y_anchor`, define the initial placement of this
            UI component.
        x_anchor : str, optional
            Define the horizontal anchor point for `position`. Can be "LEFT",
            "CENTER", or "RIGHT".
        y_anchor : str, optional
            Define the vertical anchor point for `position`. Can be "BOTTOM",
            "CENTER", or "TOP".
        z_order : int, optional
            The initial Z-order of the UI component.
        """
        self._position = np.array([0, 0])
        self._children = []
        self._anchors = [x_anchor, y_anchor]
        self.z_order = z_order

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

    @abc.abstractmethod
    def _get_actors(self):
        """Get the actors composing this UI component."""
        msg = "Subclasses of UI must implement `_get_actors(self)`."
        raise NotImplementedError(msg)

    @property
    def actors(self):
        """Get actors composing this UI component.

        Returns
        -------
        list
            List of actors composing this UI component.
        """
        return self._get_actors()

    def perform_position_validation(self, x_anchor, y_anchor):
        """Perform validation checks for anchor string and the 'size' property.

        Parameters
        ----------
        x_anchor : str
            Horizontal anchor string to validate (e.g., "LEFT", "CENTER", "RIGHT").
        y_anchor : str
            Vertical anchor string to validate (e.g., "TOP", "CENTER", "BOTTOM").
        """
        if not hasattr(self, "size"):
            msg = "Subclasses of UI must implement property `size`."
            raise NotImplementedError(msg)

        if x_anchor not in [Anchor.LEFT, Anchor.CENTER, Anchor.RIGHT]:
            raise ValueError(
                f"x_anchor should be one of these {', '.join([Anchor.LEFT, Anchor.CENTER, Anchor.RIGHT])} but received {x_anchor}"  # noqa: E501
            )

        if y_anchor not in [Anchor.TOP, Anchor.CENTER, Anchor.BOTTOM]:
            raise ValueError(
                f"y_anchor should be one of these {', '.join([Anchor.TOP, Anchor.CENTER, Anchor.BOTTOM])} but received {y_anchor}"  # noqa: E501
            )

    def set_actor_position(self, actor, center_position, z_order):
        """Set the position of the PyGfx actor.

        Parameters
        ----------
        actor : Mesh
            The PyGfx mesh actor whose position needs to be set.
        center_position : tuple or ndarray
            A 2-element array `(x, y)` representing the desired center
            position of the actor.
        z_order : int
            The Z-order of the UI component.
        """
        canvas_size = UIContext.canvas_size

        actor.local.x = center_position[0]
        actor.local.y = canvas_size[1] - center_position[1]
        actor.local.z = np.interp(z_order, UIContext.z_order_bounds, UI_Z_RANGE)

    def set_position(self, coords, x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP):
        """Position this UI component according to the specified anchor.

        Parameters
        ----------
        coords : (float, float)
            Absolute pixel coordinates (x, y). These coordinates
            are interpreted based on `x_anchor` and `y_anchor`.
        x_anchor : str, optional
            Define the horizontal anchor point for `coords`. Can be "LEFT",
            "CENTER", or "RIGHT".
        y_anchor : str, optional
            Define the vertical anchor point for `coords`. Can be "TOP",
            "CENTER", or "BOTTOM".
        """
        self.perform_position_validation(x_anchor=x_anchor, y_anchor=y_anchor)

        self._position = np.array(coords)
        self._anchors = [x_anchor.upper(), y_anchor.upper()]
        self._update_actors_position()

    def get_position(
        self,
        x_anchor=Anchor.LEFT,
        y_anchor=Anchor.TOP,
    ):
        """Get the position of this UI component according to the specified anchor.

        Parameters
        ----------
        x_anchor : str, optional
            Define the horizontal anchor point for the returned coordinates.
            Can be "LEFT", "CENTER", or "RIGHT".
        y_anchor : str, optional
            Define the vertical anchor point for the returned coordinates.
            Can be "BOTTOM", "CENTER", or "TOP".

        Returns
        -------
        (float, float)
            The (x, y) pixel coordinates of the specified anchor point.
        """

        ANCHOR_TO_MULTIPLIER = get_anchor_to_multiplier()

        self.perform_position_validation(x_anchor=x_anchor, y_anchor=y_anchor)
        size = self.size

        return np.array(
            [
                self._position[0]
                + size[0]
                * (
                    ANCHOR_TO_MULTIPLIER[x_anchor.upper()]
                    - ANCHOR_TO_MULTIPLIER[self._anchors[0].upper()]
                ),
                self._position[1]
                + size[1]
                * (
                    ANCHOR_TO_MULTIPLIER[y_anchor.upper()]
                    - ANCHOR_TO_MULTIPLIER[self._anchors[1].upper()]
                ),
            ]
        )

    @property
    def z_order(self):
        """Get the Z-order of this UI element.

        Returns
        -------
        int
            Z-order of the UI.
        """
        return self._z_order

    @z_order.setter
    def z_order(self, z_order):
        """Set the Z-order of this UI element.

        Parameters
        ----------
        z_order : int
            The new integer Z-order value.

        Raises
        ------
        ValueError
            If the provided `z_order` is not an integer.
        """
        if not isinstance(z_order, int):
            raise ValueError("Z-order must be an integer.")

        self._z_order = z_order
        UIContext.z_order_bounds = z_order

    @abc.abstractmethod
    def _update_actors_position(self):
        """Update the position of the internal actors."""
        msg = "Subclasses of UI must implement `_set_actors_position(self)`."
        raise NotImplementedError(msg)

    @property
    def size(self):
        """Get width and height of this UI component.

        Returns
        -------
        (int, int)
            Width and Height of UI component in pixels.
        """
        return np.asarray(self._get_size(), dtype=int)

    @abc.abstractmethod
    def _get_size(self):
        """Get the actual size of the UI component.

        Returns
        -------
        (int, int)
            Width and height of the UI component in pixels.
        """
        msg = "Subclasses of UI must implement property `size`."
        raise NotImplementedError(msg)

    def set_visibility(self, visibility):
        """Set visibility of this UI component.

        Parameters
        ----------
        visibility : bool
            If `True`, the UI component will be visible. If `False`, it will be hidden.
        """
        for actor in self.actors:
            actor.visible = visibility

    def handle_events(self, actor):
        """Attach event handlers to the UI object.

        Parameters
        ----------
        actor : Mesh
            The PyGfx mesh to which event handlers should be attached.
        """
        actor.add_event_handler(self.mouse_button_down_callback, EventType.POINTER_DOWN)
        actor.add_event_handler(self.mouse_button_up_callback, EventType.POINTER_UP)
        actor.add_event_handler(self.mouse_move_callback, EventType.POINTER_DRAG)
        actor.add_event_handler(self.key_press_callback, EventType.KEY_UP)

    def mouse_button_down_callback(self, event):
        """Handle mouse button press event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if event.button == 1:
            self.left_button_click_callback(event)
        elif event.button == 2:
            self.right_button_click_callback(event)
        elif event.button == 3:
            self.middle_button_click_callback(event)

    def mouse_button_up_callback(self, event):
        """Handle mouse button release event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if event.button == 1:
            self.left_button_release_callback(event)
        elif event.button == 2:
            self.right_button_release_callback(event)
        elif event.button == 3:
            self.middle_button_release_callback(event)

    def left_button_click_callback(self, event):
        """Handle left mouse button press event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self.left_button_state = "pressing"
        self.on_left_mouse_button_pressed(event)

    def left_button_release_callback(self, event):
        """Handle left mouse button release event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self.left_button_state == "pressing":
            self.on_left_mouse_button_clicked(event)
        self.left_button_state = "released"
        self.on_left_mouse_button_released(event)

    def right_button_click_callback(self, event):
        """Handle right mouse button press event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self.right_button_state = "pressing"
        self.on_right_mouse_button_pressed(event)

    def right_button_release_callback(self, event):
        """Handle right mouse button release event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self.right_button_state == "pressing":
            self.on_right_mouse_button_clicked(event)
        self.right_button_state = "released"
        self.on_right_mouse_button_released(event)

    def middle_button_click_callback(self, event):
        """Handle middle mouse button press event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self.middle_button_state = "pressing"
        self.on_middle_mouse_button_pressed(event)

    def middle_button_release_callback(self, event):
        """Handle middle mouse button release event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self.middle_button_state == "pressing":
            self.on_middle_mouse_button_clicked(event)
        self.middle_button_state = "released"
        self.on_middle_mouse_button_released(event)

    def mouse_move_callback(self, event):
        """Handle mouse move event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
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

    def key_press_callback(self, event):
        """Handle key press event.

        Parameters
        ----------
        event : KeyboardEvent
            The PyGfx keyboard event object.
        """
        self.on_key_press(event)


class Rectangle2D(UI):
    """A 2D rectangle sub-classed from UI.

    Parameters
    ----------
    size : (int, int), optional
        Initial `(width, height)` of the rectangle in pixels.
    position : (float, float), optional
        Coordinates `(x, y)` of the rectangle. The interpretation of `(x,y)`
        (e.g., top-left, bottom-left) depends on the current UI version.
    color : (float, float, float), optional
        RGB color tuple, with values in the range `[0, 1]`.
    opacity : float, optional
        Degree of transparency, with values in the range `[0, 1]`.
        `0` is fully transparent, `1` is fully opaque.
    """

    @warn_on_args_to_kwargs()
    def __init__(
        self, *, size=(100, 100), position=(0, 0), color=(1, 1, 1), opacity=1.0
    ):
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
        super(Rectangle2D, self).__init__(position=position)
        self.color = color
        self.opacity = opacity
        self.resize(size)

    def _setup(self):
        """Set up this UI component.

        Create the plane actor used internally.
        """
        geo = plane_geometry(width=1, height=1)
        mat = _create_mesh_material(
            material="basic", enable_picking=True, flat_shading=True
        )
        self.actor = create_mesh(geometry=geo, material=mat)

        self.handle_events(self.actor)

    def _get_actors(self):
        """Get the actors composing this UI component.

        Returns
        -------
        list
            List of actors composing this UI component.
        """
        return [self.actor]

    def _get_size(self):
        """Get the current size of the rectangle actor.

        Returns
        -------
        (float, float)
            The current `(width, height)` of the rectangle in pixels.
        """
        bounds = self.actor.get_bounding_box()
        minx, miny, minz = bounds[0]
        maxx, maxy, maxz = bounds[1]
        return [maxx - minx, maxy - miny]

    @property
    def width(self):
        """Get the current width of the rectangle.

        Returns
        -------
        float
            The width of the rectangle in pixels.
        """
        return self._get_size()[0]

    @width.setter
    def width(self, width):
        """Set the width of the rectangle.

        Parameters
        ----------
        width : float
            New width of the rectangle.
        """
        self.resize((width, self.height))

    @property
    def height(self):
        """Get the current height of the rectangle.

        Returns
        -------
        float
            The height of the rectangle in pixels.
        """
        return self._get_size()[1]

    @height.setter
    def height(self, height):
        """Set the height of the rectangle.

        Parameters
        ----------
        height : float
            New height of the rectangle.
        """
        self.resize((self.width, height))

    def resize(self, size):
        """Set the rectangle size.

        Parameters
        ----------
        size : (float, float)
            Rectangle size (width, height) in pixels.
        """
        self.actor.geometry = plane_geometry(width=size[0], height=size[1])
        self._update_actors_position()

    def _update_actors_position(self):
        """Set the position of the internal actor."""
        position = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)

        self.set_actor_position(self.actor, position, self.z_order)

    @property
    def color(self):
        """Get the rectangle color.

        Returns
        -------
        (float, float, float)
            RGB color.
        """
        return self.actor.material.color[:3]

    @color.setter
    def color(self, color):
        """Set the rectangle color.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].
        """
        self.actor.material.color = np.array([*color, 1.0])

    @property
    def opacity(self):
        """Get the rectangle opacity.

        Returns
        -------
        float
            Opacity value.
        """
        return self.actor.material.opacity

    @opacity.setter
    def opacity(self, opacity):
        """Set the rectangle opacity.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].
        """
        self.actor.material.opacity = opacity


class Disk2D(UI):
    """A 2D disk UI component.

    Parameters
    ----------
    outer_radius : int
        Outer radius of the disk.
    inner_radius : int
        Inner radius of the disk.
    center : (float, float), optional
        Coordinates (x, y) of the center of the disk.
    color : (float, float, float), optional
        Must take values in [0, 1].
    opacity : float, optional
        Must take values in [0, 1].
    """

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
            Inner radius of the disk.
        center : (float, float), optional
            Coordinates (x, y) of the center of the disk.
        color : (float, float, float), optional
            Must take values in [0, 1].
        opacity : float, optional
            Must take values in [0, 1].
        """
        self.actor = None
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

        super(Disk2D, self).__init__(
            position=center, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER
        )

        self.color = color
        self.opacity = opacity

    def _setup(self):
        """Set up this UI component.

        Create the disk actor used internally.
        """
        positions, indices = prim_ring(
            inner_radius=self.inner_radius, outer_radius=self.outer_radius
        )
        geo = buffer_to_geometry(positions=positions, indices=indices)
        mat = _create_mesh_material(
            material="basic", enable_picking=True, flat_shading=True
        )
        self.actor = create_mesh(geometry=geo, material=mat)

        self.handle_events(self.actor)

    def _get_actors(self):
        """Get the actors composing this UI component.

        Returns
        -------
        list
            List of actors composing this UI component.
        """
        return [self.actor]

    def _get_size(self):
        """Get the current size of the disk.

        Returns
        -------
        (float, float)
            The current `(diameter, diameter)` of the disk in pixels.
        """
        diameter = 2 * self.outer_radius
        size = (diameter, diameter)
        return size

    def _update_actors_position(self):
        """Set the position of the internal actor."""
        position = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)

        self.set_actor_position(self.actor, position, self.z_order)

    @property
    def color(self):
        """Get the color of this UI component.

        Returns
        -------
        (float, float, float)
            RGB color.
        """
        return self.actor.material.color[:3]

    @color.setter
    def color(self, color):
        """Set the color of this UI component.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].
        """
        self.actor.material.color = np.array([*color, 1.0])

    @property
    def opacity(self):
        """Get the opacity of this UI component.

        Returns
        -------
        float
            Opacity value.
        """
        return self.actor.material.opacity

    @opacity.setter
    def opacity(self, opacity):
        """Set the opacity of this UI component.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].
        """
        self.actor.material.opacity = opacity

    @property
    def outer_radius(self):
        """Get the outer radius of the disk.

        Returns
        -------
        int
            Outer radius in pixels.
        """
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, radius):
        """Set the outer radius of the disk.

        Parameters
        ----------
        radius : int
            New outer radius.
        """
        if self.actor:
            positions, indices = prim_ring(
                inner_radius=self.inner_radius, outer_radius=radius
            )
            self.actor.geometry = buffer_to_geometry(
                positions=positions, indices=indices
            )
        self._outer_radius = radius

    @property
    def inner_radius(self):
        """Get the inner radius of the disk.

        Returns
        -------
        int
            Inner radius in pixels.
        """
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, radius):
        """Set the inner radius of the disk.

        Parameters
        ----------
        radius : int
            New inner radius.
        """
        if self.actor:
            positions, indices = prim_ring(
                inner_radius=radius, outer_radius=self.outer_radius
            )
            self.actor.geometry = buffer_to_geometry(
                positions=positions, indices=indices
            )
        self._inner_radius = radius


class TextBlock2D(UI):
    """A 2D text component with optional background.

    Parameters
    ----------
    text : str, optional
        The initial text message.
    font_size : int, optional
        Size of the text font.
    font_family : str, optional
        The font family name.
    justification : str, optional
        Horizontal alignment ("left", "center", "right").
    vertical_justification : str, optional
        Vertical alignment ("top", "middle", "bottom").
    bold : bool, optional
        If True, makes text bold.
    italic : bool, optional
        If True, makes text italicized.
    size : (int, int), optional
        The (width, height) in pixels for the text bounding box.
    color : (float, float, float), optional
        RGB color for the text (0-1).
    bg_color : (float, float, float), optional
        RGB color for the background (0-1). If None, no background is drawn.
    position : (float, float), optional
        Absolute coordinates (x, y) for placement.
    dynamic_bbox : bool, optional
        If True, resizes the bounding box to fit the content.
    """

    def __init__(
        self,
        *,
        text="Text Block",
        font_size=18,
        font_family="Arial",
        justification="left",
        vertical_justification="bottom",
        bold=False,
        italic=False,
        size=None,
        color=(1, 1, 1),
        bg_color=None,
        position=(0, 0),
        dynamic_bbox=False,
    ):
        """Initialize the text block instance.

        Parameters
        ----------
        text : str, optional
            The initial text message.
        font_size : int, optional
            Size of the text font.
        font_family : str, optional
            The font family name.
        justification : str, optional
            Horizontal alignment ("left", "center", "right").
        vertical_justification : str, optional
            Vertical alignment ("top", "middle", "bottom").
        bold : bool, optional
            If True, makes text bold.
        italic : bool, optional
            If True, makes text italicized.
        size : (int, int), optional
            The (width, height) in pixels for the text bounding box.
        color : (float, float, float), optional
            RGB color for the text (0-1).
        bg_color : (float, float, float), optional
            RGB color for the background (0-1). If None, no background is drawn.
        position : (float, float), optional
            Absolute coordinates (x, y) for placement.
        dynamic_bbox : bool, optional
            If True, resizes the bounding box to fit the content.
        """
        self.boundingbox = [0, 0, 0, 0]
        self._message = text
        self._dynamic_bbox = dynamic_bbox
        self._bg_size = size

        self._last_rendered_size = (0, 0)

        if self._bg_size is None and not self.dynamic_bbox:
            raise ValueError("TextBlock size is required as it is not dynamic.")

        self._justification = justification
        self._vertical_justification = vertical_justification
        super(TextBlock2D, self).__init__(position=position)
        self.have_bg = bool(bg_color)
        self.color = color
        self.background_color = bg_color
        self.font_family = font_family
        self.bold = bold
        self.italic = italic
        self.message = text
        self.font_size = font_size

        self.update_bounding_box()

    def _setup(self):
        """Set up this UI component."""
        self.actor = Text(
            markdown=self._message, screen_space=True, anchor="middle-center"
        )
        self.background = Rectangle2D()
        self.handle_events(self.actor)

    def resize(self, size):
        """Resize the TextBlock2D bounding box.

        Parameters
        ----------
        size : (int, int)
            The new (width, height) in pixels.
        """
        self.actor.max_width = size[1]
        self.update_bounding_box(size=size)

    def update_layout(self):
        """Update the component layout based on current text dimensions."""
        current_w, current_h = self.get_text_actor_size()
        last_w, last_h = self._last_rendered_size

        if abs(current_w - last_w) > 0.1 or abs(current_h - last_h) > 0.1:
            self._last_rendered_size = (current_w, current_h)

            if self.dynamic_bbox:
                self.update_bounding_box()
            else:
                self.update_alignment()

    def _get_actors(self):
        """Get the actors composing this UI component.

        Returns
        -------
        list
            List containing the text actor and background actors.
        """
        return [self.actor] + self.background.actors

    def get_formatted_text(self, text):
        """Format the given text with markdown syntax for bold/italic styles.

        Parameters
        ----------
        text : str
            The raw text to format.

        Returns
        -------
        str
            The formatted markdown string.
        """
        affix_char = ""
        if self.bold:
            affix_char = "**"
        elif self.italic:
            affix_char = "*"

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        formatted_lines = [f"{affix_char}{line}{affix_char}" for line in lines]

        return "\n".join(formatted_lines)

    @property
    def message(self):
        """Get the current text message.

        Returns
        -------
        str
            The text message.
        """
        return self._message

    @message.setter
    def message(self, text):
        """Set the text message.

        Parameters
        ----------
        text : str
            The message to display.
        """
        self._message = text
        self.actor.set_markdown(self.get_formatted_text(text))
        if self.dynamic_bbox:
            self.update_bounding_box()

    @property
    def font_size(self):
        """Get text font size.

        Returns
        -------
        int
            Text font size.
        """
        return self.actor.font_size

    @font_size.setter
    def font_size(self, size):
        """Set text font size.

        Parameters
        ----------
        size : int
            Text font size.
        """
        self.actor.font_size = size
        if self.dynamic_bbox:
            self.update_bounding_box()

    @property
    def font_family(self):
        """Get font family.

        Returns
        -------
        str
            Text font family.
        """
        return self.actor.family

    @font_family.setter
    def font_family(self, family="Arial"):
        """Set font family.

        Parameters
        ----------
        family : str
            The font family.
        """
        self.actor.family = family
        if self.dynamic_bbox:
            self.update_bounding_box()

    @property
    def justification(self):
        """Get text justification.

        Returns
        -------
        str
            Text justification.
        """
        return self._justification

    @justification.setter
    def justification(self, justification):
        """Justify text.

        Parameters
        ----------
        justification : str
            Possible values are left, center, right.
        """
        self._justification = justification
        self.update_alignment()

    @property
    def vertical_justification(self):
        """Get text vertical justification.

        Returns
        -------
        str
            Text vertical justification.
        """
        return self._vertical_justification

    @vertical_justification.setter
    def vertical_justification(self, vertical_justification):
        """Justify text vertically.

        Parameters
        ----------
        vertical_justification : str
            Possible values are top, middle, bottom.
        """
        self._vertical_justification = vertical_justification
        self.update_alignment()

    @property
    def bold(self):
        """Return whether the text is bold.

        Returns
        -------
        bool
            Text is bold if True.
        """
        return self._bold

    @bold.setter
    def bold(self, flag):
        """Bold/un-bold text.

        Parameters
        ----------
        flag : bool
            Sets text bold if True.
        """
        self._bold = flag

    @property
    def italic(self):
        """Return whether the text is italicised.

        Returns
        -------
        bool
            Text is italicised if True.
        """
        return self._italic

    @italic.setter
    def italic(self, flag):
        """Italicise/un-italicise text.

        Parameters
        ----------
        flag : bool
            Italicises text if True.
        """
        self._italic = flag

    @property
    def color(self):
        """Get text color.

        Returns
        -------
        (float, float, float)
            Returns text color in RGB.
        """
        return self.actor.material.color[:3]

    @color.setter
    def color(self, color):
        """Set text color.

        Parameters
        ----------
        color : (float, float, float)
            RGB: Values must be between 0-1.
        """
        if color is None:
            color = (1, 1, 1)
        self.actor.material.color = np.array([*color, 1.0])

    @property
    def background_color(self):
        """Get the background color.

        Returns
        -------
        (float, float, float) or None
            The RGB color of the background, or None if no background exists.
        """
        if not self.have_bg:
            return None

        return self.background.color

    @background_color.setter
    def background_color(self, color):
        """Set the background color.

        Parameters
        ----------
        color : (float, float, float) or None
            RGB values (0-1). If None, the background is removed.
        """
        if color is None:
            # Remove background.
            self.have_bg = False
            self.background.set_visibility(False)

        else:
            self.have_bg = True
            self.background.set_visibility(True)
            self.background.color = color

    @property
    def dynamic_bbox(self):
        """Check if the bounding box is dynamic.

        Returns
        -------
        bool
            True if dynamic, False otherwise.
        """
        return self._dynamic_bbox

    @dynamic_bbox.setter
    def dynamic_bbox(self, flag):
        """Set the dynamic bounding box state.

        Parameters
        ----------
        flag : bool
            If True, the bounding box resizes to content.
        """
        self._dynamic_bbox = flag
        if flag:
            self.update_bounding_box()

    def update_alignment(self):
        """Update the text actor alignment within the bounding box."""
        updated_text_position = [0, 0]
        text_actor_size = self.get_text_actor_size()

        if self.justification.lower() == "left":
            self.actor.text_align = "left"
            updated_text_position[0] = self.boundingbox[0] + text_actor_size[0] // 2
        elif self.justification.lower() == "center":
            self.actor.text_align = "center"
            updated_text_position[0] = (
                self.boundingbox[0] + (self.boundingbox[2] - self.boundingbox[0]) // 2
            )
        elif self.justification.lower() == "right":
            self.actor.text_align = "right"
            updated_text_position[0] = self.boundingbox[2] - text_actor_size[0] // 2
        else:
            msg = "Text can only be justified left, center and right."
            raise ValueError(msg)

        if self.vertical_justification.lower() == "top":
            updated_text_position[1] = self.boundingbox[1] + text_actor_size[1] // 2
        elif self.vertical_justification.lower() == "middle":
            updated_text_position[1] = (
                self.boundingbox[1] + (self.boundingbox[3] - self.boundingbox[1]) // 2
            )
        elif self.vertical_justification.lower() == "bottom":
            updated_text_position[1] = self.boundingbox[3] - text_actor_size[1] // 2
        else:
            msg = "Vertical justification must be: top, middle or bottom."
            raise ValueError(msg)

        self.set_actor_position(self.actor, updated_text_position, self.z_order)

    def update_bounding_box(self, *, size=None):
        """Update the text bounding box and background.

        Parameters
        ----------
        size : (int, int), optional
            If provided, uses this size. Otherwise, uses the current size.
        """
        if size is None:
            size = self.size

        pos = self.get_position()
        self.boundingbox = [
            pos[0],
            pos[1],
            pos[0] + size[0],
            pos[1] + size[1],
        ]
        self.background.resize(size)
        self.background.set_position(pos)

        self.update_alignment()

    def _update_actors_position(self):
        """Update the position of the internal actors."""
        self.update_bounding_box()

    def get_text_actor_size(self):
        """Get the rendered size of the text actor.

        Returns
        -------
        (float, float)
            The (width, height) of the rendered text.
        """
        return (
            self.actor._aabb[1][0] - self.actor._aabb[0][0],
            self.actor._aabb[1][1] - self.actor._aabb[0][1],
        )

    def _get_size(self):
        """Get the size of the text block.

        Returns
        -------
        (float, float)
            The current size of the text block.
        """
        if self.dynamic_bbox:
            return self.get_text_actor_size()
        else:
            return self._bg_size


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
