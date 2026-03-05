"""UI components module."""

__all__ = [
    "TexturedButton2D",
    "TextButton2D",
    "LineSlider2D",
    "SpinBox",
    #     "TextBox2D",
    #     "LineDoubleSlider2D",
    #     "RingSlider2D",
    #     "RangeSlider",
    #     "Checkbox",
    #     "Option",
    #     "RadioButton",
    #     "ComboBox2D",
    #     "ListBox2D",
    #     "ListBoxItem2D",
    #     "FileMenu2D",
    #     "DrawShape",
    #     "DrawPanel",
    #     "PlaybackPanel",
    #     "Card2D",
    #     "SpinBox",
]


import numpy as np

from fury.actor import create_mesh
from fury.io import load_image_texture
from fury.lib import (
    plane_geometry,
)
from fury.material import _create_mesh_material
from fury.ui.core import UI, Anchor, Button2D, Disk2D, Rectangle2D, TextBlock2D


class TexturedButton2D(Button2D):
    """A button component that swaps textures based on interaction state.

    Parameters
    ----------
    states : dict
        A mapping of state names to image file paths.
    position : (float, float)
        Absolute coordinates (x, y) for placement.
    size : (int, int)
        Width and height in pixels.
    """

    def __init__(self, states, position=(0, 0), size=(30, 30)):
        """Initialize the textured button instance.

        Parameters
        ----------
        states : dict
            A mapping of state names to image file paths.
        position : (float, float)
            Absolute coordinates (x, y) for placement.
        size : (int, int)
            Width and height in pixels.
        """
        self.texture_map = self._load_textures(states)
        super().__init__(position=position, size=size)

    def _load_textures(self, states):
        """Load image files into PyGfx textures.

        Parameters
        ----------
        states : dict
            Dictionary of state names and file paths.

        Returns
        -------
        dict
            A dictionary containing loaded Texture objects.
        """
        loaded = {}
        for name, fname in states.items():
            loaded[name] = load_image_texture(fname)
        return loaded

    def _setup(self):
        """Set up the internal mesh actor."""
        geo = plane_geometry(width=1, height=1)
        mat = _create_mesh_material(material="basic")
        self.child = create_mesh(geometry=geo, material=mat)
        self.handle_events(self.child)

    def update_visual_state(self):
        """Update the mesh texture based on the current button state."""
        if not self.child:
            return

        key = self.resolve_state_key(self.texture_map)
        if key:
            tex = self.texture_map[key]

            self.child.material = _create_mesh_material(
                material="basic", texture=tex, mode="auto"
            )
            self.child.material.color = np.array([1.0, 1.0, 1.0, 1.0])
        else:
            tint = 0.5 if self.is_pressed else (0.8 if self.is_hovered else 1.0)
            self.child.material.color = np.array([tint, tint, tint, 1.0])


class TextButton2D(Button2D):
    """A button component that updates text and color based on state.

    Parameters
    ----------
    label : str
        The default text to display on the button.
    states : dict
        Configuration for visual states. Supports mapping keys to RGB
        tuples or dictionaries containing 'text' and 'color' keys.
    position : (float, float)
        Absolute coordinates (x, y) for placement.
    size : (int, int)
        Width and height in pixels for the button background.
    font_size : int
        Size of the text font.
    """

    def __init__(
        self, label, states=None, position=(0, 0), size=(100, 40), font_size=25
    ):
        """Initialize the text button instance.

        Parameters
        ----------
        label : str
            The default text to display on the button.
        states : dict
            Configuration for visual states. Supports mapping keys to RGB
            tuples or dictionaries containing 'text' and 'color' keys.
        position : (float, float)
            Absolute coordinates (x, y) for placement.
        size : (int, int)
            Width and height in pixels for the button background.
        font_size : int
            Size of the text font.
        """
        self.default_label = label
        self.font_size = font_size

        self.states = states or {
            "default": (1, 1, 1),
            "hover": (0.9, 0.9, 0.9),
            "pressed": (0.5, 0.5, 0.5),
            "disabled": (0.2, 0.2, 0.2),
        }

        super().__init__(position=position, size=size)

    def _setup(self):
        """Set up the internal TextBlock2D component.

        Initializes the child text block with the default label and font settings.
        """
        self.child = TextBlock2D(
            text=self.default_label,
            color=(0, 0, 0),
            bg_color=(1, 1, 1),
            font_size=self.font_size,
            size=self._dims,
        )
        self.handle_events(self.child.actor)
        self.handle_events(self.child.background.actor)

    def update_visual_state(self):
        """Update the text message and background color based on state."""
        if not self.child:
            return

        key = self.resolve_state_key(self.states)
        if not key:
            return

        data = self.states[key]

        target_color = (1, 1, 1)
        target_text = self.default_label

        if isinstance(data, (tuple, list, np.ndarray)):
            target_color = data
        elif isinstance(data, dict):
            target_color = data.get("color", target_color)
            target_text = data.get("text", target_text)

        self.child.background.color = target_color

        if self.child.message != target_text:
            self.child.message = target_text


class LineSlider2D(UI):
    """A 2D Line Slider component.

    Parameters
    ----------
    position : (float, float), optional
        Absolute coordinates (x, y) for placement.
    initial_value : float, optional
        The starting value of the slider.
    min_value : float, optional
        The minimum value of the slider range.
    max_value : float, optional
        The maximum value of the slider range.
    length : int, optional
        The length of the slider track in pixels.
    line_width : int, optional
        The thickness of the slider track.
    inner_radius : int, optional
        The inner radius for disk-shaped handles (for rings).
    outer_radius : int, optional
        The outer radius for disk-shaped handles.
    handle_side : int, optional
        The side length for square-shaped handles.
    font_size : int, optional
        The font size for the value label.
    orientation : str, optional
        The slider orientation: "horizontal" or "vertical".
    text_template : str, optional
        A formatting string for the label. Supports {value} and {ratio}.
    shape : str, optional
        The handle shape: "disk" or "square".
    z_order : int, optional
        The stacking priority. The handle is assigned z_order + 1.
    """

    def __init__(
        self,
        *,
        position=(0, 0),
        initial_value=50,
        min_value=0,
        max_value=100,
        length=200,
        line_width=5,
        inner_radius=0,
        outer_radius=10,
        handle_side=20,
        font_size=16,
        orientation="horizontal",
        text_template="{value:.1f} ({ratio:.0%})",
        shape="disk",
        z_order=0,
    ):
        """Initialize the slider instance.

        Parameters
        ----------
        position : (float, float), optional
            Absolute coordinates (x, y) for placement.
        initial_value : float, optional
            The starting value of the slider.
        min_value : float, optional
            The minimum value of the slider range.
        max_value : float, optional
            The maximum value of the slider range.
        length : int, optional
            The length of the slider track in pixels.
        line_width : int, optional
            The thickness of the slider track.
        inner_radius : int, optional
            The inner radius for disk-shaped handles (for rings).
        outer_radius : int, optional
            The outer radius for disk-shaped handles.
        handle_side : int, optional
            The side length for square-shaped handles.
        font_size : int, optional
            The font size for the value label.
        orientation : str, optional
            The slider orientation: "horizontal" or "vertical".
        text_template : str, optional
            A formatting string for the label. Supports {value} and {ratio}.
        shape : str, optional
            The handle shape: "disk" or "square".
        z_order : int, optional
            The stacking priority. The handle is assigned z_order + 1.
        """
        self._ratio = 0
        self._value = 0

        self.shape = shape
        self.orientation = orientation.lower().strip()
        self.default_color = (1, 1, 1)
        self.active_color = (0, 0, 1)

        self._length = length
        self._line_width = line_width
        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._handle_side = handle_side
        self._font_size = font_size

        self.min_value = min_value
        self.max_value = max_value
        self.text_template = text_template

        super(LineSlider2D, self).__init__(
            position=position,
            x_anchor=Anchor.LEFT,
            y_anchor=Anchor.TOP,
            z_order=z_order,
        )

        self.on_change = lambda ui: None
        self.on_value_changed = lambda ui: None
        self.on_moving_slider = lambda ui: None

        self.value = initial_value

    def _setup(self):
        """Set up the internal actors."""
        track_size = (
            (self._length, self._line_width)
            if self.orientation == "horizontal"
            else (self._line_width, self._length)
        )
        self.track = Rectangle2D(size=track_size)
        self.track.color = (1, 0, 0)

        if self.shape == "disk":
            self.handle = Disk2D(
                outer_radius=self._outer_radius, inner_radius=self._inner_radius
            )
        elif self.shape == "square":
            self.handle = Rectangle2D(size=(self._handle_side, self._handle_side))
        self.handle.color = self.default_color
        self.handle.z_order = self.z_order + 1

        self.text = TextBlock2D(
            text=self.text_template, font_size=self._font_size, dynamic_bbox=True
        )

        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.track.on_left_mouse_button_released = self.handle_release_callback

        self.handle.on_left_mouse_button_dragged = self.handle_move_callback
        self.handle.on_left_mouse_button_released = self.handle_release_callback

    def _get_actors(self):
        """Get the actors composing this UI component.

        Returns
        -------
        list
            List of actors from the track, handle, and text elements.
        """
        return self.track.actors + self.handle.actors + self.text.actors

    def _get_size(self):
        """Calculate the total bounding box size of the slider.

        Returns
        -------
        numpy.ndarray
            The (width, height) in pixels.
        """
        if self.orientation == "horizontal":
            width = self._length
            height = max(self._line_width, self.handle.size[1])
        else:
            width = max(self._line_width, self.handle.size[0])
            height = self._length
        return np.array([width, height])

    def _update_actors_position(self):
        """Update the position of the track and handle actors."""
        pos = self.get_position(x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP)

        self.track.set_position(
            pos + self.size / 2, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER
        )

        self._update_handle_position()

    def _update_handle_position(self):
        """Calculate specific coordinates for the handle and text label."""
        track_origin = self.track.get_position(
            x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP
        )
        if self.orientation == "horizontal":
            offset = self.ratio * self._length
            handle_center = track_origin + np.array([offset, self._line_width / 2])
            text_pos = handle_center + np.array(
                [0, -(self.handle.size[1] + self.text.size[1] / 2)]
            )
        else:
            offset = self.ratio * self._length
            handle_center = track_origin + np.array([self._line_width / 2, offset])
            text_pos = handle_center + np.array(
                [self.handle.size[0] + self.text.size[0] / 2, 0]
            )

        self.handle.set_position(
            handle_center, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER
        )
        self.text.set_position(text_pos, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)

        self.text.message = self.text_template.format(
            value=self.value, ratio=self.ratio
        )

    @property
    def value(self):
        """Get the current numeric value of the slider.

        Returns
        -------
        float
            The slider value.
        """
        return self._value

    @value.setter
    def value(self, val):
        """Set the slider numeric value.

        Parameters
        ----------
        val : float
            New numeric value. Will be clamped to [min_value, max_value].
        """
        val = np.clip(val, self.min_value, self.max_value)
        self._value = val
        range_val = self.max_value - self.min_value
        self._ratio = (val - self.min_value) / range_val if range_val != 0 else 0
        self._update_handle_position()
        self.on_value_changed(self)

    @property
    def ratio(self):
        """Get the current normalized ratio (0 to 1).

        Returns
        -------
        float
            The slider ratio.
        """
        return self._ratio

    @ratio.setter
    def ratio(self, r):
        """Set the slider ratio.

        Parameters
        ----------
        r : float
            New ratio value. Will be clamped to [0, 1].
        """
        self._ratio = np.clip(r, 0, 1)
        self._value = self.min_value + self._ratio * (self.max_value - self.min_value)
        self._update_handle_position()

    def track_click_callback(self, event):
        """Handle mouse click events on the slider track.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self.handle_move_callback(event)

    def handle_move_callback(self, event):
        """Handle mouse drag events to update the slider state.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self.handle.color = self.active_color

        left_x = self.track.get_position(x_anchor=Anchor.LEFT)[0]
        bottom_y = self.track.get_position(y_anchor=Anchor.BOTTOM)[1]
        top_y = self.track.get_position(y_anchor=Anchor.TOP)[1]

        if self.orientation == "horizontal":
            new_ratio = (event.x - left_x) / self._length
        else:
            total_dist = bottom_y - top_y
            current_dist = event.y - top_y

            if total_dist != 0:
                new_ratio = current_dist / total_dist
            else:
                new_ratio = 0

        self.ratio = new_ratio

        self.on_moving_slider(self)
        self.on_change(self)

    def handle_release_callback(self, event):
        """Handle the release of the mouse button.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self.handle.color = self.default_color


class SpinBox(UI):
    """A 2D SpinBox component that allows incrementing or decrementing a value.

    The SpinBox renders as:  [ - ]  <value>  [ + ]

    The decrement button reduces the current value by ``step``, and the
    increment button increases it. Both are clamped to [min_value, max_value].

    Parameters
    ----------
    position : (float, float), optional
        Absolute coordinates (x, y) for the top-left corner of the widget.
    size : (int, int), optional
        Total width and height in pixels. The two buttons each occupy a
        square region whose side equals *height*; the value label fills
        the remaining width between them.
    min_value : float, optional
        Smallest value the SpinBox can hold.
    max_value : float, optional
        Largest value the SpinBox can hold.
    initial_value : float, optional
        Starting value. Clamped to [min_value, max_value] on init.
    step : float, optional
        Amount added or subtracted on each button click.
    font_size : int, optional
        Font size for the value label.
    text_template : str, optional
        A Python format string used to render the label. Receives the
        keyword argument ``value``, e.g. ``"{value:.2f}"``.
    z_order : int, optional
        Stacking priority passed through to child components.

    Attributes
    ----------
    on_change : callable
        Called with this SpinBox as the single argument whenever the value
        changes. Defaults to a no-op lambda.

    Examples
    --------
    >>> sb = SpinBox(position=(50, 50), size=(160, 40),
    ...              min_value=0, max_value=10, initial_value=5, step=1)
    >>> sb.on_change = lambda ui: print("value:", ui.value)
    """

    def __init__(
        self,
        *,
        position=(0, 0),
        size=(160, 40),
        min_value=0,
        max_value=100,
        initial_value=50,
        step=1,
        font_size=18,
        text_template="{value}",
        z_order=0,
    ):
        """Initialize the SpinBox instance.

        Parameters
        ----------
        position : (float, float), optional
            Absolute coordinates (x, y) for placement.
        size : (int, int), optional
            Total width and height in pixels.
        min_value : float, optional
            Minimum allowed value.
        max_value : float, optional
            Maximum allowed value.
        initial_value : float, optional
            Starting value (clamped to [min_value, max_value]).
        step : float, optional
            Increment/decrement amount per click.
        font_size : int, optional
            Font size for the displayed value label.
        text_template : str, optional
            Format string for the label. Supports ``{value}`` keyword.
        z_order : int, optional
            Stacking priority for child components.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self._value = 0  # set properly via property after super().__init__

        self._total_size = size
        self._font_size = font_size
        self.text_template = text_template

        super(SpinBox, self).__init__(
            position=position,
            x_anchor=Anchor.LEFT,
            y_anchor=Anchor.TOP,
            z_order=z_order,
        )

        # Public callback hook – mirrors the convention used in LineSlider2D.
        self.on_change = lambda ui: None

        # Set value *after* super().__init__ so that _update_label() can
        # safely reference self.label (created inside _setup).
        self.value = initial_value

    # ------------------------------------------------------------------
    # UI protocol
    # ------------------------------------------------------------------

    def _setup(self):
        """Create the decrement button, value label, and increment button."""
        width, height = self._total_size
        btn_size = (height, height)  # square buttons sized to the widget height

        # --- Decrement button ---
        self.btn_decrement = TextButton2D(
            label="-",
            size=btn_size,
            font_size=self._font_size,
            states={
                "default": (0.85, 0.85, 0.85),
                "hover":   (0.75, 0.75, 0.95),
                "pressed": (0.55, 0.55, 0.75),
            },
        )
        self.btn_decrement.on_left_mouse_button_clicked = self._on_decrement

        # --- Value label ---
        label_width = max(width - 2 * height, 1)
        self.label = TextBlock2D(
            text=self.text_template.format(value=self.min_value),
            color=(0, 0, 0),
            bg_color=(1, 1, 1),
            font_size=self._font_size,
            size=(label_width, height),
        )

        # --- Increment button ---
        self.btn_increment = TextButton2D(
            label="+",
            size=btn_size,
            font_size=self._font_size,
            states={
                "default": (0.85, 0.85, 0.85),
                "hover":   (0.75, 0.75, 0.95),
                "pressed": (0.55, 0.55, 0.75),
            },
        )
        self.btn_increment.on_left_mouse_button_clicked = self._on_increment

    def _get_actors(self):
        """Return all actors that compose the SpinBox.

        Returns
        -------
        list
            Combined actor lists from the decrement button, value label,
            and increment button.
        """
        return (
            self.btn_decrement.actors
            + self.label.actors
            + self.btn_increment.actors
        )

    def _get_size(self):
        """Return the total bounding-box size of the SpinBox.

        Returns
        -------
        numpy.ndarray
            Array of [width, height] in pixels.
        """
        return np.array(self._total_size)

    def _update_actors_position(self):
        """Place the three child components side-by-side inside the widget."""
        _, height = self._total_size
        btn_width = height  # buttons are square

        # Top-left origin of the whole widget.
        origin = self.get_position(x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP)

        # Decrement button – leftmost.
        self.btn_decrement.set_position(
            origin, x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP
        )

        # Value label – centre strip.
        label_origin = origin + np.array([btn_width, 0])
        self.label.set_position(
            label_origin, x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP
        )

        # Increment button – rightmost.
        inc_origin = origin + np.array([self._total_size[0] - btn_width, 0])
        self.btn_increment.set_position(
            inc_origin, x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP
        )

    # ------------------------------------------------------------------
    # Value property
    # ------------------------------------------------------------------

    @property
    def value(self):
        """Get the current numeric value.

        Returns
        -------
        float
            The current value held by the SpinBox.
        """
        return self._value

    @value.setter
    def value(self, val):
        """Set the SpinBox value, clamped to [min_value, max_value].

        Parameters
        ----------
        val : float
            Desired new value. Will be clamped silently.
        """
        self._value = float(np.clip(val, self.min_value, self.max_value))
        self._update_label()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_label(self):
        """Refresh the centre text label to reflect the current value.

        Skips the update if the label widget has not yet been created (i.e.
        during ``__init__`` before ``_setup`` has run).
        """
        if not hasattr(self, "label"):
            return
        self.label.message = self.text_template.format(value=self._value)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_increment(self, event):
        """Handle a click on the '+' button.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event (unused, but required by the callback
            signature).
        """
        old_value = self._value
        self.value = self._value + self.step
        if self._value != old_value:
            self.on_change(self)

    def _on_decrement(self, event):
        """Handle a click on the '-' button.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event (unused, but required by the callback
            signature).
        """
        old_value = self._value
        self.value = self._value - self.step
        if self._value != old_value:
            self.on_change(self)