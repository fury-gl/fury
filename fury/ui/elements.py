"""UI components module."""

__all__ = [
    "TexturedButton2D",
    "TextButton2D",
    "LineSlider2D",
    "TextBox2D",
    #     "LineSlider2D",
    #     "LineDoubleSlider2D",
    "RingSlider2D",
    #     "RangeSlider",
    #     "Checkbox",
    #     "Option",
    #     "RadioButton",
    "ComboBox2D",
    "ListBox2D",
    "ListBoxItem2D",
    #     "FileMenu2D",
    #     "DrawShape",
    #     "DrawPanel",
    "PlaybackPanel",
    "Card2D",
    #     "SpinBox",
]


from numbers import Number
from string import printable
import textwrap

from PIL import UnidentifiedImageError
import numpy as np

from fury.colormap import normalize_colors
from fury.data import read_viz_icons
from fury.io import get_extension, load_image, load_image_texture
from fury.lib import EventType
from fury.ui.containers import ImageContainer2D, Panel2D
from fury.ui.context import UIContext
from fury.ui.core import (
    UI,
    Anchor,
    Button2D,
    Disk2D,
    Rectangle2D,
    Slider2D,
    TextBlock2D,
)
from fury.ui.helpers import clip_overflow

TWO_PI = 2.0 * np.pi

LOWERS = r"`1234567890-=[]\;',./"
UPPERS = r'~!@#$%^&*()_+{}|:"<>?'
SHIFT_TRANS = str.maketrans(LOWERS, UPPERS)


class TexturedButton2D(Button2D):
    """
    A button component that swaps textures based on interaction state.

    Parameters
    ----------
    states : dict
        A mapping of state names to image file paths.
    position : (float, float)
        Absolute coordinates (x, y) for placement.
    size : (int, int)
        Width and height in pixels.
    is_toggle : bool, optional
        If True, the button behaves as a toggle switch.
    """

    def __init__(self, states, position=(0, 0), size=(30, 30), is_toggle=False):
        """Initialize the textured button instance."""
        self.texture_map = self._load_textures(states)
        super().__init__(position=position, size=size, is_toggle=is_toggle)

    def _load_textures(self, states):
        """
        Load image files into PyGfx textures.

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
        dummy_img = np.zeros((1, 1, 4), dtype=np.uint8)
        self.child = ImageContainer2D(img_path=dummy_img, size=self._dims)
        self.handle_events(self.child.actor)

    def update_visual_state(self):
        """Update the mesh texture based on the current button state."""
        if not self.child:
            return

        key = self.resolve_state_key(self.texture_map)
        if key:
            tex = self.texture_map[key]

            self.child.actor.material.map = tex
            self.child.actor.material.needs_update = True
            self.child.color = (1.0, 1.0, 1.0)
        else:
            tint = 0.5 if self.is_pressed else (0.8 if self.is_hovered else 1.0)
            self.child.color = (tint, tint, tint)


class TextButton2D(Button2D):
    """
    A button component that updates text and color based on state.

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
    is_toggle : bool, optional
        If True, the button behaves as a toggle switch.
    """

    def __init__(
        self,
        label,
        states=None,
        position=(0, 0),
        size=(100, 40),
        font_size=25,
        is_toggle=False,
    ):
        """Initialize the text button instance."""
        self.default_label = label
        self.font_size = font_size

        self.states = states or {
            "default": (1, 1, 1),
            "hover": (0.9, 0.9, 0.9),
            "pressed": (0.5, 0.5, 0.5),
            "disabled": (0.2, 0.2, 0.2),
        }

        super().__init__(position=position, size=size, is_toggle=is_toggle)

    def _setup(self):
        """
        Set up the internal TextBlock2D component.

        Initializes the child text block with the default label and font
        settings.
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


class LineSlider2D(Slider2D):
    """
    A 2D Line Slider component.

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
        """Initialize the slider instance."""
        self.orientation = orientation.lower().strip()
        self._length = length
        self._line_width = line_width

        super(LineSlider2D, self).__init__(
            position=position,
            initial_value=initial_value,
            min_value=min_value,
            max_value=max_value,
            handle_inner_radius=inner_radius,
            handle_outer_radius=outer_radius,
            handle_side=handle_side,
            font_size=font_size,
            text_template=text_template,
            shape=shape,
            z_order=z_order,
        )

        self.value = initial_value

    def _setup(self):
        """Set up the internal actors."""
        super(LineSlider2D, self)._setup()
        track_size = (
            (self._length, self._line_width)
            if self.orientation == "horizontal"
            else (self._line_width, self._length)
        )
        self.track = Rectangle2D(size=track_size)
        self.track.color = (1, 0, 0)
        self.track.z_order = self.z_order

        self.handle.color = self.default_color

        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.track.on_left_mouse_button_released = self.handle_release_callback

        self.handle.on_left_mouse_button_dragged = self.handle_move_callback
        self.handle.on_left_mouse_button_released = self.handle_release_callback

        self._children.extend([self.track, self.handle, self.text])

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            Empty list as this UI uses other UI elements as children
            instead of direct actors.
        """
        return []

    def _get_size(self):
        """
        Calculate the total bounding box size of the slider.

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

        self.track.z_order = self.z_order
        self.handle.z_order = self.z_order + 1
        self.text.z_order = self.z_order + 2

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

        self.text.message = self.format_text()

    def handle_move_callback(self, event):
        """
        Handle mouse drag events to update the slider state.

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


class PlaybackPanel(UI):
    """
    A playback controller designed for FURY v2.

    Parameters
    ----------
    loop : bool, optional
        If True, the playback starts in looping mode.
    position : (float, float), optional
        Absolute coordinates (x, y) for placement.
    width : int, optional
        The total width of the playback panel in pixels.
    z_order : int, optional
        The stacking priority of the panel.
    """

    def __init__(self, *, loop=False, position=(0, 0), width=900, z_order=0):
        """
        Initialize the playback panel instance.

        Parameters
        ----------
        loop : bool, optional
            If True, the playback starts in looping mode.
        position : (float, float), optional
            Absolute coordinates (x, y) for placement.
        width : int, optional
            The total width of the playback panel in pixels.
        z_order : int, optional
            The stacking priority of the panel.
        """
        self._drag_offset = None

        self._width = width
        self._playing = False
        self._loop = None

        self.on_play_pause_toggle = lambda state: None
        self.on_play = lambda: None
        self.on_pause = lambda: None
        self.on_stop = lambda: None
        self.on_loop_toggle = lambda is_looping: None
        self.on_progress_bar_changed = lambda x: None
        self.on_speed_changed = lambda x: None

        super(PlaybackPanel, self).__init__(position=position, z_order=z_order)

        self.loop() if loop else self.play_once()
        self.current_time = 0
        self.speed = 1.0

    def _setup(self):
        """
        Set up internal components including buttons, slider, and text labels.
        """
        self.panel = Panel2D(
            size=(220, 45),
            color=(1, 1, 1),
            has_border=True,
            border_color=(0, 0.3, 0),
            border_width=2,
        )

        self.time_text = TextBlock2D(
            text="00:00.00",
            font_size=16,
            color=(1, 1, 1),
            justification="left",
            vertical_justification="middle",
            dynamic_bbox=True,
        )
        self.speed_text = TextBlock2D(
            text="1x",
            font_size=21,
            color=(0.2, 0.2, 0.2),
            bold=True,
            justification="center",
            vertical_justification="middle",
            dynamic_bbox=True,
        )

        icon_play_pause = {
            "default": read_viz_icons(fname="play3.png"),
            "pressed": read_viz_icons(fname="pause2.png"),
        }
        icon_loop = {
            "default": read_viz_icons(fname="checkmark.png"),
            "pressed": read_viz_icons(fname="infinite.png"),
        }

        self._play_pause_btn = TexturedButton2D(
            states=icon_play_pause, size=(25, 25), is_toggle=True
        )
        self._stop_btn = TexturedButton2D(
            states={"default": read_viz_icons(fname="stop2.png")}, size=(25, 25)
        )
        self._loop_btn = TexturedButton2D(
            states=icon_loop, size=(25, 25), is_toggle=True
        )
        self._speed_up_btn = TexturedButton2D(
            states={"default": read_viz_icons(fname="plus.png")}, size=(15, 15)
        )
        self._slow_down_btn = TexturedButton2D(
            states={"default": read_viz_icons(fname="minus.png")}, size=(15, 15)
        )

        self._progress_bar = LineSlider2D(
            initial_value=0,
            length=self._width - 330,
            line_width=9,
            text_template="",
            shape="disk",
            outer_radius=10,
        )
        self._progress_bar.track.color = (1, 0, 0)

        self.panel.add_element(self._play_pause_btn, (10, 10))
        self.panel.add_element(self._stop_btn, (45, 10))
        self.panel.add_element(self._loop_btn, (80, 10))
        self.panel.add_element(self._slow_down_btn, (125, 15))
        self.panel.add_element(self.speed_text, (157, 15), anchor="center")
        self.panel.add_element(self._speed_up_btn, (195, 15))

        self._play_pause_btn.on_clicked = self._play_pause_callback
        self._stop_btn.on_clicked = lambda e: self.stop()
        self._loop_btn.on_clicked = self._loop_callback
        self._speed_up_btn.on_clicked = self._speed_up_callback
        self._slow_down_btn.on_clicked = self._slow_down_callback
        self._progress_bar.on_moving_slider = self._on_progress_change

        self.panel.on_left_mouse_button_pressed = self.left_button_pressed
        self.panel.on_left_mouse_button_dragged = self.left_button_dragged

        self.panel.background.on_left_mouse_button_pressed = self.left_button_pressed
        self.panel.background.on_left_mouse_button_dragged = self.left_button_dragged

    def _update_actors_position(self):
        """Update internal actor positions."""
        pos = self.get_position(x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP)

        self.panel.set_position(pos + (5, 5))

        pbar_length = max(self._width - 330, 10.0)
        self._progress_bar._length = pbar_length

        self._progress_bar.set_position(
            (pos[0] + 240, pos[1] + 27), x_anchor=Anchor.LEFT, y_anchor=Anchor.CENTER
        )

        self.time_text.set_position(
            (pos[0] + 250 + pbar_length, pos[1] + 27),
            x_anchor=Anchor.LEFT,
            y_anchor=Anchor.CENTER,
        )

        self._children.extend([self.panel, self._progress_bar, self.time_text])

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            Empty list as this UI uses other UI elements as children
            instead of direct actors.
        """
        return []

    def _get_size(self):
        """
        Get the total width and height of the playback panel.

        Returns
        -------
        numpy.ndarray
            The (width, height) in pixels.
        """
        return np.array([self._width, 55])

    def _play_pause_callback(self, event):
        """
        Handle toggle logic between play and pause states.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self._playing = not self._playing
        self.play() if self._playing else self.pause()
        self.on_play_pause_toggle(self._playing)

    def _loop_callback(self, event):
        """
        Handle toggle logic for the looping state.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self._loop = not self._loop
        self.loop() if self._loop else self.play_once()
        self.on_loop_toggle(self._loop)

    def _speed_up_callback(self, event):
        """
        Increment the playback speed.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        inc = 10 ** np.floor(np.log10(self.speed))
        self.speed = round(self.speed + inc, 13)
        self.on_speed_changed(self._speed)

    def _slow_down_callback(self, event):
        """
        Decrement the playback speed.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        safe_speed = max(self.speed - self.speed / 10, 0.01)
        dec = 10 ** np.floor(np.log10(safe_speed))
        self.speed = round(self.speed - dec, 13)
        self.on_speed_changed(self._speed)

    def _on_progress_change(self, slider):
        """
        Update time tracking based on slider movement.

        Parameters
        ----------
        slider : LineSlider2D
            The slider component instance.
        """
        self.on_progress_bar_changed(slider.value)
        self.current_time = slider.value

    def play(self):
        """Set the controller to playing state."""
        self._playing = True
        self._play_pause_btn.toggled = True
        self.on_play()

    def pause(self):
        """Set the controller to paused state."""
        self._playing = False
        self._play_pause_btn.toggled = False
        self.on_pause()

    def stop(self):
        """Stop the playback and reset the timer."""
        self._playing = False
        self.current_time = 0
        self._play_pause_btn.toggled = False
        self.on_stop()

    def loop(self):
        """Enable looping mode."""
        self._loop = True
        self._loop_btn.toggled = True

    def play_once(self):
        """Disable looping mode."""
        self._loop = False
        self._loop_btn.toggled = False

    @property
    def current_time(self):
        """
        Get the current playback time.

        Returns
        -------
        float
            Current time in seconds.
        """
        return self._progress_bar.value

    @current_time.setter
    def current_time(self, t):
        """
        Set the current playback time.

        Parameters
        ----------
        t : float
            New time in seconds.
        """
        self._progress_bar.value = t
        self.current_time_str = t

    @property
    def final_time(self):
        """
        Get the total duration of the playback.

        Returns
        -------
        float
            Total duration in seconds.
        """
        return self._progress_bar.max_value

    @final_time.setter
    def final_time(self, t):
        """
        Set the total duration of the playback.

        Parameters
        ----------
        t : float
            New total duration.
        """
        self._progress_bar.max_value = t

    @property
    def current_time_str(self):
        """
        Get the formatted string representation of current time.

        Returns
        -------
        str
            Formatted time string.
        """
        return self.time_text.message

    @current_time_str.setter
    def current_time_str(self, t):
        """
        Update the time label string based on seconds.

        Parameters
        ----------
        t : float
            Time in seconds.
        """
        t = np.clip(t, 0, self.final_time)
        m, s = divmod(t, 60)
        if self.final_time < 3600:
            t_str = f"{int(m):02d}:{s:05.2f}"
        else:
            h, m = divmod(m, 60)
            t_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        self.time_text.message = t_str

    @property
    def speed(self):
        """
        Get the current playback speed.

        Returns
        -------
        float
            Playback speed multiplier.
        """
        return self._speed

    @speed.setter
    def speed(self, val):
        """
        Set the playback speed multiplier.

        Parameters
        ----------
        val : float
            New speed value.
        """
        self._speed = max(val, 0.01)
        speed_str = f"{self._speed}".strip("0").rstrip(".") + "x"
        self.speed_text.message = speed_str if speed_str and speed_str != "." else "0"

    def left_button_pressed(self, event):
        """
        Handle left mouse button press event for PlaybackPanel.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        click_pos = np.array([event.x, event.y])
        self._drag_offset = click_pos - self.get_position()

    def left_button_dragged(self, event):
        """
        Handle left mouse button drag event for PlaybackPanel.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self._drag_offset is not None:
            click_position = np.array([event.x, event.y])
            new_position = click_position - self._drag_offset
            self.set_position(new_position)


class TextBox2D(UI):
    """
    An editable 2D text box that behaves as a UI component.

    Currently supports:
    - Basic text editing.
    - Cursor movements.
    - Single and multi-line text boxes.
    - Pre text formatting (text needs to be formatted beforehand).

    Parameters
    ----------
    width : int
        The number of characters in a single line of text.
    height : int
        The number of lines in the textbox.
    text : str, optional
        The initial text while building the actor.
    position : (float, float), optional
        (x, y) in pixels.
    color : str, tuple, list or ndarray, optional
        A hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
    font_size : int, optional
        Size of the text font.
    font_family : str, optional
        Currently only supports Arial.
    justification : str, optional
        Left, right, or center.
    bold : bool, optional
        Makes text bold.
    italic : bool, optional
        Makes text italic.
    shadow : bool, optional
        Adds text shadow.
    z_order : int, optional
        Rendering order of the widget.

    Attributes
    ----------
    text : :class:`TextBlock2D`
        The internal text UI component.
    width : int
        The number of characters in a single line of text.
    height : int
        The number of lines in the textbox.
    window_left : int
        Left limit of visible text in the textbox.
    window_right : int
        Right limit of visible text in the textbox.
    caret_pos : int
        Position of the caret in the text.
    init : bool
        Flag which says whether the textbox has just been initialized.
    """

    def __init__(
        self,
        width,
        height,
        *,
        text="Enter Text",
        position=(100, 10),
        color=(0, 0, 0),
        font_size=18,
        font_family="Arial",
        justification="left",
        bold=False,
        italic=False,
        shadow=False,
        z_order=0,
    ):
        """
        Init this UI element.

        Parameters
        ----------
        width : int
            The number of characters in a single line of text.
        height : int
            The number of lines in the textbox.
        text : str, optional
            The initial text while building the actor.
        position : (float, float), optional
            (x, y) in pixels.
        color : str, tuple, list or ndarray, optional
            A hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
        font_size : int, optional
            Size of the text font.
        font_family : str, optional
            Currently only supports Arial.
        justification : str, optional
            Left, right, or center.
        bold : bool, optional
            Makes text bold.
        italic : bool, optional
            Makes text italic.
        shadow : bool, optional
            Adds text shadow.
        z_order : int, optional
            Rendering order of the widget.
        """
        self._width = width
        self._height = height
        self._max_height = height
        self._message = text
        self._color = color
        self._font_size = font_size
        self._font_family = font_family
        self._justification = justification
        self._bold = bold
        self._italic = italic
        self._shadow = shadow
        self._z_order = z_order

        self.window_left = 0
        self.window_right = 0
        self.caret_pos = 0
        self.init = True
        self._has_focus = False

        self._shift_pressed = False
        self._caps_lock_on = False

        super(TextBox2D, self).__init__(
            position=position,
            x_anchor=Anchor.LEFT,
            y_anchor=Anchor.TOP,
            z_order=z_order,
        )

    def _setup(self):
        """
        Setup this UI component.

        Create the TextBlock2D component used for the textbox.
        Uses dynamic_bbox so the bounding box adapts as the user types.
        """
        bold_factor = 1.25 if self._bold else 1.0
        italic_factor = 1.1 if self._italic else 1.0

        bg_width = int(
            self._width * self._font_size * 0.5 * bold_factor * italic_factor
        )
        bg_height = int(self._height * self._font_size * 1.5) + 10

        self.text = TextBlock2D(
            text=self._message,
            font_size=self._font_size,
            font_family=self._font_family,
            justification=self._justification,
            vertical_justification="middle",
            dynamic_bbox=False,
            size=(bg_width, bg_height),
        )
        self.text.color = self._color
        self.text.bold = self._bold
        self.text.italic = self._italic
        self.text.shadow = self._shadow
        self.text.background_color = (1, 1, 1)

        self._children.append(self.text)

        self.window_left = 0
        self.window_right = self._width * self._height - 1
        self.caret_pos = len(self._message) if not self.init else 0

        self.text.on_left_mouse_button_pressed = self.left_button_press
        self.text.on_blur = self.blur_textbox
        self.text.on_key_press = self.key_press
        self.text.on_key_release = self.key_release
        self.text.on_wheel = self.wheel_scroll

    def _update_height(self):
        """
        Update the window boundaries of the textbox.

        In static mode, the background height remains constant.
        """
        self.window_right = self.window_left + self._width * self._height - 1

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            List of actors from all child UI components.
        """
        actors = []
        for child in self._children:
            actors.extend(child.actors)
        return actors

    def _get_size(self):
        """
        Return the size of the textbox.

        Returns
        -------
        tuple
            Width and height of the text bounding box.
        """
        return self.text.size

    def _update_actors_position(self):
        """Update the position of the text actor (anchor-aware)."""
        pos = self.get_position(x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP)

        self.text.set_position(pos, x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP)

        self.render_text(show_caret=False)

    def set_message(self, message):
        """
        Set custom text to textbox.

        Parameters
        ----------
        message : str
            The custom message to be set.
        """
        self._message = message
        self.init = False
        self.window_left = 0
        self.window_right = self._width * self._height - 1
        self.caret_pos = len(self._message)
        self.render_text(show_caret=False)

    def width_set_text(self, text):
        """
        Add newlines to text where necessary, needed for multi-line text boxes.

        Parameters
        ----------
        text : str
            The final text to be formatted.

        Returns
        -------
        str
            A multi-line formatted text.
        """
        lines = text.split("\n")
        formatted_lines = []
        for line in lines:
            if not line:
                formatted_lines.append("")
                continue
            wrapped = textwrap.wrap(
                line,
                width=self._width,
                drop_whitespace=False,
                replace_whitespace=False,
                break_long_words=True,
            )
            if not wrapped:
                formatted_lines.append("")
            else:
                formatted_lines.extend(wrapped)
        return "\n".join(formatted_lines)

    def handle_character(self, key, key_char, modifiers=None):
        """
        Handle button events.

        # TODO: Need to handle all kinds of characters like !, +, etc.

        Parameters
        ----------
        key : str
            The key identifier.
        key_char : str
            The character representation of the key.
        modifiers : tuple, optional
            The active keyboard modifiers.

        Returns
        -------
        bool
            True if editing is finished, otherwise False.
        """
        modifiers = modifiers or []
        k = key.lower() if isinstance(key, str) else None

        if k in ("enter", "return"):
            if "Shift" in modifiers:
                self.add_character("\n")
            else:
                self.render_text(show_caret=False)
                self._has_focus = False
                UIContext.active_ui = None
                self.on_blur(None)
                return True

        if key_char != "":
            is_shift = "Shift" in modifiers
            is_caps = "CapsLock" in modifiers

            if is_shift:
                key_char = key_char.translate(SHIFT_TRANS)

            if key_char.isalpha() and len(key_char) == 1:
                if is_shift != is_caps:
                    key_char = key_char.upper()
                elif is_shift and is_caps:
                    key_char = key_char.lower()

            if key_char in printable:
                self.add_character(key_char)

        if k == "backspace":
            self.remove_character()
        elif k in ("arrowleft", "left"):
            self.move_left()
        elif k in ("arrowright", "right"):
            self.move_right()
        elif k in ("arrowup", "up"):
            self.move_up()
        elif k in ("arrowdown", "down"):
            self.move_down()

        self.render_text()
        return False

    def move_caret_right(self):
        """Move the caret towards right."""
        self.caret_pos = min(self.caret_pos + 1, len(self._message))

    def move_caret_left(self):
        """Move the caret towards left."""
        self.caret_pos = max(self.caret_pos - 1, 0)

    def right_move_right(self):
        """Move right boundary of the text window right-wards."""
        if self.window_right <= len(self._message):
            self.window_right += 1

    def right_move_left(self):
        """Move right boundary of the text window left-wards."""
        if self.window_right > 0:
            self.window_right -= 1

    def left_move_right(self):
        """Move left boundary of the text window right-wards."""
        if self.window_left <= len(self._message):
            self.window_left += 1

    def left_move_left(self):
        """Move left boundary of the text window left-wards."""
        if self.window_left > 0:
            self.window_left -= 1

    def _adjust_window(self):
        """Adjust the window boundaries to ensure caret and text visibility."""
        if self.caret_pos < self.window_left:
            if self._height > 1:
                self.window_left = (self.caret_pos // self._width) * self._width
            else:
                self.window_left = self.caret_pos

        self._update_height()
        if self.caret_pos > self.window_right:
            if self._height > 1:
                line_of_caret = self.caret_pos // self._width
                self.window_left = max(
                    0, (line_of_caret - self._height + 1) * self._width
                )
            else:
                self.window_left = self.caret_pos - self._width + 1

        if self._height > 1:
            max_window_left = max(
                0, (len(self._message) // self._width - self._height + 1) * self._width
            )
        else:
            max_window_left = max(0, len(self._message) - self._width + 1)

        if self.window_left > max_window_left:
            self.window_left = max_window_left

        self._update_height()

    def add_character(self, character):
        """
        Insert a character into the text and moves window and caret.

        Parameters
        ----------
        character : str
                The character to be inserted.
        """
        if len(character) > 1 and character.lower() != "space":
            return
        if character.lower() == "space":
            character = " "
        self._message = (
            self._message[: self.caret_pos]
            + character
            + self._message[self.caret_pos :]
        )
        self.move_caret_right()
        self._adjust_window()

    def remove_character(self):
        """Remove a character and moves window and caret accordingly."""
        if self.caret_pos == 0:
            return
        self._message = (
            self._message[: self.caret_pos - 1] + self._message[self.caret_pos :]
        )
        self.move_caret_left()
        self._adjust_window()

    def move_left(self):
        """Handle left button press."""
        self.move_caret_left()
        self._adjust_window()

    def move_right(self):
        """Handle right button press."""
        self.move_caret_right()
        self._adjust_window()

    def move_up(self):
        """Handle up button press."""
        if self._height > 1:
            self.caret_pos = max(0, self.caret_pos - self._width)
        else:
            self.caret_pos = 0
        self._adjust_window()

    def move_down(self):
        """Handle down button press."""
        if self._height > 1:
            self.caret_pos = min(len(self._message), self.caret_pos + self._width)
        else:
            self.caret_pos = len(self._message)
        self._adjust_window()

    def showable_text(self, show_caret):
        """
        Chop out text to be shown on the screen.

        Parameters
        ----------
        show_caret : bool
            Whether or not to show the caret.

        Returns
        -------
        str
            The visible portion of the text.
        """
        ret_text = self._message[self.window_left : self.window_right + 1]

        rel_caret = self.caret_pos - self.window_left
        if 0 <= rel_caret <= len(ret_text):
            marker = "\x00_" if show_caret else "\x00"
            ret_text = ret_text[:rel_caret] + marker + ret_text[rel_caret:]

        return ret_text

    def render_text(self, *, show_caret=True):
        """
        Render text after processing.

        Parameters
        ----------
        show_caret : bool
            Whether or not to show the caret.
        """
        text = self.showable_text(show_caret)
        if text == "" or text == "\x00":
            text = "\x00Enter Text"

        formatted = self.width_set_text(text)

        lines = formatted.split("\n")
        if len(lines) > self._height:
            caret_line = 0
            for i, line in enumerate(lines):
                if "\x00" in line:
                    caret_line = i
                    break

            start_line = max(0, caret_line - self._height + 1)
            end_line = start_line + self._height
            if end_line > len(lines):
                end_line = len(lines)
                start_line = max(0, end_line - self._height)

            lines = lines[start_line:end_line]
            formatted = "\n".join(lines)

        formatted = formatted.replace("\x00", "")
        self.text.message = formatted
        self.text.update_alignment()

    def edit_mode(self):
        """Turn on edit mode."""
        if self.init:
            if self._message == "Enter Text":
                self._message = ""
            self.init = False
            self.caret_pos = len(self._message)
        self._has_focus = True
        UIContext.active_ui = self.text
        self.render_text()

    def blur_textbox(self, event=None):
        """
        Handle blur event for textbox.

        Parameters
        ----------
        event : PointerEvent
            The pointer event.
        """
        if self._has_focus:
            self._has_focus = False
            self.render_text(show_caret=False)
            self.on_blur(event)

    def left_button_press(self, event):
        """
        Handle left button press for textbox.

        Parameters
        ----------
        event : PointerEvent
            The pointer event.
        """
        if self._has_focus:
            UIContext.active_ui = None
            self.blur_textbox(event)
        else:
            self.edit_mode()

    def key_press(self, event):
        """
        Handle Key press for textbox.

        Parameters
        ----------
        event : KeyboardEvent
            The keyboard event.
        """
        key = event.key

        if key == "Shift":
            self._shift_pressed = True
        elif key == "CapsLock":
            self._caps_lock_on = not self._caps_lock_on

        key_char = key if key and len(key) == 1 else ""
        modifiers = list(getattr(event, "modifiers", []))
        if self._shift_pressed and "Shift" not in modifiers:
            modifiers.append("Shift")
        if self._caps_lock_on and "CapsLock" not in modifiers:
            modifiers.append("CapsLock")

        is_done = self.handle_character(key, key_char, modifiers)
        if is_done:
            self._has_focus = False
            UIContext.active_ui = None
            self.render_text(show_caret=False)
            self.on_blur(event)
            return

    def key_release(self, event):
        """
        Handle Key release for textbox.

        Parameters
        ----------
        event : KeyboardEvent
            The keyboard event.
        """
        key = getattr(event, "key", None)
        if key == "Shift":
            self._shift_pressed = False

    def wheel_scroll(self, event):
        """
        Handle mouse wheel event for textbox.

        Parameters
        ----------
        event : WheelEvent
            The wheel event.
        """
        if event.dy > 0:
            self.move_down()
        elif event.dy < 0:
            self.move_up()
        self.render_text()


class LineDoubleSlider2D(UI):
    """
    A 2D Line Slider with two sliding handles.

    Useful for setting min and max values for something.

    Parameters
    ----------
    position : (float, float), optional
        Absolute coordinates (x, y) of the lower-left corner of the slider.
    initial_values : (float, float), optional
        Initial values for the left and right handles respectively.
    min_value : float, optional
        Minimum value for the slider.
    max_value : float, optional
        Maximum value for the slider.
    length : int, optional
        Length of the slider track in pixels.
    line_width : int, optional
        Width of the line on which the handles will slide.
    inner_radius : int, optional
        Inner radius of the handles (when shape is 'disk').
    outer_radius : int, optional
        Outer radius of the handles (when shape is 'disk').
    handle_side : int, optional
        Length of the square handles (when shape is 'square').
    font_size : int, optional
        Size of the text font displaying the values.
    text_template : str or callable, optional
        Template for the text displaying the values. Can use {value} and {ratio}.
    orientation : str, optional
        Orientation of the slider ('horizontal' or 'vertical').
    shape : str, optional
        Shape of the handles ('disk' or 'square').
    z_order : int, optional
        Stacking order of the slider.
    """

    def __init__(
        self,
        *,
        position=(0, 0),
        initial_values=(0, 100),
        min_value=0,
        max_value=100,
        length=200,
        line_width=5,
        inner_radius=0,
        outer_radius=10,
        handle_side=20,
        font_size=16,
        text_template="{value:.1f}",
        orientation="horizontal",
        shape="disk",
        z_order=0,
    ):
        self.orientation = orientation.lower().strip()
        self._length = length
        self._line_width = line_width

        self.default_color = (1, 1, 1)
        self.active_color = (0, 0, 1)

        if min_value >= max_value:
            raise ValueError(
                f"min_value ({min_value}) must be less than max_value ({max_value})."
            )
        self._min_value = min_value
        self._max_value = max_value
        self.text_template = text_template

        self._handle_inner_radius = inner_radius
        self._handle_outer_radius = outer_radius
        self._handle_side = handle_side
        self._font_size = font_size
        self.shape = shape

        self.on_change = lambda ui: None
        self.on_value_changed = lambda ui: None
        self.on_moving_slider = lambda ui: None

        self._values = [np.clip(v, min_value, max_value) for v in initial_values]
        range_val = max_value - min_value
        self._ratios = [(v - min_value) / range_val for v in self._values]

        self.track = None
        self.handles = []
        self.texts = []
        super(LineDoubleSlider2D, self).__init__(position=position, z_order=z_order)

    def _setup(self):
        """Set up the internal actors for the slider."""
        track_size = (
            (self._length, self._line_width)
            if self.orientation == "horizontal"
            else (self._line_width, self._length)
        )
        self.track = Rectangle2D(size=track_size)
        self.track.color = (1, 0, 0)
        self.track.z_order = self.z_order

        for _ in range(2):
            if self.shape == "disk":
                handle = Disk2D(
                    outer_radius=self._handle_outer_radius,
                    inner_radius=self._handle_inner_radius,
                )
            elif self.shape == "square":
                handle = Rectangle2D(size=(self._handle_side, self._handle_side))
            else:
                raise ValueError("shape must be 'disk' or 'square'")

            handle.color = self.default_color
            handle.z_order = self.z_order + 1
            self.handles.append(handle)

            text = TextBlock2D(
                justification="center",
                vertical_justification="middle",
                dynamic_bbox=True,
                font_size=self._font_size,
            )
            text.z_order = self.z_order + 2
            self.texts.append(text)

        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.track.on_left_mouse_button_released = self.handle_release_callback

        self.handles[0].on_left_mouse_button_pressed = lambda e: self.handle_down(0)
        self.handles[1].on_left_mouse_button_pressed = lambda e: self.handle_down(1)

        self.handles[0].on_left_mouse_button_dragged = lambda e: (
            self.handle_move_callback(e, 0)
        )
        self.handles[1].on_left_mouse_button_dragged = lambda e: (
            self.handle_move_callback(e, 1)
        )

        self.handles[0].on_left_mouse_button_released = lambda e: (
            self.handle_release_callback(e, 0)
        )
        self.handles[1].on_left_mouse_button_released = lambda e: (
            self.handle_release_callback(e, 1)
        )

        self._active_handle = None

        self._children.extend(
            [self.track, self.handles[0], self.handles[1], self.texts[0], self.texts[1]]
        )

    def handle_down(self, idx):
        """
        Mark the specific handle as active.

        Parameters
        ----------
        idx : int
            Index of the handle (0 for left/bottom, 1 for right/top).
        """
        self._active_handle = idx

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            Empty list since FURY UI components act as children.
        """
        return []

    def _get_size(self):
        """
        Get the total size of the component.

        Returns
        -------
        numpy.ndarray
            The width and height of the component.
        """
        if self.orientation == "horizontal":
            width = self._length
            height = max(self._line_width, self.handles[0].size[1])
        else:
            width = max(self._line_width, self.handles[0].size[0])
            height = self._length
        return np.array([width, height])

    def _update_actors_position(self):
        """Update the position of the track and handle actors."""
        pos = self.get_position(x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP)
        self.track.set_position(
            pos + self.size / 2, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER
        )
        self._update_handle_positions()

    def format_text(self, idx):
        """
        Format the text for a specific handle.

        Parameters
        ----------
        idx : int
            Index of the handle (0 for left/bottom, 1 for right/top).

        Returns
        -------
        str
            The formatted text.
        """
        if callable(self.text_template):
            return self.text_template(self, idx)
        context = {"value": self._values[idx], "ratio": self._ratios[idx]}
        return self.text_template.format(**context)

    def _update_handle_positions(self):
        """Update the physical positions of the handles and text labels."""
        track_origin = self.track.get_position(
            x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP
        )
        for i in range(2):
            if self.orientation == "horizontal":
                offset = self._ratios[i] * self._length
                handle_center = track_origin + np.array([offset, self._line_width / 2])
                text_pos = handle_center + np.array(
                    [0, -(self.handles[i].size[1] + self.texts[i].size[1] / 2)]
                )
            else:
                offset = self._ratios[i] * self._length
                handle_center = track_origin + np.array([self._line_width / 2, offset])
                text_pos = handle_center + np.array(
                    [self.handles[i].size[0] + self.texts[i].size[0] / 2, 0]
                )

            self.handles[i].set_position(
                handle_center, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER
            )
            self.texts[i].set_position(
                text_pos, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER
            )
            self.texts[i].message = self.format_text(i)

    def track_click_callback(self, event):
        """
        Handle mouse click events on the slider track.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        left_x = self.track.get_position(x_anchor=Anchor.LEFT)[0]
        bottom_y = self.track.get_position(y_anchor=Anchor.BOTTOM)[1]
        top_y = self.track.get_position(y_anchor=Anchor.TOP)[1]

        if self.orientation == "horizontal":
            ratio = (event.x - left_x) / self._length
        else:
            total_dist = bottom_y - top_y
            ratio = (event.y - top_y) / total_dist if total_dist != 0 else 0

        dist0 = abs(ratio - self._ratios[0])
        dist1 = abs(ratio - self._ratios[1])

        idx = 0 if dist0 < dist1 else 1
        self.handle_move_callback(event, idx)

    def handle_move_callback(self, event, idx=None):
        """
        Handle mouse drag events to update the slider state.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        idx : int, optional
            Index of the handle being moved. If None, uses the active handle.
        """
        if idx is None:
            if self._active_handle is not None:
                idx = self._active_handle
            else:
                return

        self.handles[idx].color = self.active_color

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

        new_ratio = np.clip(new_ratio, 0, 1)

        if idx == 0 and new_ratio > self._ratios[1]:
            new_ratio = self._ratios[1]
        elif idx == 1 and new_ratio < self._ratios[0]:
            new_ratio = self._ratios[0]

        self._ratios[idx] = new_ratio
        self._values[idx] = self.min_value + new_ratio * (
            self.max_value - self.min_value
        )

        self.on_moving_slider(self)
        self._update_actors_position()

    def handle_release_callback(self, event, idx=None):
        """
        Handle the release of the mouse button.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        idx : int, optional
            Index of the handle being released.
        """
        if idx is not None:
            self.handles[idx].color = self.default_color
        else:
            self.handles[0].color = self.default_color
            self.handles[1].color = self.default_color
        self._active_handle = None

    @property
    def min_value(self):
        """
        Get the minimum value of the slider.

        Returns
        -------
        float
            The minimum value.
        """
        return self._min_value

    @min_value.setter
    def min_value(self, val):
        """
        Set the minimum value of the slider.

        Parameters
        ----------
        val : float
            The minimum value.
        """
        if val >= self._max_value:
            raise ValueError(
                f"min_value ({val}) must be less than max_value ({self._max_value})."
            )
        self._min_value = val

    @property
    def max_value(self):
        """
        Get the maximum value of the slider.

        Returns
        -------
        float
            The maximum value.
        """
        return self._max_value

    @max_value.setter
    def max_value(self, val):
        """
        Set the maximum value of the slider.

        Parameters
        ----------
        val : float
            The maximum value.
        """
        if val <= self._min_value:
            raise ValueError(
                f"max_value ({val}) must be greater than min_value ({self._min_value})."
            )
        self._max_value = val

    @property
    def left_disk_value(self):
        """
        Get the value of the left/bottom handle.

        Returns
        -------
        float
            The current value.
        """
        return self._values[0]

    @left_disk_value.setter
    def left_disk_value(self, val):
        """
        Set the value of the left/bottom handle.

        Parameters
        ----------
        val : float
            The new value.
        """
        val = np.clip(val, self.min_value, self.max_value)
        self._values[0] = val
        range_val = self.max_value - self.min_value
        self._ratios[0] = (val - self.min_value) / range_val if range_val != 0 else 0
        self.on_moving_slider(self)
        self._update_actors_position()

    @property
    def right_disk_value(self):
        """
        Get the value of the right/top handle.

        Returns
        -------
        float
            The current value.
        """
        return self._values[1]

    @right_disk_value.setter
    def right_disk_value(self, val):
        """
        Set the value of the right/top handle.

        Parameters
        ----------
        val : float
            The new value.
        """
        val = np.clip(val, self.min_value, self.max_value)
        self._values[1] = val
        range_val = self.max_value - self.min_value
        self._ratios[1] = (val - self.min_value) / range_val if range_val != 0 else 0
        self.on_moving_slider(self)
        self._update_actors_position()


class RingSlider2D(Slider2D):
    """
    A disk slider.

    A disk moves along the boundary of a ring.
    Goes from 0-360 degrees.

    Parameters
    ----------
    center : (float, float), optional
        Position (x, y) of the slider's center.
    initial_value : float, optional
        Initial value of the slider.
    min_value : float, optional
        Minimum value of the slider.
    max_value : float, optional
        Maximum value of the slider.
    slider_inner_radius : int, optional
        Inner radius of the base disk.
    slider_outer_radius : int, optional
        Outer radius of the base disk.
    handle_inner_radius : int, optional
        Inner radius of the slider's handle.
    handle_outer_radius : int, optional
        Outer radius of the slider's handle.
    handle_side : int, optional
        The side length of the square handle when shape="square".
    font_size : int, optional
        Size of the text to display alongside the slider (pt).
    text_template : str or callable, optional
        If str, text template can contain one or multiple of the
        replacement fields: `{value:}`, `{ratio:}`, `{angle:}`.
        If callable, this instance of `:class:RingSlider2D` will be
        passed as argument to the text template function.
    shape : str, optional
        The handle shape. Supported values are "disk" and "square".
    z_order : int, optional
            Stacking priority of the slider. The handle and text
            are placed above the track.

    Attributes
    ----------
    track : :class:`Disk2D`
        The circle on which the slider's handle moves.
    handle : :class:`Disk2D`
        The moving part of the slider.
    text : :class:`TextBlock2D`
        The text that shows percentage.
    default_color : str, tuple, list or ndarray
        Color of the handle when in unpressed state. A hex string
        ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
    active_color : str, tuple, list or ndarray
        Color of the handle when it is pressed, same formats as
        ``default_color``.
    """

    def __init__(
        self,
        *,
        center=(0, 0),
        initial_value=0,
        min_value=0,
        max_value=360,
        slider_inner_radius=40,
        slider_outer_radius=44,
        handle_inner_radius=0,
        handle_outer_radius=10,
        handle_side=20,
        font_size=16,
        text_template="{ratio:.0%}",
        shape="disk",
        z_order=0,
    ):
        """
        Init this UI element.

        Parameters
        ----------
        center : (float, float), optional
            Position (x, y) of the slider's center.
        initial_value : float, optional
            Initial value of the slider.
        min_value : float, optional
            Minimum value of the slider.
        max_value : float, optional
            Maximum value of the slider.
        slider_inner_radius : int, optional
            Inner radius of the base disk.
        slider_outer_radius : int, optional
            Outer radius of the base disk.
        handle_inner_radius : int, optional
            Inner radius of the slider's handle.
        handle_outer_radius : int, optional
            Outer radius of the slider's handle.
        handle_side : int, optional
            The side length of the square handle when shape="square".
        font_size : int, optional
            Size of the text to display alongside the slider (pt).
        text_template : str or callable, optional
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`, `{angle:}`.
            If callable, this instance of `:class:RingSlider2D` will be
            passed as argument to the text template function.
        shape : str, optional
            The handle shape. Supported values are "disk" and "square".
        z_order : int, optional
            Stacking priority of the slider. The handle and text
            are placed above the track.
        """
        self._track_inner_radius = slider_inner_radius
        self._track_outer_radius = slider_outer_radius
        self._angle = 0.0

        super(RingSlider2D, self).__init__(
            position=center,
            initial_value=initial_value,
            min_value=min_value,
            max_value=max_value,
            handle_inner_radius=handle_inner_radius,
            handle_outer_radius=handle_outer_radius,
            handle_side=handle_side,
            font_size=font_size,
            text_template=text_template,
            shape=shape,
            z_order=z_order,
        )

        self.value = initial_value

    def _setup(self):
        """
        Setup this UI component.

        Create the slider's circle (Disk2D), the handle (Disk2D) and the
        text (TextBlock2D).
        """
        super(RingSlider2D, self)._setup()
        self.track = Disk2D(
            outer_radius=self._track_outer_radius,
            inner_radius=self._track_inner_radius,
        )
        self.track.color = (1, 0, 0)
        self.track.z_order = self.z_order

        self.handle.color = self.default_color

        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.track.on_left_mouse_button_released = self.handle_release_callback

        self.handle.on_left_mouse_button_dragged = self.handle_move_callback
        self.handle.on_left_mouse_button_released = self.handle_release_callback

        self._children.append(self.track)

    def _get_size(self):
        """
        Get the size of this UI component.

        Returns
        -------
        ndarray
            The size of the component.
        """
        diameter = 2 * (self._track_outer_radius + self._handle_outer_radius)
        return np.array([diameter, diameter])

    def _update_actors_position(self):
        """Update the position of the internal actors."""
        center = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)
        self.track.set_position(center, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)
        self._update_handle_position()

    @property
    def mid_track_radius(self):
        """
        Return the distance from the center of the slider to the track middle.

        Returns
        -------
        float
            The mid track radius.
        """
        return (self.track.inner_radius + self.track.outer_radius) / 2.0

    def _update_handle_position(self):
        """
        Place the handle and the text according to the current angle / ratio.
        """
        center = self.track.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)
        angle = self.angle
        x = self.mid_track_radius * np.sin(angle) + center[0]
        y = center[1] - self.mid_track_radius * np.cos(angle)
        self.handle.set_position((x, y), x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)

        self.text.message = self.format_text()
        self.text.set_position(center, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)

    @property
    def angle(self):
        """
        Return Angle (in rad) the handle makes with the y-axis.

        Returns
        -------
        float
            The angle.
        """
        angle = self.ratio * TWO_PI
        if np.isclose(angle, TWO_PI):
            angle = 0.0
        return angle

    def handle_move_callback(self, event):
        """
        Handle mouse drag events to update the slider state.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self.handle.color = self.active_color
        center = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)
        x, y = event.x - center[0], center[1] - event.y
        angle = np.arctan2(x, y) % TWO_PI
        ratio = angle / TWO_PI
        if np.isclose(ratio, 1.0):
            ratio = 0.0
            angle = 0.0

        self._angle = angle
        self.ratio = ratio
        self.on_moving_slider(self)


class RangeSlider(UI):
    """
    A compound UI element containing a LineSlider2D and a LineDoubleSlider2D.

    The double slider is used to set the minimum and maximum value bounds
    for the single value slider.

    Parameters
    ----------
    line_width : int, optional
        Width of the line on which the handles will slide.
    inner_radius : int, optional
        Inner radius of the handles (when shape is 'disk').
    outer_radius : int, optional
        Outer radius of the handles (when shape is 'disk').
    handle_side : int, optional
        Length of the square handles (when shape is 'square').
    range_slider_center : (float, float), optional
        Position of the LineDoubleSlider2D object.
    value_slider_center : (float, float), optional
        Position of the LineSlider2D object.
    length : int, optional
        Length of both sliders in pixels.
    min_value : float, optional
        Minimum value for the range slider.
    max_value : float, optional
        Maximum value for the range slider.
    font_size : int, optional
        Size of the text font displaying the values.
    range_precision : int, optional
        Number of decimal places to show on the range slider text.
    orientation : str, optional
        Orientation of the sliders ('horizontal' or 'vertical').
    value_precision : int, optional
        Number of decimal places to show on the value slider text.
    shape : str, optional
        Shape of the handles ('disk' or 'square').
    """

    def __init__(
        self,
        *,
        line_width=5,
        inner_radius=0,
        outer_radius=10,
        handle_side=20,
        range_slider_center=(450, 400),
        value_slider_center=(450, 300),
        length=200,
        min_value=0,
        max_value=100,
        font_size=16,
        range_precision=1,
        orientation="horizontal",
        value_precision=2,
        shape="disk",
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.handle_side = handle_side
        self.length = length
        self.line_width = line_width
        self.font_size = font_size
        self.shape = shape
        self.orientation = orientation.lower()

        self.range_slider_text_template = "{value:." + str(range_precision) + "f}"
        self.value_slider_text_template = "{value:." + str(value_precision) + "f}"

        self.range_slider_center = range_slider_center
        self.value_slider_center = value_slider_center

        pos_x = min(range_slider_center[0], value_slider_center[0])
        pos_y = min(range_slider_center[1], value_slider_center[1])
        super(RangeSlider, self).__init__(position=(pos_x, pos_y))

    def _setup(self):
        """Setup the internal range and value sliders."""
        self.range_slider = LineDoubleSlider2D(
            line_width=self.line_width,
            inner_radius=self.inner_radius,
            outer_radius=self.outer_radius,
            handle_side=self.handle_side,
            position=self.range_slider_center,
            length=self.length,
            min_value=self.min_value,
            max_value=self.max_value,
            initial_values=(self.min_value, self.max_value),
            font_size=self.font_size,
            shape=self.shape,
            orientation=self.orientation,
            text_template=self.range_slider_text_template,
        )

        self.value_slider = LineSlider2D(
            line_width=self.line_width,
            length=self.length,
            inner_radius=self.inner_radius,
            outer_radius=self.outer_radius,
            handle_side=self.handle_side,
            position=self.value_slider_center,
            min_value=self.min_value,
            max_value=self.max_value,
            initial_value=(self.min_value + self.max_value) / 2,
            font_size=self.font_size,
            shape=self.shape,
            orientation=self.orientation,
            text_template=self.value_slider_text_template,
        )

        self.range_slider.on_moving_slider = self.range_slider_handle_move_callback
        self._children.extend([self.range_slider, self.value_slider])

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            Empty list since FURY UI components act as children.
        """
        return []

    def _get_size(self):
        """
        Get the total size of the component.

        Returns
        -------
        numpy.ndarray
            The width and height of the component.
        """
        if self.orientation == "horizontal":
            w = max(self.range_slider.size[0], self.value_slider.size[0])
            h = self.range_slider.size[1] + self.value_slider.size[1]
        else:
            w = self.range_slider.size[0] + self.value_slider.size[0]
            h = max(self.range_slider.size[1], self.value_slider.size[1])
        return np.array([w, h])

    def _update_actors_position(self):
        """Update the position of the internal sliders."""
        self.range_slider.set_position(self.range_slider_center)
        self.value_slider.set_position(self.value_slider_center)

    def range_slider_handle_move_callback(self, ui):
        """
        Handle updates to the range bounds.

        Parameters
        ----------
        ui : UI
            The UI component triggering the callback.
        """
        self.value_slider.min_value = self.range_slider.left_disk_value
        self.value_slider.max_value = self.range_slider.right_disk_value


# class Option(UI):
#     """A set of a Button2D and a TextBlock2D to act as a single option
#     for checkboxes and radio buttons.
#     Clicking the button toggles its checked/unchecked status.

#     Attributes
#     ----------
#     label : str
#         The label for the option.
#     font_size : int
#             Font Size of the label.

#     """

#     @warn_on_args_to_kwargs()
#     def __init__(self, label, *, position=(0, 0), font_size=18, checked=False):
#         """Init this class instance.

#         Parameters
#         ----------
#         label : str
#             Text to be displayed next to the option's button.
#         position : (float, float)
#             Absolute coordinates (x, y) of the lower-left corner of
#             the button of the option.
#         font_size : int
#             Font size of the label.
#         checked : bool, optional
#             Boolean value indicates the initial state of the option

#         """
#         self.label = label
#         self.font_size = font_size
#         self.checked = checked
#         self.button_size = (font_size * 1.2, font_size * 1.2)
#         self.button_label_gap = 10
#         super(Option, self).__init__(position=position)

#         # Offer some standard hooks to the user.
#         self.on_change = lambda obj: None

#     def _setup(self):
#         """Setup this UI component."""
#         # Option's button
#         self.button_icons = []
#         self.button_icons.append(("unchecked", read_viz_icons(fname="stop2.png")))
#         self.button_icons.append(("checked", read_viz_icons(fname="checkmark.png")))
#         self.button = Button2D(icon_fnames=self.button_icons, size=self.button_size)

#         self.text = TextBlock2D(text=self.label, font_size=self.font_size)

#         # Display initial state
#         if self.checked:
#             self.button.set_icon_by_name("checked")

#         # Add callbacks
#         self.button.on_left_mouse_button_clicked = self.toggle
#         self.text.on_left_mouse_button_clicked = self.toggle

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         return self.button.actors + self.text.actors

#     def _add_to_scene(self, scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : scene

#         """
#         self.button.add_to_scene(scene)
#         self.text.add_to_scene(scene)

#     def _get_size(self):
#         width = self.button.size[0] + self.button_label_gap + self.text.size[0]
#         height = max(self.button.size[1], self.text.size[1])
#         return np.array([width, height])

#     def _set_position(self, coords):
#         """Set the lower-left corner position of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         num_newlines = self.label.count("\n")
#         self.button.position = coords + (0, num_newlines * self.font_size * 0.5)
#         offset = (self.button.size[0] + self.button_label_gap, 0)
#         self.text.position = coords + offset

#     def toggle(self, i_ren, _obj, _element):
#         if self.checked:
#             self.deselect()
#         else:
#             self.select()

#         self.on_change(self)
#         i_ren.force_render()

#     def select(self):
#         self.checked = True
#         self.button.set_icon_by_name("checked")

#     def deselect(self):
#         self.checked = False
#         self.button.set_icon_by_name("unchecked")


# class Checkbox(UI):
#     """A 2D set of :class:'Option' objects.
#     Multiple options can be selected.

#     Attributes
#     ----------
#     labels : list(string)
#         List of labels of each option.
#     options : dict(Option)
#         Dictionary of all the options in the checkbox set.
#     padding : float
#         Distance between two adjacent options

#     """

#     @warn_on_args_to_kwargs()
#     def __init__(
#         self,
#         labels,
#         *,
#         checked_labels=(),
#         padding=1,
#         font_size=18,
#         font_family="Arial",
#         position=(0, 0),
#     ):
#         """Init this class instance.

#         Parameters
#         ----------
#         labels : list(str)
#             List of labels of each option.
#         checked_labels: list(str), optional
#             List of labels that are checked on setting up.
#         padding : float, optional
#             The distance between two adjacent options
#         font_size : int, optional
#             Size of the text font.
#         font_family : str, optional
#             Currently only supports Arial.
#         position : (float, float), optional
#             Absolute coordinates (x, y) of the lower-left corner of
#             the button of the first option.

#         """
#         self.labels = list(reversed(list(labels)))
#         self._padding = padding
#         self._font_size = font_size
#         self.font_family = font_family
#         self.checked_labels = list(checked_labels)
#         super(Checkbox, self).__init__(position=position)
#         self.on_change = lambda checkbox: None

#     def _setup(self):
#         """Setup this UI component."""
#         self.options = OrderedDict()
#         button_y = self.position[1]
#         for label in self.labels:
#             option = Option(
#                 label=label,
#                 font_size=self.font_size,
#                 position=(self.position[0], button_y),
#                 checked=(label in self.checked_labels),
#             )

#             line_spacing = option.text.actor.GetTextProperty().GetLineSpacing()
#             button_y = (
#                 button_y
#                 + self.font_size * (label.count("\n") + 1) * (line_spacing + 0.1)
#                 + self.padding
#             )
#             self.options[label] = option

#             # Set callback
#             option.on_change = self._handle_option_change

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         actors = []
#         for option in self.options.values():
#             actors = actors + option.actors
#         return actors

#     def _add_to_scene(self, scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : scene

#         """
#         for option in self.options.values():
#             option.add_to_scene(scene)

#     def _get_size(self):
#         option_width, option_height = self.options.values()[0].get_size()
#         height = len(self.labels) * (option_height + self.padding) - self.padding
#         return np.asarray([option_width, height])

#     def _handle_option_change(self, option):
#         """Update whenever an option changes.

#         Parameters
#         ----------
#         option : :class:`Option`

#         """
#         if option.checked:
#             self.checked_labels.append(option.label)
#         else:
#             self.checked_labels.remove(option.label)

#         self.on_change(self)

#     def _set_position(self, coords):
#         """Set the lower-left corner position of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         button_y = coords[1]
#         for option_no, option in enumerate(self.options.values()):
#             option.position = (coords[0], button_y)
#             line_spacing = option.text.actor.GetTextProperty().GetLineSpacing()
#             button_y = (
#                 button_y
#                 + self.font_size
#                 * (self.labels[option_no].count("\n") + 1)
#                 * (line_spacing + 0.1)
#                 + self.padding
#             )

#     @property
#     def font_size(self):
#         """Gets the font size of text."""
#         return self._font_size

#     @property
#     def padding(self):
#         """Get the padding between options."""
#         return self._padding


# class RadioButton(Checkbox):
#     """A 2D set of :class:'Option' objects.
#     Only one option can be selected.

#     Attributes
#     ----------
#     labels : list(string)
#         List of labels of each option.
#     options : dict(Option)
#         Dictionary of all the options in the checkbox set.
#     padding : float
#         Distance between two adjacent options

#     """

#     @warn_on_args_to_kwargs()
#     def __init__(
#         self,
#         labels,
#         checked_labels,
#         *,
#         padding=1,
#         font_size=18,
#         font_family="Arial",
#         position=(0, 0),
#     ):
#         """Init class instance.

#         Parameters
#         ----------
#         labels : list(str)
#             List of labels of each option.
#         checked_labels: list(str), optional
#             List of labels that are checked on setting up.
#         padding : float, optional
#             The distance between two adjacent options
#         font_size : int, optional
#             Size of the text font.
#         font_family : str, optional
#             Currently only supports Arial.
#         position : (float, float), optional
#             Absolute coordinates (x, y) of the lower-left corner of
#             the button of the first option.

#         """
#         if len(checked_labels) > 1:
#             err_msg = "Only one option can be preselected for radio buttons."
#             raise ValueError(err_msg)

#         super(RadioButton, self).__init__(
#             labels=labels,
#             position=position,
#             padding=padding,
#             font_size=font_size,
#             font_family=font_family,
#             checked_labels=checked_labels,
#         )

#     def _handle_option_change(self, option):
#         for option_ in self.options.values():
#             option_.deselect()

#         option.select()
#         self.checked_labels = [option.label]
#         self.on_change(self)


class ButtonGroup(UI):
    """
    Base class for a group of labeled toggle buttons.

    Each entry pairs a :class:`TexturedButton2D` (used as a toggle) with a
    :class:`TextBlock2D` label. Entries are laid out either vertically or
    horizontally. This class is not meant to be used directly; it provides
    the shared machinery for the :class:`Checkbox` and :class:`RadioButton`
    components, which subclass it to define their selection semantics.

    Parameters
    ----------
    labels : list of str
        Text label for each option.
    checked_labels : list of str, optional
        Labels that should be toggled on initially.
    button_states : dict, optional
        Mapping of state names to icon file paths for the toggle button.
        The ``"default"`` icon is shown when an option is off and the
        ``"pressed"`` icon when it is on. Defaults to checkbox-style icons.
    orientation : str, optional
        Layout direction of the options: ``"vertical"`` or ``"horizontal"``.
    padding : float, optional
        Spacing in pixels between two adjacent options.
    font_size : int, optional
        Font size of the labels in pixels.
    font_family : str, optional
        Font family of the labels. Currently only supports "Arial".
    text_color : str, tuple, list or ndarray, optional
        Color of the label text. Accepts a hex string ("#FF0000"), RGB(A) in
        [0, 1], or RGB(A) in [0, 255].
    position : (float, float), optional
        Absolute coordinates (x, y) of the top-left corner of this component.
    z_order : int, optional
        Z-order of the UI component.

    Attributes
    ----------
    labels : list of str
        Ordered list of option labels.
    options : dict
        Mapping of label to its ``(button, text)`` pair.
    """

    def __init__(
        self,
        labels,
        *,
        checked_labels=(),
        button_states=None,
        orientation="vertical",
        padding=10,
        font_size=18,
        font_family="Arial",
        text_color=(1, 1, 1),
        position=(0, 0),
        z_order=0,
    ):
        """Init class instance."""
        if orientation not in ("vertical", "horizontal"):
            raise ValueError(
                f"orientation should be 'vertical' or 'horizontal', "
                f"got {orientation!r}."
            )

        self.labels = list(labels)
        self._checked_labels = list(checked_labels)
        self.orientation = orientation
        self._padding = padding
        self._font_size = font_size
        self.font_family = font_family
        self.text_color = text_color

        self.button_states = button_states or {
            "default": read_viz_icons(fname="stop2.png"),
            "pressed": read_viz_icons(fname="checkmark.png"),
        }

        # Size of the toggle button, scaled to the label font size.
        self.button_size = (int(font_size * 1.2), int(font_size * 1.2))
        # Gap in pixels between a button and its label.
        self.button_label_gap = 8

        super(ButtonGroup, self).__init__(position=position, z_order=z_order)

        # User hook: called with this group whenever a selection changes.
        self.on_change = lambda group: None

    def _setup(self):
        """Set up this UI component."""
        self.options = {}
        self._label_by_button = {}

        for label in self.labels:
            button = TexturedButton2D(
                states=self.button_states,
                size=self.button_size,
                is_toggle=True,
            )
            text = TextBlock2D(
                text=label,
                font_size=self.font_size,
                font_family=self.font_family,
                color=self.text_color,
                dynamic_bbox=True,
            )

            if label in self._checked_labels:
                button.toggled = True

            # Clicking either the button or its label toggles the option.
            button.on_clicked = self._handle_button_clicked
            text.on_left_mouse_button_clicked = lambda _event, b=button: b.do_click()

            self.options[label] = (button, text)
            self._label_by_button[id(button)] = label
            self._children.extend([button, text])

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            An empty list; rendering is delegated to the child components.
        """
        return []

    def _get_size(self):
        """
        Get the size of the UI component.

        Returns
        -------
        (int, int)
            Width and height of the UI component in pixels.
        """
        along = 0
        cross = 0
        for index, label in enumerate(self.labels):
            button, text = self.options[label]
            row_w = button.size[0] + self.button_label_gap + text.size[0]
            row_h = max(button.size[1], text.size[1])

            if self.orientation == "horizontal":
                along += row_w + (self.padding if index else 0)
                cross = max(cross, row_h)
            else:
                along += row_h + (self.padding if index else 0)
                cross = max(cross, row_w)

        if self.orientation == "horizontal":
            return (along, cross)
        return (cross, along)

    def _update_actors_position(self):
        """Update the position of the internal actors."""
        origin = self.get_position()
        cursor = np.array(origin, dtype=float)

        for label in self.labels:
            button, text = self.options[label]
            b_w, b_h = button.size
            t_w, t_h = text.size
            row_h = max(b_h, t_h)

            # Vertically center the button and label within the row.
            button.set_position((cursor[0], cursor[1] + (row_h - b_h) / 2))
            text.set_position(
                (
                    cursor[0] + b_w + self.button_label_gap,
                    cursor[1] + (row_h - t_h) / 2,
                )
            )

            if self.orientation == "horizontal":
                cursor[0] += b_w + self.button_label_gap + t_w + self.padding
            else:
                cursor[1] += row_h + self.padding

    def _handle_button_clicked(self, button):
        """
        Handle a click on one of the option buttons.

        Parameters
        ----------
        button : TexturedButton2D
            The button that was clicked. Its ``toggled`` state has already
            been updated by the time this is called.
        """
        label = self._label_by_button[id(button)]
        self._handle_option_change(label)
        self.on_change(self)

    def _handle_option_change(self, label):
        """
        Update the checked state after an option is toggled.

        Subclasses override this to enforce their selection semantics (e.g.
        a radio button clears the other options). The base implementation
        keeps :attr:`checked_labels` in sync with the buttons' toggled state,
        allowing multiple options to be checked at once.

        Parameters
        ----------
        label : str
            The label of the option that changed.
        """
        button, _ = self.options[label]
        if button.toggled and label not in self._checked_labels:
            self._checked_labels.append(label)
        elif not button.toggled and label in self._checked_labels:
            self._checked_labels.remove(label)

    def select(self, label):
        """
        Toggle the option on.

        Parameters
        ----------
        label : str
            The label of the option to select.
        """
        button, _ = self.options[label]
        button.toggled = True
        if label not in self._checked_labels:
            self._checked_labels.append(label)

    def deselect(self, label):
        """
        Toggle the option off.

        Parameters
        ----------
        label : str
            The label of the option to deselect.
        """
        button, _ = self.options[label]
        button.toggled = False
        if label in self._checked_labels:
            self._checked_labels.remove(label)

    @property
    def checked_labels(self):
        """
        Get the labels of the currently checked options.

        Returns
        -------
        list of str
            Labels of the options that are currently toggled on.
        """
        return list(self._checked_labels)

    @property
    def font_size(self):
        """
        Get the font size of the labels.

        Returns
        -------
        int
            Font size of the labels in pixels.
        """
        return self._font_size

    @property
    def padding(self):
        """
        Get the padding between options.

        Returns
        -------
        float
            Spacing in pixels between two adjacent options.
        """
        return self._padding


class ComboBox2D(UI):
    """
    UI element to create drop-down menus.

    Parameters
    ----------
    items : list of str, optional
        List of items to be displayed as choices.
    position : tuple of 2 floats, optional
        Absolute coordinates (x, y) of the lower-left corner of this
        UI component.
    size : tuple of 2 ints, optional
        Width and height in pixels of this UI component.
    placeholder : str, optional
        Holds the default text to be displayed.
    draggable : bool, optional
        Whether the UI element is draggable or not.
    selection_text_color : str, tuple, list or ndarray, optional
        Color of the selected text to be displayed. All color parameters
        accept a hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in
        [0, 255].
    selection_bg_color : str, tuple, list or ndarray, optional
        Background color of the selection text.
    menu_text_color : str, tuple, list or ndarray, optional
        Color of the options displayed in drop down menu.
    selected_color : str, tuple, list or ndarray, optional
        Background color of the selected option in drop down menu.
    unselected_color : str, tuple, list or ndarray, optional
        Background color of the unselected option in drop down menu.
    scroll_bar_active_color : str, tuple, list or ndarray, optional
        Color of the scrollbar when in active use.
    scroll_bar_inactive_color : str, tuple, list or ndarray, optional
        Color of the scrollbar when inactive.
    menu_opacity : float, optional
        Opacity of the drop down menu background.
    reverse_scrolling : bool, optional
        If True, scrolling up will move the list of files down.
    font_size : int, optional
        The font size of selected text in pixels.
    line_spacing : float, optional
        Distance between drop down menu's items in pixels.
    z_order : int, optional
        Z-order of the UI component.

    Attributes
    ----------
    selection_box : TextBox2D
        Display selection and placeholder text.
    drop_down_button : Button2D
        Button to show or hide menu.
    drop_down_menu : ListBox2D
        Container for item list.
    """

    def __init__(
        self,
        *,
        items=None,
        position=(0, 0),
        size=(300, 200),
        placeholder="Choose selection...",
        draggable=True,
        selection_text_color=(0, 0, 0),
        selection_bg_color=(1, 1, 1),
        menu_text_color=(0.2, 0.2, 0.2),
        selected_color=(0.9, 0.6, 0.6),
        unselected_color=(0.6, 0.6, 0.6),
        scroll_bar_active_color=(0.6, 0.2, 0.2),
        scroll_bar_inactive_color=(0.9, 0.0, 0.0),
        menu_opacity=1.0,
        reverse_scrolling=False,
        font_size=20,
        line_spacing=1.4,
        z_order=0,
    ):
        """Init class instance."""
        if items is None:
            items = []

        self.items = items.copy()
        self.font_size = font_size
        self.reverse_scrolling = reverse_scrolling
        self.line_spacing = line_spacing
        self.panel_size = size
        self._selection = placeholder
        self._menu_visibility = False
        self._selection_ID = None
        self._drag_offset = None
        self.draggable = draggable
        self.sel_text_color = selection_text_color
        self.sel_bg_color = selection_bg_color
        self.menu_txt_color = menu_text_color
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        self.scroll_active_color = scroll_bar_active_color
        self.scroll_inactive_color = scroll_bar_inactive_color
        self.menu_opacity = menu_opacity

        (
            self.text_block_size,
            self.drop_menu_size,
            self.drop_button_size,
            self._sel_pos,
            self._btn_pos,
            self._menu_pos,
        ) = self._calculate_layout(size)

        self._icon_files = {
            "default": read_viz_icons(fname="circle-down.png"),
            "pressed": read_viz_icons(fname="circle-up.png"),
        }

        super(ComboBox2D, self).__init__(position=position, z_order=z_order)

    def _setup(self):
        """
        Setup this UI component.
        """
        self.selection_box = TextBlock2D(
            size=self.text_block_size,
            color=self.sel_text_color,
            bg_color=self.sel_bg_color,
            text=self._selection,
            font_size=self.font_size,
            justification="center",
            vertical_justification="middle",
            bold=True,
        )

        self.drop_down_button = TexturedButton2D(
            states=self._icon_files, size=self.drop_button_size, is_toggle=True
        )

        self.drop_down_menu = ListBox2D(
            values=self.items,
            multiselection=False,
            font_size=self.font_size,
            line_spacing=self.line_spacing,
            text_color=self.menu_txt_color,
            selected_color=self.selected_color,
            unselected_color=self.unselected_color,
            scroll_bar_active_color=self.scroll_active_color,
            scroll_bar_inactive_color=self.scroll_inactive_color,
            background_opacity=self.menu_opacity,
            reverse_scrolling=self.reverse_scrolling,
            size=self.drop_menu_size,
        )

        self.drop_down_menu.set_visibility(False)

        self.drop_down_menu.panel.background.on_left_mouse_button_pressed = lambda e: (
            None
        )
        self.drop_down_menu.panel.background.on_left_mouse_button_dragged = lambda e: (
            None
        )

        self.panel = Panel2D(self.panel_size, opacity=0.0)
        self.panel.add_element(self.selection_box, self._sel_pos)
        self.panel.add_element(self.drop_down_button, self._btn_pos)
        self.panel.add_element(self.drop_down_menu, self._menu_pos)

        self._setup_drag_events()

        self.drop_down_menu.on_change = self.select_option_callback
        self.drop_down_button.on_clicked = self.menu_toggle_callback

        self.on_change = lambda ui: None

        self._children.extend([self.panel])

    def _setup_drag_events(self):
        """Attach drag event handlers to all interactive surfaces."""
        if self.draggable:
            drag_targets = [
                self.selection_box,
                self.selection_box.background,
            ]

            for target in drag_targets:
                target.on_left_mouse_button_dragged = self.left_button_dragged
                target.on_left_mouse_button_pressed = self.left_button_pressed
        else:
            self.panel.background.on_left_mouse_button_dragged = lambda event: None

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            A list of actors.
        """
        return []

    def _calculate_layout(self, size):
        """
        Calculate subcomponent sizes and positions for the given overall size.

        Parameters
        ----------
        size : tuple of 2 ints
            ComboBox size (width, height) in pixels.

        Returns
        -------
        tuple
            Tuple of (text_block_size, drop_menu_size, drop_button_size,
            sel_pos, btn_pos, menu_pos).
        """
        text_block_size = (int(0.85 * size[0]), int(0.2 * size[1]))
        drop_menu_size = (size[0], int(0.8 * size[1]))
        drop_button_size = (int(0.15 * size[0]), int(0.2 * size[1]))

        sel_pos = (0.0, 0.0)
        btn_pos = (0.85, 0.0)
        menu_pos = (0.0, 0.2)

        return (
            text_block_size,
            drop_menu_size,
            drop_button_size,
            sel_pos,
            btn_pos,
            menu_pos,
        )

    def resize(self, size):
        """
        Resize ComboBox2D.

        Parameters
        ----------
        size : tuple of 2 ints
            ComboBox size(width, height) in pixels.
        """
        ratio = size[1] / self.panel_size[1]
        self.font_size = max(1, int(self.font_size * ratio))
        self.panel_size = size

        self.panel.resize(size)

        (
            self.text_block_size,
            self.drop_menu_size,
            self.drop_button_size,
            sel_pos,
            btn_pos,
            menu_pos,
        ) = self._calculate_layout(size)

        self.panel.update_element(self.selection_box, sel_pos)
        self.panel.update_element(self.drop_down_button, btn_pos)
        self.panel.update_element(self.drop_down_menu, menu_pos)

        self.drop_down_button.resize(self.drop_button_size)

        self.drop_down_menu.font_size = self.font_size
        self.drop_down_menu.slot_height = max(
            1, int(self.font_size * self.drop_down_menu.line_spacing)
        )
        self.drop_down_menu.resize(self.drop_menu_size)

        self.selection_box.font_size = self.font_size
        self.selection_box.resize(self.text_block_size)

        if not self._menu_visibility:
            self.drop_down_menu.set_visibility(False)

    def _update_actors_position(self):
        """Update the position of the actors."""
        pos = self.get_position()
        self.panel.set_position((pos[0], pos[1] - self.drop_menu_size[1]))

    def _get_size(self):
        """
        Get the size of the UI component.

        Returns
        -------
        tuple of 2 ints
            Size of the UI component.
        """
        return self.panel.size

    @property
    def selected_text(self):
        """
        Get the currently selected text.

        Returns
        -------
        str
            Currently selected text.
        """
        return self._selection

    @property
    def selected_text_index(self):
        """
        Get the index of the currently selected text.

        Returns
        -------
        int
            Index of the currently selected text.
        """
        return self._selection_ID

    def set_visibility(self, visibility):
        """
        Set the visibility of the UI component.

        Parameters
        ----------
        visibility : bool
            Whether the UI element is visible or not.
        """
        super().set_visibility(visibility)
        if not self._menu_visibility:
            self.drop_down_menu.set_visibility(False)

    def append_item(self, *items):
        """
        Append additional options to the menu.

        Parameters
        ----------
        *items : str or float or list or tuple
            Additional options.
        """
        for item in items:
            if isinstance(item, (list, tuple)):
                self.append_item(*item)
            elif isinstance(item, (str, Number)):
                self.items.append(str(item))
            else:
                raise TypeError("Invalid item instance {}".format(type(item)))

        self.drop_down_menu.values = self.items
        self.drop_down_menu.update_scrollbar()
        self.drop_down_menu.update()
        if not self._menu_visibility:
            self.drop_down_menu.set_visibility(False)

    def select_option_callback(self):
        """Select the appropriate option based on ListBox selection."""
        if not self.drop_down_menu.selected:
            return

        self._selection = self.drop_down_menu.selected[0]
        self._selection_ID = self.drop_down_menu.last_selection_idx

        self.selection_box.message = self._selection
        clip_overflow(self.selection_box, self.selection_box.background.size[0])
        self.drop_down_menu.set_visibility(False)
        self._menu_visibility = False
        self.drop_down_button.toggled = False
        self.on_change(self)

    def menu_toggle_callback(self, event):
        """
        Toggle visibility of drop down menu list.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self._menu_visibility = not self._menu_visibility
        self.drop_down_menu.set_visibility(self._menu_visibility)
        self.drop_down_button.toggled = self._menu_visibility
        if self._menu_visibility:
            self.drop_down_menu.update()
            self.drop_down_menu.update_scrollbar()

    def left_button_pressed(self, event):
        """
        Handle left mouse button press event for dragging.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        click_pos = np.array([event.x, event.y])
        self._drag_offset = click_pos - self.get_position()

    def left_button_dragged(self, event):
        """
        Handle left mouse button drag event for movement.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        if self._drag_offset is not None:
            click_position = np.array([event.x, event.y])
            new_position = click_position - self._drag_offset
            self.set_position(new_position)


class ListBox2D(UI):
    """
    UI component that allows the user to select items from a list.

    Parameters
    ----------
    values : list of objects
        Values used to populate this listbox. Objects must be castable
        to string.
    position : (float, float), optional
        Absolute coordinates (x, y) of the lower-left corner of this
        UI component.
    size : (int, int), optional
        Width and height in pixels of this UI component.
    multiselection : bool, optional
        Whether multiple values can be selected at once.
    reverse_scrolling : bool, optional
        If True, scrolling up will move the list of files down.
    font_size : int, optional
        The font size in pixels.
    line_spacing : float, optional
        Distance between listbox's items in pixels.
    text_color : str, tuple, list or ndarray, optional
        Color of the text. All color parameters accept a hex string
        ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
    selected_color : str, tuple, list or ndarray, optional
        Background color of selected item.
    unselected_color : str, tuple, list or ndarray, optional
        Background color of unselected item.
    scroll_bar_active_color : str, tuple, list or ndarray, optional
        Color of active scroll bar.
    scroll_bar_inactive_color : str, tuple, list or ndarray, optional
        Color of inactive scroll bar.
    background_opacity : float, optional
        Opacity of the background.

    Attributes
    ----------
    on_change : function
        Callback function for when the selected items have changed.
    """

    def __init__(
        self,
        values,
        *,
        position=(0, 0),
        size=(100, 300),
        multiselection=True,
        reverse_scrolling=False,
        font_size=20,
        line_spacing=1.4,
        text_color=(0.2, 0.2, 0.2),
        selected_color=(0.9, 0.6, 0.6),
        unselected_color=(0.6, 0.6, 0.6),
        scroll_bar_active_color=(0.6, 0.2, 0.2),
        scroll_bar_inactive_color=(0.9, 0.0, 0.0),
        background_opacity=1.0,
    ):
        """Init class instance."""
        self.view_offset = 0
        self.slots = []
        self.selected = []

        self.panel_size = np.array(size, dtype=int)
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.slot_height = int(self.font_size * self.line_spacing)

        self.text_color = text_color
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        self.background_opacity = background_opacity

        self.values = values
        self.multiselection = multiselection
        self.last_selection_idx = 0
        self.reverse_scrolling = reverse_scrolling
        super(ListBox2D, self).__init__()

        denom = len(self.values) - self.nb_slots
        if not denom:
            denom += 1
        self.scroll_step_size = (
            self.slot_height * self.nb_slots - self.scroll_bar.height
        ) / denom

        self.scroll_bar_active_color = scroll_bar_active_color
        self.scroll_bar_inactive_color = scroll_bar_inactive_color
        self.scroll_bar.color = self.scroll_bar_inactive_color
        self.scroll_bar.opacity = self.background_opacity

        self.position = position
        self.scroll_init_position = 0
        self.update()

        # Offer some standard hooks to the user.
        self.on_change = lambda: None

    def _setup(self):
        """
        Setup this UI component.

        Create the ListBox (Panel2D) filled with empty slots (ListBoxItem2D).
        """
        self.margin = 10
        size = self.panel_size
        font_size = self.font_size
        # Calculating the number of slots.
        self.nb_slots = int((size[1] - 2 * self.margin) // max(1, self.slot_height))

        # This panel facilitates adding slots at the right position.
        self.panel = Panel2D(size=size, color=(1, 1, 1))

        # Add a scroll bar
        denom = len(self.values) if len(self.values) > 0 else 1
        scroll_bar_height = int(self.nb_slots * (size[1] - 2 * self.margin) / denom)
        scroll_bar_width = int(size[0] / 20)
        self.scroll_bar = Rectangle2D(size=(scroll_bar_width, scroll_bar_height))
        if len(self.values) <= self.nb_slots:
            self.scroll_bar.set_visibility(False)
            self.scroll_bar.height = 0

        scroll_bar_x = int(size[0] - scroll_bar_width - self.margin)
        scroll_bar_y = self.margin
        self._scroll_bar_top_y = scroll_bar_y
        self._scroll_bar_x = scroll_bar_x
        self.panel.add_element(self.scroll_bar, (scroll_bar_x, scroll_bar_y))

        # Initialisation of empty text actors
        self.slot_width = int(size[0] - scroll_bar_width - 3 * self.margin)
        x = self.margin
        y = self.margin
        for _ in range(self.nb_slots):
            item = ListBoxItem2D(
                on_select=self.select,
                size=(self.slot_width, self.slot_height),
                text_color=self.text_color,
                selected_color=self.selected_color,
                unselected_color=self.unselected_color,
                background_opacity=self.background_opacity,
            )
            item.textblock.font_size = font_size
            self.slots.append(item)
            self.panel.add_element(item, (int(x), int(y)))
            y += self.slot_height

        # Add default events listener for this UI component.
        self.scroll_bar.on_left_mouse_button_pressed = self.scroll_click_callback
        self.scroll_bar.on_left_mouse_button_released = self.scroll_release_callback
        self.scroll_bar.on_left_mouse_button_dragged = self.scroll_drag_callback

        self.panel.background.actor.add_event_handler(
            self.wheel_callback, EventType.WHEEL
        )

        # Handle mouse wheel events on the slots.
        for slot in self.slots:
            slot.background.actor.add_event_handler(
                self.wheel_callback, EventType.WHEEL
            )
            for text_actor in slot.textblock.actors:
                text_actor.add_event_handler(self.wheel_callback, EventType.WHEEL)

        self._children.extend([self.panel])

    def resize(self, size):
        """
        Resize the component.

        Parameters
        ----------
        size : (int, int)
            Size to resize to.
        """
        self.panel_size = np.array(size, dtype=int)
        self.panel.resize(size)

        self.nb_slots = int((size[1] - 2 * self.margin) // max(1, self.slot_height))

        new_scrollbar_width = int(size[0] / 20)
        self._scroll_bar_x = int(size[0] - new_scrollbar_width - self.margin)

        self.slot_width = int(size[0] - new_scrollbar_width - 3 * self.margin)
        x = self.margin

        if self.nb_slots > len(self.slots):
            font_size = self.font_size

            for _ in range(self.nb_slots - len(self.slots)):
                item = ListBoxItem2D(
                    on_select=self.select,
                    size=(self.slot_width, self.slot_height),
                    text_color=self.text_color,
                    selected_color=self.selected_color,
                    unselected_color=self.unselected_color,
                    background_opacity=self.background_opacity,
                )
                item.textblock.font_size = font_size
                item.background.actor.add_event_handler(
                    self.wheel_callback, EventType.WHEEL
                )
                for text_actor in item.textblock.actors:
                    text_actor.add_event_handler(self.wheel_callback, EventType.WHEEL)

                self.slots.append(item)
                self.panel.add_element(item, (0, 0))

        while len(self.slots) > self.nb_slots:
            item = self.slots.pop()
            self.panel.remove_element(item)

        y = self.margin
        for slot in self.slots:
            self.panel.update_element(slot, (int(x), int(y)))
            slot.textblock.font_size = self.font_size
            slot.resize((self.slot_width, self.slot_height))
            y += self.slot_height

        if self.view_offset + self.nb_slots > len(self.values):
            self.view_offset = max(0, len(self.values) - self.nb_slots)

        self.scroll_bar.width = new_scrollbar_width
        self.update()
        self.update_scrollbar()

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            List of actors.
        """
        return []

    def _get_size(self):
        """
        Get the dimensions of the component.

        Returns
        -------
        (int, int)
            Size.
        """
        return self.panel.size

    def _update_actors_position(self):
        """Position the lower-left corner of this UI component."""
        self.panel.set_position(self.get_position())

    def _update_scroll_bar_position(self):
        """Update the scroll bar position in the panel based on view_offset."""
        if len(self.values) <= self.nb_slots:
            return

        scroll_bar_y = int(
            self._scroll_bar_top_y + self.view_offset * self.scroll_step_size
        )
        self.panel.update_element(self.scroll_bar, (self._scroll_bar_x, scroll_bar_y))

    def wheel_callback(self, event):
        """
        Handle mouse wheel scroll events.

        Parameters
        ----------
        event : object
            The pygfx event.
        """
        dy = event.dy
        if self.reverse_scrolling:
            dy = -dy
        if dy > 0:
            self.scroll_down()
        elif dy < 0:
            self.scroll_up()

    def scroll_up(self):
        """Scroll up by one item."""
        if self.view_offset > 0:
            self.view_offset -= 1
            self.update()
            self._update_scroll_bar_position()

    def scroll_down(self):
        """Scroll down by one item."""
        view_end = self.view_offset + self.nb_slots
        if view_end < len(self.values):
            self.view_offset += 1
            self.update()
            self._update_scroll_bar_position()

    def scroll_click_callback(self, event):
        """
        Callback to change the color of the bar when it is clicked.

        Parameters
        ----------
        event : object
            The pygfx event.
        """
        self.scroll_bar.color = self.scroll_bar_active_color
        self.scroll_init_position = event.y

    def scroll_release_callback(self, event):
        """
        Callback to change the color of the bar when it is released.

        Parameters
        ----------
        event : object
            The pygfx event.
        """
        self.scroll_bar.color = self.scroll_bar_inactive_color

    def scroll_drag_callback(self, event):
        """
        Drag scroll bar in the combo box.

        Parameters
        ----------
        event : object
            The pygfx event.
        """
        position_y = event.y
        if self.scroll_step_size == 0:
            return
        offset = int((position_y - self.scroll_init_position) / self.scroll_step_size)
        if offset > 0 and (self.view_offset + self.nb_slots < len(self.values)):
            offset = min(offset, len(self.values) - self.nb_slots - self.view_offset)
        elif offset < 0 and self.view_offset > 0:
            offset = max(offset, -self.view_offset)
        else:
            return

        self.view_offset += offset
        self.update()
        self._update_scroll_bar_position()
        self.scroll_init_position += offset * self.scroll_step_size

    def update(self):
        """Refresh listbox content."""
        view_start = self.view_offset
        view_end = view_start + self.nb_slots
        values_to_show = self.values[view_start:view_end]

        # Populate slots according to the view.
        for i, choice in enumerate(values_to_show):
            slot = self.slots[i]
            slot.element = choice
            clip_overflow(slot.textblock, self.slot_width)
            slot.set_visibility(True)
            if slot.size[1] != self.slot_height:
                slot.resize((self.slot_width, self.slot_height))
            if slot.element in self.selected:
                slot.select()
            else:
                slot.deselect()

        # Flush remaining slots.
        for slot in self.slots[len(values_to_show) :]:
            slot.element = None
            slot.set_visibility(False)
            slot.resize((self.slot_width, 0))
            slot.deselect()

    def update_scrollbar(self):
        """Change the scroll-bar height when the values change."""
        self.scroll_bar.set_visibility(True)

        denom = len(self.values) if len(self.values) > 0 else 1
        new_scrollbar_height = int(
            self.nb_slots * (self.panel_size[1] - 2 * self.margin) / denom
        )
        self.scroll_bar.height = new_scrollbar_height

        step_denom = len(self.values) - self.nb_slots
        if step_denom == 0:
            step_denom = 1

        self.scroll_step_size = max(
            1,
            (self.slot_height * self.nb_slots - new_scrollbar_height) / step_denom,
        )

        self._scroll_bar_top_y = self.margin
        self._update_scroll_bar_position()

        if len(self.values) <= self.nb_slots:
            self.scroll_bar.set_visibility(False)
            self.scroll_bar.height = 0

    def clear_selection(self):
        """Clear all items from the current selection."""
        del self.selected[:]

    def select(self, item, *, multiselect=False, range_select=False):
        """
        Select the item.

        Parameters
        ----------
        item : object
            Item to select.
        multiselect : bool, optional
            If True and multiselection is allowed, the item is added to the selection.
        range_select : bool, optional
            If True and multiselection is allowed, all items between the
            last selected item and the current one will be added.
        """
        selection_idx = self.values.index(item.element)
        if self.multiselection and range_select:
            self.clear_selection()
            step = 1 if selection_idx >= self.last_selection_idx else -1
            for i in range(self.last_selection_idx, selection_idx + step, step):
                self.selected.append(self.values[i])

        elif self.multiselection and multiselect:
            if item.element in self.selected:
                self.selected.remove(item.element)
            else:
                self.selected.append(item.element)
            self.last_selection_idx = selection_idx

        else:
            self.clear_selection()
            self.selected.append(item.element)
            self.last_selection_idx = selection_idx

        self.update()
        self.on_change()


class ListBoxItem2D(UI):
    """
    The text displayed in a listbox.

    Parameters
    ----------
    on_select : callable
        Callback invoked when the item is clicked.
    size : (int, int)
        Size of the item.
    text_color : str, tuple, list or ndarray, optional
        Text color. All color parameters accept a hex string ("#FF0000"),
        RGB(A) in [0, 1], or RGB(A) in [0, 255].
    selected_color : str, tuple, list or ndarray, optional
        Selected background color.
    unselected_color : str, tuple, list or ndarray, optional
        Unselected background color.
    background_opacity : float, optional
        Opacity.
    """

    def __init__(
        self,
        on_select,
        size,
        *,
        text_color=(1.0, 0.0, 0.0),
        selected_color=(0.4, 0.4, 0.4),
        unselected_color=(0.9, 0.9, 0.9),
        background_opacity=1.0,
    ):
        """
        Init ListBox Item instance.

        Parameters
        ----------
        on_select : callable
            Callback invoked when the item is clicked.
        size : tuple of 2 ints
            The size of the listbox item.
        text_color : str, tuple, list or ndarray
        unselected_color : str, tuple, list or ndarray
        selected_color : str, tuple, list or ndarray
        background_opacity : float

        """
        self._item_size = size
        super(ListBoxItem2D, self).__init__()
        self._element = None
        self._on_select = on_select
        self.background.resize(size)
        self.background_opacity = background_opacity
        self.selected = False
        self.text_color = text_color
        self.textblock.color = self.text_color
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        self.background.opacity = self.background_opacity
        self.deselect()

    def _setup(self):
        """
        Setup this UI component.

        Create the ListBoxItem2D with its background (Rectangle2D) and its
        label (TextBlock2D).
        """
        self.background = Rectangle2D(size=self._item_size)
        self.textblock = TextBlock2D(
            size=self._item_size,
            justification="left",
            vertical_justification="middle",
        )

        # Add default events listener for this UI component.
        self.textblock.on_left_mouse_button_clicked = self.left_button_clicked
        self.background.on_left_mouse_button_clicked = self.left_button_clicked

        self._children.extend([self.background, self.textblock])

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            List of actors.
        """
        return []

    def _get_size(self):
        """
        Get the dimensions of the component.

        Returns
        -------
        (int, int)
            Size.
        """
        return self.background.size

    def _update_actors_position(self):
        """Set the lower-left corner position of this UI component."""
        coords = self.get_position()
        self.background.set_position(coords)
        self.textblock.set_position(coords)

        # left-alignment to prevent text unaligning during updates.
        self.textblock.actor.anchor = "middle-left"
        pos = self.textblock.actor.local.position
        self.textblock.actor.local.position = (coords[0], pos[1], pos[2])

    def deselect(self):
        """Deselect the item and remove highlight."""
        self.background.color = self.unselected_color
        self.textblock.bold = False
        self.textblock.message = self.textblock.message  # Force redraw
        self.selected = False

    def select(self):
        """Select the item and highlight its background."""
        self.textblock.bold = True
        self.textblock.message = self.textblock.message  # Force redraw
        self.background.color = self.selected_color
        self.selected = True

    @property
    def element(self):
        """
        Get the stored element.

        Returns
        -------
        object
            Element.
        """
        return self._element

    @element.setter
    def element(self, element):
        """
        Set the element and update the text message.

        Parameters
        ----------
        element : object
            Element to set.
        """
        self._element = element
        self.textblock.message = "" if self._element is None else str(element)

    def left_button_clicked(self, event):
        """
        Handle left click for this UI element.

        Parameters
        ----------
        event : object
            The pygfx event.
        """
        modifiers = getattr(event, "modifiers", None) or ()
        multiselect = "Control" in modifiers
        range_select = "Shift" in modifiers
        self._on_select(item=self, multiselect=multiselect, range_select=range_select)

    def resize(self, size):
        """
        Resize the component.

        Parameters
        ----------
        size : (int, int)
            Size to resize to.
        """
        self.background.resize(size)


# class FileMenu2D(UI):
#     """A menu to select files in the current folder.

#     Can go to new folder, previous folder and select multiple files.

#     Attributes
#     ----------
#     extensions: ['extension1', 'extension2', ....]
#         To show all files, extensions=["*"] or [""]
#         List of extensions to be shown as files.
#     listbox : :class: 'ListBox2D'
#         Container for the menu.

#     """

#     @warn_on_args_to_kwargs()
#     def __init__(
#         self,
#         directory_path,
#         *,
#         extensions=None,
#         position=(0, 0),
#         size=(100, 300),
#         multiselection=True,
#         reverse_scrolling=False,
#         font_size=20,
#         line_spacing=1.4,
#     ):
#         """Init class instance.

#         Parameters
#         ----------
#         extensions: list(string)
#             List of extensions to be shown as files.
#         directory_path: string
#             Path of the directory where this dialog should open.
#         position : (float, float)
#             Absolute coordinates (x, y) of the lower-left corner of this
#             UI component.
#         size : (int, int)
#             Width and height in pixels of this UI component.
#         multiselection: {True, False}
#             Whether multiple values can be selected at once.
#         reverse_scrolling: {True, False}
#             If True, scrolling up will move the list of files down.
#         font_size: int
#             The font size in pixels.
#         line_spacing: float
#             Distance between listbox's items in pixels.

#         """
#         self.font_size = font_size
#         self.multiselection = multiselection
#         self.reverse_scrolling = reverse_scrolling
#         self.line_spacing = line_spacing
#         self.extensions = extensions or ["*"]
#         self.current_directory = directory_path
#         self.menu_size = size
#         self.directory_contents = []

#         super(FileMenu2D, self).__init__()
#         self.position = position
#         self.set_slot_colors()

#     def _setup(self):
#         """Setup this UI component.

#         Create the ListBox (Panel2D) filled with empty slots (ListBoxItem2D).

#         """
#         self.directory_contents = self.get_all_file_names()
#         content_names = [x[0] for x in self.directory_contents]
#         self.listbox = ListBox2D(
#             values=content_names,
#             multiselection=self.multiselection,
#             font_size=self.font_size,
#             line_spacing=self.line_spacing,
#             reverse_scrolling=self.reverse_scrolling,
#             size=self.menu_size,
#         )

#         self.add_callback(
#             self.listbox.scroll_bar.actor, "MouseMoveEvent", self.scroll_callback
#         )

#         # Handle mouse wheel events on the panel.
#         up_event = "MouseWheelForwardEvent"
#         down_event = "MouseWheelBackwardEvent"
#         if self.reverse_scrolling:
#             up_event, down_event = down_event, up_event  # Swap events

#         self.add_callback(
#             self.listbox.panel.background.actor, up_event, self.scroll_callback
#         )
#         self.add_callback(
#             self.listbox.panel.background.actor, down_event, self.scroll_callback
#         )

#         # Handle mouse wheel events on the slots.
#         for slot in self.listbox.slots:
#             self.add_callback(slot.background.actor, up_event, self.scroll_callback)
#             self.add_callback(slot.background.actor, down_event, self.scroll_callback)
#             self.add_callback(slot.textblock.actor, up_event, self.scroll_callback)
#             self.add_callback(slot.textblock.actor, down_event, self.scroll_callback)
#             slot.add_callback(
#                 slot.textblock.actor,
#                 "LeftButtonPressEvent",
#                 self.directory_click_callback,
#             )
#             slot.add_callback(
#                 slot.background.actor,
#                 "LeftButtonPressEvent",
#                 self.directory_click_callback,
#             )

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         return self.listbox.actors

#     def resize(self, size):
#         pass

#     def _set_position(self, coords):
#         """Set the lower-left corner position of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         self.listbox.position = coords

#     def _add_to_scene(self, scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : scene

#         """
#         self.listbox.add_to_scene(scene)

#     def _get_size(self):
#         return self.listbox.size

#     def get_all_file_names(self):
#         """Get file and directory names.

#         Returns
#         -------
#         all_file_names: list((string, {"directory", "file"}))
#             List of all file and directory names as string.

#         """
#         all_file_names = []

#         directory_names = self.get_directory_names()
#         for directory_name in directory_names:
#             all_file_names.append((directory_name, "directory"))

#         file_names = self.get_file_names()
#         for file_name in file_names:
#             all_file_names.append((file_name, "file"))

#         return all_file_names

#     def get_directory_names(self):
#         """Find names of all directories in the current_directory

#         Returns
#         -------
#         directory_names: list(string)
#             List of all directory names as string.

#         """
#         # A list of directory names in the current directory
#         directory_names = []
#         for _, dirnames, _ in os.walk(self.current_directory):
#             directory_names += dirnames
#             break
#         directory_names.sort(key=lambda s: s.lower())
#         directory_names.insert(0, "../")
#         return directory_names

#     def get_file_names(self):
#         """Find names of all files in the current_directory

#         Returns
#         -------
#         file_names: list(string)
#             List of all file names as string.

#         """
#         # A list of file names with extension in the current directory
#         files = []
#         for _, _, f in os.walk(self.current_directory):
#             files += f
#             break

#         file_names = []
#         if "*" in self.extensions or "" in self.extensions:
#             file_names = files
#         else:
#             for ext in self.extensions:
#                 for file in files:
#                     if file.endswith("." + ext):
#                         file_names.append(file)
#         file_names.sort(key=lambda s: s.lower())
#         return file_names

#     def set_slot_colors(self):
#         """Set the text color of the slots based on the type of element
#         they show. Blue for directories and green for files.
#         """
#         for idx, slot in enumerate(self.listbox.slots):
#             list_idx = min(
#                 self.listbox.view_offset + idx, len(self.directory_contents) - 1
#             )
#             if self.directory_contents[list_idx][1] == "directory":
#                 slot.textblock.color = (0, 0.6, 0)
#             elif self.directory_contents[list_idx][1] == "file":
#                 slot.textblock.color = (0, 0, 0.7)

#     def scroll_callback(self, i_ren, _obj, _filemenu_item):
#         """Handle scroll and change the slot text colors.

#         Parameters
#         ----------
#         i_ren: :class:`CustomInteractorStyle`
#         obj: :class:`vtkActor`
#             The picked actor
#         _filemenu_item: :class:`FileMenu2D`

#         """
#         self.set_slot_colors()
#         i_ren.force_render()
#         i_ren.event.abort()

#     def directory_click_callback(self, i_ren, _obj, listboxitem):
#         """Handle the move into a directory if it has been clicked.

#         Parameters
#         ----------
#         i_ren: :class:`CustomInteractorStyle`
#         obj: :class:`vtkActor`
#             The picked actor
#         listboxitem: :class:`ListBoxItem2D`

#         """
#         if (listboxitem.element, "directory") in self.directory_contents:
#             new_directory_path = os.path.join(
#                 self.current_directory, listboxitem.element
#             )
#             if os.access(new_directory_path, os.R_OK):
#                 self.current_directory = new_directory_path
#                 self.directory_contents = self.get_all_file_names()
#                 content_names = [x[0] for x in self.directory_contents]
#                 self.listbox.clear_selection()
#                 self.listbox.values = content_names
#                 self.listbox.view_offset = 0
#                 self.listbox.update()
#                 self.listbox.update_scrollbar()
#                 self.set_slot_colors()
#         i_ren.force_render()
#         i_ren.event.abort()


# class DrawShape(UI):
#     """Create and Manage 2D Shapes."""

#     @warn_on_args_to_kwargs()
#     def __init__(self, shape_type, *, drawpanel=None, position=(0, 0)):
#         """Init this UI element.

#         Parameters
#         ----------
#         shape_type : string
#             Type of shape to be created.
#         drawpanel : DrawPanel, optional
#             Reference to the main canvas on which it is drawn.
#         position : (float, float), optional
#             (x, y) in pixels.

#         """
#         self.shape = None
#         self.shape_type = shape_type.lower()
#         self.drawpanel = drawpanel
#         self.max_size = None
#         self.rotation = 0
#         super(DrawShape, self).__init__(position=position)
#         self.shape.color = np.random.random(3)

#     def _setup(self):
#         """Setup this UI component.

#         Create a Shape.
#         """
#         if self.shape_type == "line":
#             self.shape = Rectangle2D(size=(3, 3))
#         elif self.shape_type == "quad":
#             self.shape = Rectangle2D(size=(3, 3))
#         elif self.shape_type == "circle":
#             self.shape = Disk2D(outer_radius=2)
#         else:
#             raise IOError("Unknown shape type: {}.".format(self.shape_type))

#         self.shape.on_left_mouse_button_pressed = self.left_button_pressed
#         self.shape.on_left_mouse_button_dragged = self.left_button_dragged
#         self.shape.on_left_mouse_button_released = self.left_button_released

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         return self.shape

#     def _add_to_scene(self, scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : scene

#         """
#         self._scene = scene
#         self.shape.add_to_scene(scene)

#     def _get_size(self):
#         return self.shape.size

#     def _set_position(self, coords):
#         """Set the lower-left corner position of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         if self.shape_type == "circle":
#             self.shape.center = coords
#         else:
#             self.shape.position = coords

#     def update_shape_position(self, center_position):
#         """Update the center position on the canvas.

#         Parameters
#         ----------
#         center_position: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         new_center = self.clamp_position(center=center_position)
#         self.drawpanel.canvas.update_element(self, new_center, anchor="center")
#         self.cal_bounding_box()

#     @property
#     def center(self):
#         return self._bounding_box_min + self._bounding_box_size // 2

#     @center.setter
#     def center(self, coords):
#         """Position the center of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         new_center = np.array(coords)
#         new_lower_left_corner = new_center - self._bounding_box_size // 2
#         self.position = new_lower_left_corner + self._bounding_box_offset
#         self.cal_bounding_box()

#     @property
#     def is_selected(self):
#         return self._is_selected

#     @is_selected.setter
#     def is_selected(self, value):
#         if self.drawpanel and value:
#             self.drawpanel.current_shape = self
#         self._is_selected = value
#         self.selection_change()

#     def selection_change(self):
#         if self.is_selected:
#             self.drawpanel.rotation_slider.value = self.rotation
#         else:
#             self.drawpanel.rotation_slider.set_visibility(False)

#     def rotate(self, angle):
#         """Rotate the vertices of the UI component using specific angle.

#         Parameters
#         ----------
#         angle: float
#             Value by which the vertices are rotated in radian.

#         """
#         if self.shape_type == "circle":
#             return
#         points_arr = vertices_from_actor(self.shape.actor)
#         new_points_arr = rotate_2d(points_arr, angle)
#         set_polydata_vertices(self.shape._polygonPolyData, new_points_arr)
#         update_actor(self.shape.actor)

#         self.cal_bounding_box()

#     def cal_bounding_box(self):
#         """Calculate the min, max position and the size of the bounding box."""
#         vertices = self.position + vertices_from_actor(self.shape.actor)[:, :-1]

#         (
#             self._bounding_box_min,
#             self._bounding_box_max,
#             self._bounding_box_size,
#         ) = cal_bounding_box_2d(vertices)

#         self._bounding_box_offset = self.position - self._bounding_box_min

#     @warn_on_args_to_kwargs()
#     def clamp_position(self, *, center=None):
#         """Clamp the given center according to the DrawPanel canvas.

#         Parameters
#         ----------
#         center : (float, float)
#             (x, y) in pixels.

#         Returns
#         -------
#         new_center: ndarray(int)
#             New center for the shape.

#         """
#         center = self.center if center is None else center
#         new_center = np.clip(
#             center,
#             self._bounding_box_size // 2,
#             self.drawpanel.canvas.size - self._bounding_box_size // 2,
#         )
#         return new_center.astype(int)

#     def resize(self, size):
#         """Resize the UI."""
#         if self.shape_type == "line":
#             hyp = np.hypot(size[0], size[1])
#             self.shape.resize((hyp, 3))
#             self.rotate(angle=np.arctan2(size[1], size[0]))

#         elif self.shape_type == "quad":
#             self.shape.resize(size)

#         elif self.shape_type == "circle":
#             hyp = np.hypot(size[0], size[1])
#             if self.max_size and hyp > self.max_size:
#                 hyp = self.max_size
#             self.shape.outer_radius = hyp

#         self.cal_bounding_box()

#     def remove(self):
#         """Remove the Shape and all related actors."""
#         self._scene.rm(self.shape.actor)
#         self.drawpanel.rotation_slider.set_visibility(False)

#     def left_button_pressed(self, i_ren, _obj, shape):
#         mode = self.drawpanel.current_mode
#         if mode == "selection":
#             self.drawpanel.update_shape_selection(self)

#             click_pos = np.array(i_ren.event.position)
#             self._drag_offset = click_pos - self.center
#             self.drawpanel.show_rotation_slider()
#             i_ren.event.abort()
#         elif mode == "delete":
#             self.remove()
#         else:
#             self.drawpanel.left_button_pressed(i_ren, _obj, self.drawpanel)
#         i_ren.force_render()

#     def left_button_dragged(self, i_ren, _obj, shape):
#         if self.drawpanel.current_mode == "selection":
#             self.drawpanel.rotation_slider.set_visibility(False)
#             if self._drag_offset is not None:
#                 click_position = i_ren.event.position
#                 relative_center_position = (
#
#                   click_position - self._drag_offset - self.drawpanel.canvas.position
#                 )
#                 self.update_shape_position(relative_center_position)
#             i_ren.force_render()
#         else:
#             self.drawpanel.left_button_dragged(i_ren, _obj, self.drawpanel)

#     def left_button_released(self, i_ren, _obj, shape):
#         if self.drawpanel.current_mode == "selection":
#             self.drawpanel.show_rotation_slider()
#             i_ren.force_render()


# class DrawPanel(UI):
#     """The main Canvas(Panel2D) on which everything would be drawn."""

#     @warn_on_args_to_kwargs()
#     def __init__(self, *, size=(400, 400), position=(0, 0), is_draggable=False):
#         """Init this UI element.

#         Parameters
#         ----------
#         size : (int, int), optional
#             Width and height in pixels of this UI component.
#         position : (float, float), optional
#             (x, y) in pixels.
#         is_draggable : bool, optional
#             Whether the background canvas will be draggble or not.

#         """
#         self.panel_size = size
#         super(DrawPanel, self).__init__(position=position)
#         self.is_draggable = is_draggable
#         self.current_mode = None

#         if is_draggable:
#             self.current_mode = "selection"

#         self.shape_list = []
#         self.current_shape = None

#     def _setup(self):
#         """Setup this UI component.

#         Create a Canvas(Panel2D).
#         """
#         self.canvas = Panel2D(size=self.panel_size)
#         self.canvas.background.on_left_mouse_button_pressed = self.left_button_pressed
#         self.canvas.background.on_left_mouse_button_dragged = self.left_button_dragged

#         # Todo
#         # Convert mode_data into a private variable and make it read-only
#         # Then add the ability to insert user-defined mode
#         mode_data = {
#             "selection": ["selection.png", "selection-pressed.png"],
#             "line": ["line.png", "line-pressed.png"],
#             "quad": ["quad.png", "quad-pressed.png"],
#             "circle": ["circle.png", "circle-pressed.png"],
#             "delete": ["delete.png", "delete-pressed.png"],
#         }

#         padding = 5
#         # Todo
#         # Add this size to __init__
#         mode_panel_size = (len(mode_data) * 35 + 2 * padding, 40)
#         self.mode_panel = Panel2D(size=mode_panel_size, color=(0.5, 0.5, 0.5))
#         btn_pos = np.array([0, 0])

#         for mode, fname in mode_data.items():
#             icon_files = []
#             icon_files.append(
#               (mode, read_viz_icons(style="new_icons", fname=fname[0])))
#             icon_files.append(
#                 (mode + "-pressed", read_viz_icons(style="new_icons", fname=fname[1]))
#             )
#             btn = Button2D(icon_fnames=icon_files)

#             def mode_selector(i_ren, _obj, btn):
#                 self.current_mode = btn.icon_names[0]
#                 i_ren.force_render()

#             btn.on_left_mouse_button_pressed = mode_selector

#             self.mode_panel.add_element(btn, btn_pos + padding)
#             btn_pos[0] += btn.size[0] + padding

#         self.canvas.add_element(self.mode_panel, (0, -mode_panel_size[1]))

#         self.mode_text = TextBlock2D(
#             text="Select appropriate drawing mode using below icon"
#         )
#         self.canvas.add_element(self.mode_text, (0.0, 1.0))

#         self.rotation_slider = RingSlider2D(
#             initial_value=0, text_template="{angle:5.1f}°"
#         )
#         self.rotation_slider.set_visibility(False)

#         def rotate_shape(slider):
#             angle = slider.value
#             previous_angle = slider.previous_value
#             rotation_angle = angle - previous_angle

#             current_center = self.current_shape.center
#             self.current_shape.rotate(np.deg2rad(rotation_angle))
#             self.current_shape.rotation = slider.value
#             self.current_shape.update_shape_position(
#                 current_center - self.canvas.position
#             )

#         self.rotation_slider.on_moving_slider = rotate_shape

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         return self.canvas.actors

#     def _add_to_scene(self, scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : scene

#         """
#         self._scene = scene
#         self.canvas.add_to_scene(scene)

#     def _get_size(self):
#         return self.canvas.size

#     def _set_position(self, coords):
#         """Set the lower-left corner position of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         self.canvas.position = coords + [0, self.mode_panel.size[1]]
#         slider_position = self.canvas.position + [
#             self.canvas.size[0] - self.rotation_slider.size[0] / 2,
#             self.rotation_slider.size[1] / 2,
#         ]
#         self.rotation_slider.center = slider_position

#     def resize(self, size):
#         """Resize the UI."""
#         pass

#     @property
#     def current_mode(self):
#         return self._current_mode

#     @current_mode.setter
#     def current_mode(self, mode):
#         self.update_button_icons(mode)
#         self._current_mode = mode
#         if mode is not None:
#             self.mode_text.message = f"Mode: {mode}"

#     def cal_min_boundary_distance(self, position):
#         """Calculate minimum distance between the current
# position and canvas boundary.

#         Parameters
#         ----------
#         position: (float,float)
#             current position of the shape.

#         Returns
#         -------
#         float
#             Minimum distance from the boundary.

#         """
#         distance_list = []
#         # calculate distance from element to left and lower boundary
#         distance_list.extend(position - self.canvas.position)
#         # calculate distance from element to upper and right boundary
#         distance_list.extend(self.canvas.position + self.canvas.size - position)

#         return min(distance_list)

#     def draw_shape(self, shape_type, current_position):
#         """Draw the required shape at the given position.

#         Parameters
#         ----------
#         shape_type: string
#             Type of shape - line, quad, circle.
#         current_position: (float,float)
#             Lower left corner position for the shape.

#         """
#         shape = DrawShape(
#             shape_type=shape_type, drawpanel=self, position=current_position
#         )
#         if shape_type == "circle":
#             shape.max_size = self.cal_min_boundary_distance(current_position)
#         self.shape_list.append(shape)
#         self._scene.add(shape)
#         self.canvas.add_element(shape, current_position - self.canvas.position)
#         self.update_shape_selection(shape)

#     def resize_shape(self, current_position):
#         """Resize the shape.

#         Parameters
#         ----------
#         current_position: (float,float)
#             Lower left corner position for the shape.

#         """
#         self.current_shape = self.shape_list[-1]
#         size = current_position - self.current_shape.position
#         self.current_shape.resize(size)

#     def update_shape_selection(self, selected_shape):
#         for shape in self.shape_list:
#             if selected_shape == shape:
#                 shape.is_selected = True
#             else:
#                 shape.is_selected = False

#     def show_rotation_slider(self):
#         """Display the  RingSlider2D to allow rotation of shape from the center."""
#         self._scene.rm(*self.rotation_slider.actors)
#         self.rotation_slider.add_to_scene(self._scene)
#         self.rotation_slider.set_visibility(True)

#     def update_button_icons(self, current_mode):
#         """Update the button icon.

#         Parameters
#         ----------
#         current_mode: string
#             Current mode of the UI.

#         """
#         for btn in self.mode_panel._elements[1:]:
#             if btn.icon_names[0] == current_mode:
#                 btn.next_icon()
#             elif btn.current_icon_id == 1:
#                 btn.next_icon()

#     def clamp_mouse_position(self, mouse_position):
#         """Restrict the mouse position to the canvas boundary.

#         Parameters
#         ----------
#         mouse_position: (float,float)
#             Current mouse position.

#         Returns
#         -------
#         list(float)
#             New clipped position.

#         """
#         return np.clip(
#             mouse_position,
#             self.canvas.position,
#             self.canvas.position + self.canvas.size,
#         )

#     def handle_mouse_click(self, position):
#         if self.current_mode == "selection":
#             if self.is_draggable:
#                 self._drag_offset = position - self.position
#             self.current_shape.is_selected = False
#         if self.current_mode in ["line", "quad", "circle"]:
#             self.draw_shape(self.current_mode, position)

#     def left_button_pressed(self, i_ren, _obj, element):
#         self.handle_mouse_click(i_ren.event.position)
#         i_ren.force_render()

#     def handle_mouse_drag(self, position):
#         if self.is_draggable and self.current_mode == "selection":
#             if self._drag_offset is not None:
#                 new_position = position - self._drag_offset
#                 self.position = new_position
#         if self.current_mode in ["line", "quad", "circle"]:
#             self.resize_shape(position)

#     def left_button_dragged(self, i_ren, _obj, element):
#         mouse_position = self.clamp_mouse_position(i_ren.event.position)
#         self.handle_mouse_drag(mouse_position)
#         i_ren.force_render()


class Card2D(UI):
    """
    A 2D card UI component that displays an image with title and body text.

    The card layout places the image at the top, followed by the title and
    body text below. It can optionally be dragged around the scene.

    Parameters
    ----------
    image_path : str
        Path to the image file. Supports png and jpg/jpeg images.
    body_text : str, optional
        Card body text.
    draggable : bool, optional
        If True, the card can be dragged with the mouse.
    title_text : str, optional
        Card title text.
    padding : int, optional
        Padding between image, title, and body in pixels.
    position : (float, float), optional
        Absolute coordinates (x, y) for placement.
    size : (int, int), optional
        Width and height in pixels of the card.
    image_scale : float, optional
        Fraction of the card height taken by the image (between 0 and 1).
    bg_color : str, tuple, list or ndarray, optional
        Background color of the card. A hex string ("#FF0000"), RGB(A) in
        [0, 1], or RGB(A) in [0, 255].
    bg_opacity : float, optional
        Background opacity. Must be in [0, 1].
    title_color : str, tuple, list or ndarray, optional
        Title text color, same formats as ``bg_color``.
    body_color : str, tuple, list or ndarray, optional
        Body text color, same formats as ``bg_color``.
    border_color : str, tuple, list or ndarray, optional
        Border color, same formats as ``bg_color``.
    border_width : int, optional
        Width of the border in pixels.
    maintain_aspect : bool, optional
        If True, the image is scaled to maintain its aspect ratio.
    z_order : int, optional
        The stacking priority of the card.

    Attributes
    ----------
    image : :class:`ImageContainer2D`
        Renders the image on the card.
    title_box : :class:`TextBlock2D`
        Displays the title on the card.
    body_box : :class:`TextBlock2D`
        Displays the body text on the card.
    panel : :class:`Panel2D`
        The background panel that holds all card elements.
    """

    def __init__(
        self,
        image_path,
        *,
        body_text="",
        draggable=True,
        title_text="",
        padding=10,
        position=(0, 0),
        size=(400, 400),
        image_scale=0.5,
        bg_color=(0.5, 0.5, 0.5),
        bg_opacity=1,
        title_color=(0.0, 0.0, 0.0),
        body_color=(0.0, 0.0, 0.0),
        border_color=(1.0, 1.0, 1.0),
        border_width=0,
        maintain_aspect=False,
        z_order=0,
    ):
        """Initialize the Card2D instance."""
        self._drag_offset = None
        self.image_path = image_path
        self._extension = get_extension(self.image_path)
        if self._extension not in ["jpg", "jpeg", "png"]:
            raise UnidentifiedImageError(
                f"Image extension {self._extension} not supported"
            )

        self.body_text = body_text
        self.title_text = title_text
        self.draggable = draggable
        self.card_size = size
        self.padding = padding

        self._title_color = normalize_colors(title_color)[0]
        self._body_color = normalize_colors(body_color)[0]
        self._bg_color = normalize_colors(bg_color)[0]
        self._border_color = normalize_colors(border_color)[0]
        self._bg_opacity = np.clip(bg_opacity, 0, 1)

        self.text_scale = np.clip(1 - image_scale, 0, 1)
        self.image_scale = np.clip(image_scale, 0, 1)

        self._image_data = load_image(self.image_path)

        self.maintain_aspect = maintain_aspect
        if self.maintain_aspect:
            self._true_image_size = self._image_data.shape[:2][::-1]

        self.border_width = border_width
        self.has_border = bool(border_width)

        super(Card2D, self).__init__(position=position, z_order=z_order)

        if self.maintain_aspect:
            self._new_size = (
                self._true_image_size[0],
                self._true_image_size[1] // self.image_scale,
            )
            self.resize(self._new_size)
        else:
            self.resize(size)

    def _setup(self):
        """
        Set up this UI component.

        Creates the image, title, body text, and a Panel2D to hold them.
        """
        self._image_size, _title_box_size, _body_box_size = self._calculate_sizes(
            self.card_size
        )

        self.image = ImageContainer2D(img_path=self._image_data, size=self._image_size)

        self.body_box = TextBlock2D(
            text=self.body_text, color=self._body_color, size=_body_box_size
        )

        self.title_box = TextBlock2D(
            text=self.title_text,
            bold=True,
            color=self._title_color,
            size=_title_box_size,
        )

        self.panel = Panel2D(
            self.card_size,
            color=self._bg_color,
            opacity=self._bg_opacity,
            border_color=self._border_color,
            border_width=self.border_width,
            has_border=self.has_border,
        )

        self.panel.add_element(self.image, (0, 0))
        self.panel.add_element(self.title_box, (0, 0))
        self.panel.add_element(self.body_box, (0, 0))

        self._setup_drag_events()
        self._children.append(self.panel)

    def _setup_drag_events(self):
        """Attach drag event handlers to all interactive card surfaces."""
        if self.draggable:
            drag_targets = [
                self.panel.background,
                self.image,
                self.title_box,
                self.body_box,
            ]
            if self.has_border:
                drag_targets.extend(self.panel.borders.values())

            for target in drag_targets:
                target.on_left_mouse_button_dragged = self.left_button_dragged
                target.on_left_mouse_button_pressed = self.left_button_pressed
        else:
            self.panel.background.on_left_mouse_button_dragged = lambda event: None

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            Empty list as this UI uses other UI elements as children
            instead of direct actors.
        """
        return []

    def _get_size(self):
        """
        Get the total size of the card.

        Returns
        -------
        (int, int)
            Width and height in pixels.
        """
        return self.panel.size

    def resize(self, size):
        """
        Resize the Card2D and reposition internal elements.

        Parameters
        ----------
        size : (int, int)
            Card size (width, height) in pixels.
        """
        self.card_size = size
        self.panel.resize(size)

        self._image_size, _title_box_size, _body_box_size = self._calculate_sizes(size)

        bw = int(self.border_width)
        img_h = self._image_size[1]
        title_h = _title_box_size[1]

        _img_coords = (bw, bw)
        _title_coords = (self.padding, img_h + self.padding + bw)
        _body_coords = (
            self.padding,
            img_h + self.padding + title_h + self.padding + bw,
        )

        self.panel.update_element(self.image, _img_coords)
        self.panel.update_element(self.title_box, _title_coords)
        self.panel.update_element(self.body_box, _body_coords)

        self.image.resize(self._image_size)
        self.title_box.resize(_title_box_size)
        self.body_box.resize(_body_box_size)

    def _update_actors_position(self):
        """Update the internal position of the UI element."""
        self.panel.set_position(self.get_position())

    def update_layout(self):
        """
        Propagate layout updates to child text elements.

        The render loop calls this method on top-level UI elements so
        that :class:`TextBlock2D` children can re-align their text
        actors once the actual bounding box is known after the first
        render pass.
        """
        self.title_box.update_layout()
        self.body_box.update_layout()

    @property
    def color(self):
        """
        Get the background color of the card.

        Returns
        -------
        (float, float, float)
            RGB color of the card background.
        """
        return self.panel.color

    @color.setter
    def color(self, color):
        """
        Set the background color of the card.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            A hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
        """
        self.panel.color = normalize_colors(color)[0]

    @property
    def body(self):
        """
        Get the body text of the card.

        Returns
        -------
        str
            The body text.
        """
        return self.body_box.message

    @body.setter
    def body(self, text):
        """
        Set the body text of the card.

        Parameters
        ----------
        text : str
            The new body text.
        """
        self.body_box.message = text

    @property
    def title(self):
        """
        Get the title text of the card.

        Returns
        -------
        str
            The title text.
        """
        return self.title_box.message

    @title.setter
    def title(self, text):
        """
        Set the title text of the card.

        Parameters
        ----------
        text : str
            The new title text.
        """
        self.title_box.message = text

    @property
    def opacity(self):
        """
        Get the opacity of the card.

        Returns
        -------
        float
            The opacity of the card.
        """
        return self.panel.opacity

    @opacity.setter
    def opacity(self, value):
        """
        Set the opacity of the card.

        Parameters
        ----------
        value : float
            The new opacity of the card.
        """
        self.panel.opacity = np.clip(value, 0, 1)

    def left_button_pressed(self, event):
        """
        Handle left mouse button press event for card dragging.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        click_pos = np.array([event.x, event.y])
        self._drag_offset = click_pos - self.get_position()

    def left_button_dragged(self, event):
        """
        Handle left mouse button drag event for card movement.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self._drag_offset is not None:
            click_position = np.array([event.x, event.y])
            new_position = click_position - self._drag_offset
            self.set_position(new_position)

    def _calculate_sizes(self, size):
        """
        Calculate internal layout sizes based on the given card size.

        Parameters
        ----------
        size : (int, int)
            Card size (width, height) in pixels.

        Returns
        -------
        tuple
            Tuple of (image_size, title_box_size, body_box_size) where each
            is a tuple of (width, height) in pixels.
        """
        _width, _height = size
        bw = int(self.border_width)

        img_w = max(_width - 2 * bw, 1)
        img_h = max(int(self.image_scale * _height), 1)
        image_size = (img_w, img_h)

        text_area_w = max(_width - 2 * self.padding, 1)
        remaining_h = max(_height - img_h - 3 * self.padding, 2)

        title_h = max(int(remaining_h * 0.25), 1)
        body_h = max(remaining_h - title_h - self.padding, 1)

        title_box_size = (text_area_w, title_h)
        body_box_size = (text_area_w, body_h)

        return image_size, title_box_size, body_box_size


# class SpinBox(UI):
#     """SpinBox UI."""

#     @warn_on_args_to_kwargs()
#     def __init__(
#         self,
#         *,
#         position=(350, 400),
#         size=(300, 100),
#         padding=10,
#         panel_color=(1, 1, 1),
#         min_val=0,
#         max_val=100,
#         initial_val=50,
#         step=1,
#         max_column=10,
#         max_line=2,
#     ):
#         """Init this UI element.

#         Parameters
#         ----------
#         position : (int, int), optional
#             Absolute coordinates (x, y) of the lower-left corner of this
#             UI component.
#         size : (int, int), optional
#             Width and height in pixels of this UI component.
#         padding : int, optional
#             Distance between TextBox and Buttons.
#         panel_color : (float, float, float), optional
#             Panel color of SpinBoxUI.
#         min_val: int, optional
#             Minimum value of SpinBoxUI.
#         max_val: int, optional
#             Maximum value of SpinBoxUI.
#         initial_val: int, optional
#             Initial value of SpinBoxUI.
#         step: int, optional
#             Step value of SpinBoxUI.
#         max_column: int, optional
#             Max number of characters in a line.
#         max_line: int, optional
#             Max number of lines in the textbox.

#         """
#         self.panel_size = size
#         self.padding = padding
#         self.panel_color = panel_color
#         self.min_val = min_val
#         self.max_val = max_val
#         self.step = step
#         self.max_column = max_column
#         self.max_line = max_line

#         super(SpinBox, self).__init__(position=position)
#         self.value = initial_val
#         self.resize(size)

#         self.on_change = lambda ui: None

#     def _setup(self):
#         """Setup this UI component.

#         Create the SpinBoxUI with Background (Panel2D) and InputBox (TextBox2D)
#         and Increment,Decrement Button (Button2D).
#         """
#         self.panel = Panel2D(size=self.panel_size, color=self.panel_color)

#         self.textbox = TextBox2D(width=self.max_column, height=self.max_line)
#         self.textbox.text.dynamic_bbox = False
#         self.textbox.text.auto_font_scale = True
#         self.increment_button = Button2D(
#             icon_fnames=[("up", read_viz_icons(fname="circle-up.png"))]
#         )
#         self.decrement_button = Button2D(
#             icon_fnames=[("down", read_viz_icons(fname="circle-down.png"))]
#         )

#         self.panel.add_element(self.textbox, (0, 0))
#         self.panel.add_element(self.increment_button, (0, 0))
#         self.panel.add_element(self.decrement_button, (0, 0))

#         # Adding button click callbacks
#         self.increment_button.on_left_mouse_button_pressed = self.increment_callback
#         self.decrement_button.on_left_mouse_button_pressed = self.decrement_callback
#         self.textbox.off_focus = self.textbox_update_value

#     def resize(self, size):
#         """Resize SpinBox.

#         Parameters
#         ----------
#         size : (float, float)
#             SpinBox size(width, height) in pixels.

#         """
#         self.panel_size = size
#         self.textbox_size = (int(0.7 * size[0]), int(0.8 * size[1]))
#         self.button_size = (int(0.2 * size[0]), int(0.3 * size[1]))
#         self.padding = int(0.03 * self.panel_size[0])

#         self.panel.resize(size)
#         self.textbox.text.resize(self.textbox_size)
#         self.increment_button.resize(self.button_size)
#         self.decrement_button.resize(self.button_size)

#         textbox_pos = (self.padding, int((size[1] - self.textbox_size[1]) / 2))
#         inc_btn_pos = (
#             size[0] - self.padding - self.button_size[0],
#             int((1.5 * size[1] - self.button_size[1]) / 2),
#         )
#         dec_btn_pos = (
#             size[0] - self.padding - self.button_size[0],
#             int((0.5 * size[1] - self.button_size[1]) / 2),
#         )

#         self.panel.update_element(self.textbox, textbox_pos)
#         self.panel.update_element(self.increment_button, inc_btn_pos)
#         self.panel.update_element(self.decrement_button, dec_btn_pos)

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         return self.panel.actors

#     def _add_to_scene(self, scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : Scene

#         """
#         self.panel.add_to_scene(scene)

#     def _get_size(self):
#         return self.panel.size

#     def _set_position(self, coords):
#         """Set the lower-left corner position of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         self.panel.center = coords

#     def increment_callback(self, i_ren, _obj, _button):
#         self.increment()
#         i_ren.force_render()
#         i_ren.event.abort()

#     def decrement_callback(self, i_ren, _obj, _button):
#         self.decrement()
#         i_ren.force_render()
#         i_ren.event.abort()

#     @property
#     def value(self):
#         return self._value

#     @value.setter
#     def value(self, value):
#         if value >= self.max_val:
#             self._value = self.max_val
#         elif value <= self.min_val:
#             self._value = self.min_val
#         else:
#             self._value = value

#         self.textbox.set_message(str(self._value))

#     def validate_value(self, value):
#         """Validate and convert the given value into integer.

#         Parameters
#         ----------
#         value : str
#             Input value received from the textbox.

#         Returns
#         -------
#         int
#             If valid return converted integer else the previous value.

#         """
#         if value.isnumeric():
#             return int(value)

#         return self.value

#     def increment(self):
#         """Increment the current value by the step."""
#         current_val = self.validate_value(self.textbox.message)
#         self.value = current_val + self.step
#         self.on_change(self)

#     def decrement(self):
#         """Decrement the current value by the step."""
#         current_val = self.validate_value(self.textbox.message)
#         self.value = current_val - self.step
#         self.on_change(self)

#     def textbox_update_value(self, textbox):
#         self.value = self.validate_value(textbox.message)
#         self.on_change(self)
