"""UI container module."""

# from warnings import warn

import numpy as np

from fury.ui.core import UI, Anchor, Rectangle2D


class Panel2D(UI):
    """A 2D UI Panel.

    Can contain one or more UI elements.

    Attributes
    ----------
    alignment : [left, right]
        Alignment of the panel with respect to the overall screen.

    Parameters
    ----------
    size : (int, int)
        Size (width, height) in pixels of the panel.
    position : (float, float)
        Absolute coordinates (x, y) of the lower-left corner of the panel.
    color : (float, float, float)
        Must take values in [0, 1].
    opacity : float
        Must take values in [0, 1].
    align : [left, right]
        Alignment of the panel with respect to the overall screen.
    border_color : (float, float, float), optional
        RGB color of the border. Must take values in [0, 1].
    border_width : float, optional
        Width of the border.
    has_border : bool, optional
        If the panel should have borders.
    """

    def __init__(
        self,
        size,
        *,
        position=(0, 0),
        color=(0.1, 0.1, 0.1),
        opacity=0.7,
        align="left",
        border_color=(1, 1, 1),
        border_width=0,
        has_border=False,
    ):
        """Initialize class instance.

        Parameters
        ----------
        size : (int, int)
            Size (width, height) in pixels of the panel.
        position : (float, float), optional
            Absolute coordinates (x, y) of the lower-left corner of the panel.
        color : (float, float, float), optional
            Must take values in [0, 1].
        opacity : float, optional
            Must take values in [0, 1].
        align : [left, right], optional
            Alignment of the panel with respect to the overall screen.
        border_color : (float, float, float), optional
            RGB color of the border. Must take values in [0, 1].
        border_width : float, optional
            Width of the border.
        has_border : bool, optional
            If the panel should have borders.
        """
        self.border_sides = ["left", "right", "top", "bottom"]
        self.border_coords = {
            "left": (0.0, 0.0),
            "right": (1.0, 0.0),
            "top": (0.0, 0.0),
            "bottom": (0.0, 1.0),
        }
        self.has_border = has_border
        self._border_color = border_color
        self._border_width = border_width
        super(Panel2D, self).__init__(position=position)
        self.resize(size)
        self.alignment = align
        self.color = color
        self.opacity = opacity
        self._drag_offset = None

    def _setup(self):
        """Set up this UI component.

        Create the background (Rectangle2D) of the panel and initialize the
        border elements (Rectangle2D).
        """
        self._elements = []
        self.element_offsets = []
        self.background = Rectangle2D(size=(1, 1))

        if self.has_border:
            self.borders = {
                "left": Rectangle2D(size=(1, 1)),
                "right": Rectangle2D(size=(1, 1)),
                "top": Rectangle2D(size=(1, 1)),
                "bottom": Rectangle2D(size=(1, 1)),
            }

            for key in self.borders.keys():
                self.borders[key].color = self._border_color
                self.borders[
                    key
                ].on_left_mouse_button_pressed = self.left_button_pressed

                self.borders[
                    key
                ].on_left_mouse_button_dragged = self.left_button_dragged
                self.add_element(
                    self.borders[key], self.border_coords[key], _is_internal=True
                )
        else:
            # FIX: initialize empty borders dict when has_border=False
            # so other methods don't crash when accessing self.borders
            self.borders = {}

        self.add_element(self.background, (0, 0), _is_internal=True)

        self.background.on_left_mouse_button_pressed = self.left_button_pressed
        self.background.on_left_mouse_button_dragged = self.left_button_dragged

    def _get_actors(self):
        """Get actors composing this UI component.

        Returns
        -------
        list
            List of actors composing this UI component.
        """
        actors = []

        actors.extend(self.background.actors)
        # FIX: only iterate borders if has_border is True
        if self.has_border:
            for border in self.borders.values():
                actors.extend(border.actors)

        return actors

    def _get_size(self):
        """Get the actual size of the panel.

        Returns
        -------
        (float, float)
            The (width, height) size of the panel.
        """
        return self.background.size

    def resize(self, size):
        """Set the panel size.

        Parameters
        ----------
        size : (float, float)
            Panel size (width, height) in pixels.
        """
        self.background.resize(size)

        if self.has_border:
            self.borders["left"].resize(
                (self._border_width, size[1] + self._border_width)
            )

            self.borders["right"].resize(
                (self._border_width, size[1] + self._border_width)
            )

            self.borders["top"].resize(
                (self.size[0] + self._border_width, self._border_width)
            )

            self.borders["bottom"].resize(
                (self.size[0] + self._border_width, self._border_width)
            )

            self.update_border_coords()

    def _update_actors_position(self):
        """Update the position of the internal actors."""
        coords = self.get_position()

        for element, offset in self.element_offsets:
            if element == self.background:
                element.z_order = self.z_order
            # FIX: only check borders dict if has_border is True
            elif self.has_border and element in self.borders.values():
                element.z_order = self.z_order + 1
            else:
                element.z_order = self.z_order + 2

            element.set_position(coords + offset)

    def set_visibility(self, visibility):
        """Set visibility of this UI component.

        Parameters
        ----------
        visibility : bool
            If True, the panel and its elements will be visible. If False, it will
            be hidden.
        """
        for element in self._elements:
            element.set_visibility(visibility)

    @property
    def color(self):
        """Get the background color of the panel.

        Returns
        -------
        (float, float, float)
            RGB color of the panel background.
        """
        return self.background.color

    @color.setter
    def color(self, color):
        """Set the background color of the panel.

        Parameters
        ----------
        color : (float, float, float)
            New RGB color of the panel background. Must take values in [0, 1].
        """
        self.background.color = color

    @property
    def opacity(self):
        """Get the opacity of the panel.

        Returns
        -------
        float
            Opacity value.
        """
        return self.background.opacity

    @opacity.setter
    def opacity(self, opacity):
        """Set the opacity of the panel.

        Parameters
        ----------
        opacity : float
            New opacity value.
        """
        self.background.opacity = opacity

    def add_element(self, element, coords, *, anchor="position", _is_internal=False):
        """Add a UI component to the panel.

        The coordinates represent an offset from the lower left corner of the
        panel.

        Parameters
        ----------
        element : UI
            The UI item to be added.
        coords : (float, float) or (int, int)
            If float, normalized coordinates are assumed and they must be
            between [0,1].
            If int, pixels coordinates are assumed and it must fit within the
            panel's size.
        anchor : str, optional
            Supported anchors are 'position' (top-left) and 'center'.
        _is_internal : bool, optional
            Flag used to distinguish between user-added elements
            and internal elements added by the Panel itself.

        Raises
        ------
        ValueError
            If coordinates are normalized but outside the [0,1] range, or if
            an unknown anchor is provided.
        """
        coords = np.array(coords)

        if np.issubdtype(coords.dtype, np.floating):
            if np.any(coords < 0) or np.any(coords > 1):
                raise ValueError("Normalized coordinates must be in [0,1].")

            coords = coords * self.size

        if anchor == "center":
            element.set_position(
                self.get_position() + coords,
                x_anchor=Anchor.CENTER,
                y_anchor=Anchor.CENTER,
            )
        elif anchor == "position":
            element.set_position(
                self.get_position() + coords,
            )
        else:
            msg = f"Unknown anchor {anchor}. Supported anchors are 'position' and \
                'center'."
            raise ValueError(msg)

        self._elements.append(element)
        if not _is_internal:
            self._children.append(element)
        offset = element.get_position() - self.get_position()
        self.element_offsets.append((element, offset))

    def remove_element(self, element):
        """Remove a UI component from the panel.

        Parameters
        ----------
        element : UI
            The UI item to be removed.

        Raises
        ------
        ValueError
            If the element is not found in the panel's elements list.
        """
        idx = self._elements.index(element)
        del self._elements[idx]
        del self.element_offsets[idx]
        if element in self._children:
            self._children.remove(element)

    def update_element(self, element, coords, *, anchor="position"):
        """Update the position of a UI component in the panel.

        Parameters
        ----------
        element : UI
            The UI item to be updated.
        coords : (float, float) or (int, int)
            New coordinates.
            If float, normalized coordinates are assumed and they must be
            between [0,1].
            If int, pixel coordinates are assumed and it must fit within the
            panel's size.
        anchor : str, optional
            Supported anchors are 'position' (top-left) and 'center'.

        Raises
        ------
        ValueError
            If coordinates are normalized but outside the [0,1] range, or if
            an unknown anchor is provided.
        """
        self.remove_element(element)
        self.add_element(element, coords, anchor=anchor)

    def left_button_pressed(self, event):
        """Handle left mouse button press event for panel.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        click_pos = np.array([event.x, event.y])
        self._drag_offset = click_pos - self.get_position()

    def left_button_dragged(self, event):
        """Handle left mouse button drag event for panel movement.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self._drag_offset is not None:
            click_position = np.array([event.x, event.y])
            new_position = click_position - self._drag_offset
            self.set_position(new_position)

    def re_align(self, window_size_change):
        """Re-organise the elements in case the window size is changed.

        Parameters
        ----------
        window_size_change : (int, int)
            New window size (width, height) in pixels.

        Raises
        ------
        ValueError
            If alignment is not 'left' or 'right'.
        """
        if self.alignment == "left":
            pass
        elif self.alignment == "right":
            self.set_position(window_size_change)
        else:
            msg = "You can only left-align or right-align objects in a panel."
            raise ValueError(msg)

    def update_border_coords(self):
        """Update the coordinates of the borders."""

        for key in self.borders.keys():
            self.update_element(self.borders[key], self.border_coords[key])

    @property
    def border_color(self):
        """Get the current color of all four borders.

        Returns
        -------
        list
            A list containing the color (RGB tuple) of the left, right, top, and bottom
            borders, respectively.
        """

        return [self.borders[side].color for side in self.border_sides]

    @border_color.setter
    def border_color(self, side_color):
        """Set the color of a specific border.

        Parameters
        ----------
        side_color : Iterable
            Iterable `[side, color]` containing the side (str) and color (RGB tuple).
        """
        side, color = side_color

        if side.lower() not in ["left", "right", "top", "bottom"]:
            raise ValueError(f"{side} not a valid border side")

        self.borders[side].color = color

    @property
    def border_width(self):
        """Get the current width/height of the borders.

        Returns
        -------
        list
            A list containing the width (for left/right) and height (for top/bottom)
            of the borders.
        """

        widths = []

        for side in self.border_sides:
            if side in ["left", "right"]:
                widths.append(self.borders[side].width)
            elif side in ["top", "bottom"]:
                widths.append(self.borders[side].height)
            else:
                raise ValueError(f"{side} not a valid border side")
        return widths

    @border_width.setter
    def border_width(self, side_width):
        """Set the width of a specific border.

        Parameters
        ----------
        side_width : Iterable
            Iterable `[side, width]` containing the side (str) and the width (float).
        """
        side, border_width = side_width

        if side.lower() in ["left", "right"]:
            self.borders[side].width = border_width
        elif side.lower() in ["top", "bottom"]:
            self.borders[side].height = border_width
        else:
            raise ValueError(f"{side} not a valid border side")
