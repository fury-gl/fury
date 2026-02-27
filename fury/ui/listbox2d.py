"""ListBox2D UI component for FURY v2."""

from fury.ui.containers import Panel2D
from fury.ui.core import UI, Rectangle2D, TextBlock2D


class ListBoxItem2D(UI):
    """A single item (slot) displayed inside a ListBox2D.

    Attributes
    ----------
    list_box : :class:`ListBox2D`
        The parent ListBox2D this item belongs to.
    selected : bool
        Whether this item is currently selected.

    Parameters
    ----------
    list_box : :class:`ListBox2D`
        The parent ListBox2D reference.
    size : (int, int)
        Width and height in pixels of this item.
    text_color : tuple of 3 floats, optional
        RGB color of the item text.
    selected_color : tuple of 3 floats, optional
        Background color when item is selected.
    unselected_color : tuple of 3 floats, optional
        Background color when item is not selected.
    background_opacity : float, optional
        Opacity of the item background.
    """

    def __init__(
        self,
        list_box,
        size,
        *,
        text_color=(0.2, 0.2, 0.2),
        selected_color=(0.9, 0.6, 0.6),
        unselected_color=(0.6, 0.6, 0.6),
        background_opacity=1.0,
    ):
        """Initialize a ListBoxItem2D.

        Parameters
        ----------
        list_box : :class:`ListBox2D`
            The parent ListBox2D reference.
        size : (int, int)
            Width and height in pixels of this item.
        text_color : tuple of 3 floats, optional
            RGB color of the item text.
        selected_color : tuple of 3 floats, optional
            Background color when item is selected.
        unselected_color : tuple of 3 floats, optional
            Background color when item is not selected.
        background_opacity : float, optional
            Opacity of the item background.
        """
        self._element = None
        self.list_box = list_box
        self.text_color = text_color
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        self.background_opacity = background_opacity
        self.selected = False
        super().__init__()
        self.resize(size)
        self.deselect()

    def _setup(self):
        """Set up background and text block for this item."""
        self.background = Rectangle2D(size=(1, 1))
        self.background.opacity = self.background_opacity
        self.textblock = TextBlock2D(
            justification="left",
            vertical_justification="middle",
            dynamic_bbox=True,
        )
        self.textblock.color = self.text_color
        self.background.on_left_mouse_button_clicked = self.left_button_clicked
        self.textblock.on_left_mouse_button_clicked = self.left_button_clicked

    def _get_actors(self):
        """Return actors composing this UI component.

        Returns
        -------
        list
            List of actors.
        """
        return self.background.actors + self.textblock.actors

    def _add_to_scene(self, scene):
        """Add subcomponents to the scene.

        Parameters
        ----------
        scene : scene
            The scene to add the component to.
        """
        self.background.add_to_scene(scene)
        self.textblock.add_to_scene(scene)

    def _get_size(self):
        """Return the size of this item.

        Returns
        -------
        (int, int)
            Width and height in pixels.
        """
        return self.background.size

    def _update_actors_position(self):
        """Update positions of background and textblock."""
        coords = self.get_position()
        self.background.set_position(coords)
        self.textblock.set_position(coords)

    def _set_position(self, coords):
        """Set position of this item.

        Parameters
        ----------
        coords : (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.background.set_position(coords)
        self.textblock.set_position(coords)

    def resize(self, size):
        """Resize this item.

        Parameters
        ----------
        size : (int, int)
            New (width, height) in pixels.
        """
        self.background.resize(size)

    def deselect(self):
        """Mark this item as deselected."""
        self.background.color = self.unselected_color
        self.textblock.bold = False
        self.selected = False

    def select(self):
        """Mark this item as selected."""
        self.background.color = self.selected_color
        self.textblock.bold = True
        self.selected = True

    @property
    def element(self):
        """Return the value associated with this slot.

        Returns
        -------
        object
            The value displayed by this slot.
        """
        return self._element

    @element.setter
    def element(self, element):
        """Set the value and update the displayed text.

        Parameters
        ----------
        element : object
            The value to display. Will be cast to str.
        """
        self._element = element
        self.textblock.message = "" if element is None else str(element)

    def left_button_clicked(self, event):
        """Handle left click and delegate selection to parent ListBox2D.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self.list_box.select(item=self)


class ListBox2D(UI):
    """A 2D UI component that displays a scrollable list of selectable items.

    Attributes
    ----------
    on_change : callable
        Callback invoked whenever the selection changes.

    Parameters
    ----------
    values : list
        Values used to populate the listbox. Objects must be castable to str.
    position : (float, float), optional
        Absolute coordinates (x, y) of the lower-left corner.
    size : (int, int), optional
        Width and height in pixels of this component.
    multiselection : bool, optional
        Whether multiple values can be selected simultaneously.
    reverse_scrolling : bool, optional
        If True, scroll direction is reversed.
    font_size : int, optional
        Font size of the list items in pixels.
    line_spacing : float, optional
        Spacing multiplier between items.
    text_color : tuple of 3 floats, optional
        RGB color of item text.
    selected_color : tuple of 3 floats, optional
        Background color of selected items.
    unselected_color : tuple of 3 floats, optional
        Background color of unselected items.
    scroll_bar_active_color : tuple of 3 floats, optional
        Color of the scroll bar when dragging.
    scroll_bar_inactive_color : tuple of 3 floats, optional
        Color of the scroll bar when idle.
    background_opacity : float, optional
        Opacity of item backgrounds.
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
        """Initialize a ListBox2D.

        Parameters
        ----------
        values : list
            Values used to populate the listbox.
        position : (float, float), optional
            Absolute coordinates (x, y) of the lower-left corner.
        size : (int, int), optional
            Width and height in pixels of this component.
        multiselection : bool, optional
            Whether multiple values can be selected simultaneously.
        reverse_scrolling : bool, optional
            If True, scroll direction is reversed.
        font_size : int, optional
            Font size of the list items in pixels.
        line_spacing : float, optional
            Spacing multiplier between items.
        text_color : tuple of 3 floats, optional
            RGB color of item text.
        selected_color : tuple of 3 floats, optional
            Background color of selected items.
        unselected_color : tuple of 3 floats, optional
            Background color of unselected items.
        scroll_bar_active_color : tuple of 3 floats, optional
            Color of the scroll bar when dragging.
        scroll_bar_inactive_color : tuple of 3 floats, optional
            Color of the scroll bar when idle.
        background_opacity : float, optional
            Opacity of item backgrounds.
        """
        self.view_offset = 0
        self.slots = []
        self.selected = []
        self.panel_size = size
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.slot_height = int(font_size * line_spacing)
        self.text_color = text_color
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        self.background_opacity = background_opacity
        self.values = list(values)
        self.multiselection = multiselection
        self.last_selection_idx = 0
        self.reverse_scrolling = reverse_scrolling
        super().__init__(position=position)

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
        self.scroll_init_position = 0
        self.update()
        self.on_change = lambda: None

    def _setup(self):
        """Set up the panel, scroll bar and item slots."""
        self.margin = 10
        size = self.panel_size
        self.nb_slots = int((size[1] - 2 * self.margin) // self.slot_height)
        self.panel = Panel2D(size=size, color=(1, 1, 1))

        scroll_bar_height = (
            self.nb_slots * (size[1] - 2 * self.margin) / len(self.values)
        )
        self.scroll_bar = Rectangle2D(size=(int(size[0] / 20), int(scroll_bar_height)))
        if len(self.values) <= self.nb_slots:
            self.scroll_bar.set_visibility(False)
            self.scroll_bar.height = 0

        scroll_bar_x = size[0] - self.scroll_bar.size[0] - self.margin
        scroll_bar_y = size[1] - self.scroll_bar.size[1] - self.margin
        self.panel.add_element(self.scroll_bar, (int(scroll_bar_x), int(scroll_bar_y)))

        self.slot_width = (
            size[0] - self.scroll_bar.size[0] - 2 * self.margin - self.margin
        )

        x = self.margin
        y = size[1] - self.margin
        for _ in range(self.nb_slots):
            y -= self.slot_height
            item = ListBoxItem2D(
                list_box=self,
                size=(self.slot_width, self.slot_height),
                text_color=self.text_color,
                selected_color=self.selected_color,
                unselected_color=self.unselected_color,
                background_opacity=self.background_opacity,
            )
            item.textblock.font_size = self.font_size
            self.slots.append(item)
            self.panel.add_element(item, (int(x), int(y + self.margin)))

        self.scroll_bar.on_left_mouse_button_pressed = self.scroll_click_callback
        self.scroll_bar.on_left_mouse_button_released = self.scroll_release_callback
        self.scroll_bar.on_left_mouse_button_dragged = self.scroll_drag_callback

        # TODO: wire up mouse wheel scrolling once WheelEvent is supported
        # in the v2 UI event system (fury/lib.py has gfx.WheelEvent available
        # but core.py does not yet define on_mouse_wheel_up/down callbacks).

    def _get_actors(self):
        """Return actors composing this UI component.

        Returns
        -------
        list
            List of actors.
        """
        return self.panel.actors

    def _add_to_scene(self, scene):
        """Add all subcomponents to the scene.

        Parameters
        ----------
        scene : scene
            The scene to add the component to.
        """
        self.panel.add_to_scene(scene)

    def _get_size(self):
        """Return the size of this component.

        Returns
        -------
        (int, int)
            Width and height in pixels.
        """
        return self.panel.size

    def _update_actors_position(self):
        """Update position of the panel."""
        self.panel.set_position(self.get_position())

    def _set_position(self, coords):
        """Set the lower-left corner position.

        Parameters
        ----------
        coords : (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.panel.set_position(coords)

    def resize(self, size):
        """Resize the listbox (no-op; resize on init only).

        Parameters
        ----------
        size : (int, int)
            New size (unused).
        """

    def up_button_callback(self, event):
        """Scroll the list up by one slot.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self.view_offset > 0:
            self.view_offset -= 1
            self.update()

    def down_button_callback(self, event):
        """Scroll the list down by one slot.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self.view_offset + self.nb_slots < len(self.values):
            self.view_offset += 1
            self.update()

    def scroll_click_callback(self, event):
        """Change scroll bar color on click.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self.scroll_bar.color = self.scroll_bar_active_color
        self.scroll_init_position = event.y

    def scroll_release_callback(self, event):
        """Restore scroll bar color on release.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self.scroll_bar.color = self.scroll_bar_inactive_color

    def scroll_drag_callback(self, event):
        """Drag the scroll bar to scroll the list.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        offset = int((event.y - self.scroll_init_position) / self.scroll_step_size)
        if offset > 0 and self.view_offset > 0:
            offset = min(offset, self.view_offset)
        elif offset < 0 and (self.view_offset + self.nb_slots < len(self.values)):
            offset = max(
                offset,
                -(len(self.values) - self.nb_slots - self.view_offset),
            )
        else:
            return
        self.view_offset -= offset
        self.update()
        self.scroll_init_position += offset * self.scroll_step_size

    def update(self):
        """Refresh the visible slots to match the current view_offset."""
        view_start = self.view_offset
        view_end = view_start + self.nb_slots
        values_to_show = self.values[view_start:view_end]
        for i, choice in enumerate(values_to_show):
            slot = self.slots[i]
            slot.element = choice
            slot.set_visibility(True)
            if slot.size[1] != self.slot_height:
                slot.resize((self.slot_width, self.slot_height))
            if slot.element in self.selected:
                slot.select()
            else:
                slot.deselect()
        for slot in self.slots[len(values_to_show) :]:
            slot.element = None
            slot.set_visibility(False)
            slot.resize((self.slot_width, 0))
            slot.deselect()

    def update_scrollbar(self):
        """Recalculate and reposition the scroll bar after values change."""
        self.scroll_bar.set_visibility(True)
        self.scroll_bar.height = int(
            self.nb_slots * (self.panel_size[1] - 2 * self.margin) / len(self.values)
        )
        denom = len(self.values) - self.nb_slots
        if not denom:
            denom += 1
        self.scroll_step_size = (
            self.slot_height * self.nb_slots - self.scroll_bar.height
        ) / denom
        scroll_bar_x = self.panel_size[0] - self.scroll_bar.size[0] - self.margin
        scroll_bar_y = self.panel_size[1] - self.scroll_bar.size[1] - self.margin
        self.panel.update_element(
            self.scroll_bar, (int(scroll_bar_x), int(scroll_bar_y))
        )
        if len(self.values) <= self.nb_slots:
            self.scroll_bar.set_visibility(False)
            self.scroll_bar.height = 0

    def clear_selection(self):
        """Clear all selected items."""
        del self.selected[:]

    def select(self, item, *, multiselect=False, range_select=False):
        """Select an item in the listbox.

        Parameters
        ----------
        item : :class:`ListBoxItem2D`
            The item to select.
        multiselect : bool, optional
            If True and multiselection is enabled, add to the current
            selection rather than replacing it.
        range_select : bool, optional
            If True and multiselection is enabled, select all items
            between the last selected item and this one.
        """
        if item.element not in self.values:
            return
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
        self.on_change()
        self.update()
