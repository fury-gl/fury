"""UI container module."""

__all__ = ['Panel2D', 'TabPanel2D', 'TabUI', 'ImageContainer2D', 'GridUI']

from warnings import warn

import numpy as np

from fury.actor import grid
from fury.io import load_image
from fury.lib import (
    CellArray,
    FloatArray,
    Points,
    PolyData,
    PolyDataMapper2D,
    Property2D,
    Texture,
    TexturedActor2D,
)
from fury.ui.core import UI, Rectangle2D, TextBlock2D
from fury.utils import rotate, set_input


class Panel2D(UI):
    """A 2D UI Panel.

    Can contain one or more UI elements.

    Attributes
    ----------
    alignment : [left, right]
        Alignment of the panel with respect to the overall screen.

    """

    def __init__(
        self,
        size,
        position=(0, 0),
        color=(0.1, 0.1, 0.1),
        opacity=0.7,
        align='left',
        border_color=(1, 1, 1),
        border_width=0,
        has_border=False,
    ):
        """Init class instance.

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
        border_color: (float, float, float), optional
            Must take values in [0, 1].
        border_width: float, optional
            width of the border
        has_border: bool, optional
            If the panel should have borders.

        """
        self.has_border = has_border
        self._border_color = border_color
        self._border_width = border_width
        super(Panel2D, self).__init__(position)
        self.resize(size)
        self.alignment = align
        self.color = color
        self.opacity = opacity
        self.position = position
        self._drag_offset = None

    def _setup(self):
        """Setup this UI component.

        Create the background (Rectangle2D) of the panel.
        Create the borders (Rectangle2D) of the panel.
        """
        self._elements = []
        self.element_offsets = []
        self.background = Rectangle2D()

        if self.has_border:
            self.borders = {
                'left': Rectangle2D(),
                'right': Rectangle2D(),
                'top': Rectangle2D(),
                'bottom': Rectangle2D(),
            }

            self.border_coords = {
                'left': (0.0, 0.0),
                'right': (1.0, 0.0),
                'top': (0.0, 1.0),
                'bottom': (0.0, 0.0),
            }

            for key in self.borders.keys():
                self.borders[key].color = self._border_color
                self.add_element(self.borders[key], self.border_coords[key])

            for key in self.borders.keys():
                self.borders[
                    key
                ].on_left_mouse_button_pressed = self.left_button_pressed

                self.borders[
                    key
                ].on_left_mouse_button_dragged = self.left_button_dragged

        self.add_element(self.background, (0, 0))

        # Add default events listener for this UI component.
        self.background.on_left_mouse_button_pressed = self.left_button_pressed
        self.background.on_left_mouse_button_dragged = self.left_button_dragged

    def _get_actors(self):
        """Get the actors composing this UI component."""
        actors = []
        for element in self._elements:
            actors += element.actors

        return actors

    def _add_to_scene(self, scene):
        """Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene

        """
        for element in self._elements:
            element.add_to_scene(scene)

    def _get_size(self):
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
            self.borders['left'].resize(
                (self._border_width, size[1] + self._border_width)
            )

            self.borders['right'].resize(
                (self._border_width, size[1] + self._border_width)
            )

            self.borders['top'].resize(
                (self.size[0] + self._border_width, self._border_width)
            )

            self.borders['bottom'].resize(
                (self.size[0] + self._border_width, self._border_width)
            )

            self.update_border_coords()

    def _set_position(self, coords):
        """Set the lower-left corner position of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).

        """
        coords = np.array(coords)
        for element, offset in self.element_offsets:
            element.position = coords + offset

    def set_visibility(self, visibility):
        for element in self._elements:
            element.set_visibility(visibility)

    @property
    def color(self):
        return self.background.color

    @color.setter
    def color(self, color):
        self.background.color = color

    @property
    def opacity(self):
        return self.background.opacity

    @opacity.setter
    def opacity(self, opacity):
        self.background.opacity = opacity

    def add_element(self, element, coords, anchor='position'):
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

        """
        coords = np.array(coords)

        if np.issubdtype(coords.dtype, np.floating):
            if np.any(coords < 0) or np.any(coords > 1):
                raise ValueError('Normalized coordinates must be in [0,1].')

            coords = coords * self.size

        if anchor == 'center':
            element.center = self.position + coords
        elif anchor == 'position':
            element.position = self.position + coords
        else:
            msg = "Unknown anchor {}. Supported anchors are 'position'" " and 'center'."
            raise ValueError(msg)

        self._elements.append(element)
        offset = element.position - self.position
        self.element_offsets.append((element, offset))

    def remove_element(self, element):
        """Remove a UI component from the panel.

        Parameters
        ----------
        element : UI
            The UI item to be removed.

        """
        idx = self._elements.index(element)
        del self._elements[idx]
        del self.element_offsets[idx]

    def update_element(self, element, coords, anchor='position'):
        """Update the position of a UI component in the panel.

        Parameters
        ----------
        element : UI
            The UI item to be updated.
        coords : (float, float) or (int, int)
            New coordinates.
            If float, normalized coordinates are assumed and they must be
            between [0,1].
            If int, pixels coordinates are assumed and it must fit within the
            panel's size.

        """
        self.remove_element(element)
        self.add_element(element, coords, anchor)

    def left_button_pressed(self, i_ren, _obj, panel2d_object):
        click_pos = np.array(i_ren.event.position)
        self._drag_offset = click_pos - self.position
        i_ren.event.abort()  # Stop propagating the event.

    def left_button_dragged(self, i_ren, _obj, _panel2d_object):
        if self._drag_offset is not None:
            click_position = np.array(i_ren.event.position)
            new_position = click_position - self._drag_offset
            self.position = new_position
        i_ren.force_render()

    def re_align(self, window_size_change):
        """Re-organise the elements in case the window size is changed.

        Parameters
        ----------
        window_size_change : (int, int)
            New window size (width, height) in pixels.

        """
        if self.alignment == 'left':
            pass
        elif self.alignment == 'right':
            self.position += np.array(window_size_change)
        else:
            msg = 'You can only left-align or right-align objects in a panel.'
            raise ValueError(msg)

    def update_border_coords(self):
        """Update the coordinates of the borders"""
        self.border_coords = {
            'left': (0.0, 0.0),
            'right': (1.0, 0.0),
            'top': (0.0, 1.0),
            'bottom': (0.0, 0.0),
        }

        for key in self.borders.keys():
            self.update_element(self.borders[key], self.border_coords[key])

    @property
    def border_color(self):
        sides = ['left', 'right', 'top', 'bottom']
        return [self.borders[side].color for side in sides]

    @border_color.setter
    def border_color(self, side_color):
        """Set the color of a specific border

        Parameters
        ----------
        side_color: Iterable
            Iterable to pack side, color values

        """
        side, color = side_color

        if side.lower() not in ['left', 'right', 'top', 'bottom']:
            raise ValueError(f'{side} not a valid border side')

        self.borders[side].color = color

    @property
    def border_width(self):
        sides = ['left', 'right', 'top', 'bottom']
        widths = []

        for side in sides:
            if side in ['left', 'right']:
                widths.append(self.borders[side].width)
            elif side in ['top', 'bottom']:
                widths.append(self.borders[side].height)
            else:
                raise ValueError(f'{side} not a valid border side')
        return widths

    @border_width.setter
    def border_width(self, side_width):
        """Set the border width of a specific border

        Parameters
        ----------
        side_width: Iterable
            Iterable to pack side, width values

        """
        side, border_width = side_width

        if side.lower() in ['left', 'right']:
            self.borders[side].width = border_width
        elif side.lower() in ['top', 'bottom']:
            self.borders[side].height = border_width
        else:
            raise ValueError(f'{side} not a valid border side')


class TabPanel2D(UI):
    """Render content within a Tab.

    Attributes
    ----------
    content_panel: :class: 'Panel2D'
        Hold all the content UI components.
    text_block: :class: 'TextBlock2D'
        Renders the title of the tab.

    """

    def __init__(
        self,
        position=(0, 0),
        size=(100, 100),
        title='New Tab',
        color=(0.5, 0.5, 0.5),
        content_panel=None,
    ):
        """Init class instance.

        Parameters
        ----------
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of the
            UI component
        size : (int, int)
            Width and height of the pixels of this UI component.
        title : str
            Renders the title for Tab panel.
        color : list of 3 floats
            Background color of tab panel.
        content_panel : Panel2D
            Panel consisting of the content UI elements.

        """
        self.content_panel = content_panel
        self.panel_size = size
        self._text_size = (int(1.0 * size[0]), size[1])

        super(TabPanel2D, self).__init__()
        self.title = title
        self.panel.position = position
        self.color = color

    def _setup(self):
        """Setup this UI component.

        Create parent panel.
        Create Text to hold tab information.
        Create Button to close tab.

        """
        self.panel = Panel2D(size=self.panel_size)
        self.text_block = TextBlock2D(size=self._text_size, color=(0, 0, 0))
        self.panel.add_element(self.text_block, (0, 0))

    def _get_actors(self):
        """Get the actors composing this UI component."""
        return self.panel.actors + self.content_panel.actors

    def _add_to_scene(self, _scene):
        """Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene

        """
        self.panel.add_to_scene(_scene)
        self.content_panel.add_to_scene(_scene)

    def _set_position(self, _coords):
        """Set the lower-left corner position of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).

        """
        self.panel.position = _coords

    def _get_size(self):
        self.panel.size

    def resize(self, size):
        """Resize Tab panel.

        Parameters
        ----------
        size : (int, int)
            New width and height in pixels.

        """
        self._text_size = (int(0.7 * size[0]), size[1])
        self._button_size = (int(0.3 * size[0]), size[1])
        self.panel.resize(size)
        self.text_block.resize(self._text_size)

    @property
    def color(self):
        """Return the background color of tab panel."""
        return self.panel.color

    @color.setter
    def color(self, color):
        """Set background color of tab panel.

        Parameters
        ----------
        color : list of 3 floats.

        """
        self.panel.color = color

    @property
    def title(self):
        """Return the title of tab panel."""
        return self.text_block.message

    @title.setter
    def title(self, text):
        """Set the title of tab panel.

        Parameters
        ----------
        text : str
            New title for tab panel.

        """
        self.text_block.message = text

    @property
    def title_bold(self):
        """Is the title of a tab panel bold."""
        return self.text_block.bold

    @title_bold.setter
    def title_bold(self, bold):
        """Determine if the text title of a tab panel must be bold.

        Parameters
        ----------
        bold : bool
            Bold property for a text title in a tab panel.

        """
        self.text_block.bold = bold

    @property
    def title_color(self):
        """Return the title color of tab panel."""
        return self.text_block.color

    @title_color.setter
    def title_color(self, color):
        """Set the title color of tab panel.

        Parameters
        ----------
        color : tuple
            New title color for tab panel.

        """
        self.text_block.color = color

    @property
    def title_font_size(self):
        """Return the title font size of tab panel."""
        return self.text_block.font_size

    @title_font_size.setter
    def title_font_size(self, font_size):
        """Set the title font size of tab panel.

        Parameters
        ----------
        font_size : int
            New title font size for tab panel.

        """
        self.text_block.font_size = font_size

    @property
    def title_italic(self):
        """Is the title of a tab panel italic."""
        return self.text_block.italic

    @title_italic.setter
    def title_italic(self, italic):
        """Determine if the text title of a tab panel must be italic.

        Parameters
        ----------
        italic : bool
            Italic property for a text title in a tab panel.

        """
        self.text_block.italic = italic

    def add_element(self, element, coords, anchor='position'):
        """Add a UI component to the content panel.

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

        """
        element.set_visibility(False)
        self.content_panel.add_element(element, coords, anchor)

    def remove_element(self, element):
        """Remove a UI component from the content panel.

        Parameters
        ----------
        element : UI
            The UI item to be removed.

        """
        self.content_panel.remove_element(element)

    def update_element(self, element, coords, anchor='position'):
        """Update the position of a UI component in the content panel.

        Parameters
        ----------
        element : UI
            The UI item to be updated.
        coords : (float, float) or (int, int)
            New coordinates.
            If float, normalized coordinates are assumed and they must be
            between [0,1].
            If int, pixels coordinates are assumed and it must fit within the
            panel's size.

        """
        self.content_panel.update_element(element, coords, anchor='position')


class TabUI(UI):
    """UI element to add multiple panels within a single window.

    Attributes
    ----------
    tabs: :class: List of 'TabPanel2D'
        Stores all the instances of 'TabPanel2D' that renders the contents.

    """

    def __init__(
        self,
        position=(0, 0),
        size=(100, 100),
        nb_tabs=1,
        active_color=(1, 1, 1),
        inactive_color=(0.5, 0.5, 0.5),
        draggable=False,
        startup_tab_id=None,
        tab_bar_pos="top",
    ):
        """Init class instance.

        Parameters
        ----------
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of this
            UI component.
        size : (int, int)
            Width and height in pixels of this UI component.
        nb_tabs : int
            Number of tabs to be renders.
        active_color : tuple of 3 floats.
            Background color of active tab panel.
        inactive_color : tuple of 3 floats.
            Background color of inactive tab panels.
        draggable : bool
            Whether the UI element is draggable or not.
        startup_tab_id : int, optional
            Tab to be activated and uncollapsed on startup.
            by default None is activated/ all collapsed.
        tab_bar_pos : str, optional
            Position of the Tab Bar in the panel
        """
        self.tabs = []
        self.nb_tabs = nb_tabs
        self.parent_size = size
        self.content_size = (size[0], int(0.9 * size[1]))
        self.draggable = draggable
        self.active_color = active_color
        self.inactive_color = inactive_color
        self.active_tab_idx = startup_tab_id
        self.collapsed = True
        self.tab_bar_pos = tab_bar_pos

        super(TabUI, self).__init__()
        self.position = position

    def _setup(self):
        """Setup this UI component.

        Create parent panel.
        Create tab panels.
        """
        self.parent_panel = Panel2D(self.parent_size, opacity=0.0)

        # Offer some standard hooks to the user.
        self.on_change = lambda ui: None
        self.on_collapse = lambda ui: None

        for _ in range(self.nb_tabs):
            content_panel = Panel2D(size=self.content_size)
            content_panel.set_visibility(False)
            tab_panel = TabPanel2D(content_panel=content_panel)
            self.tabs.append(tab_panel)
        self.update_tabs()

        if self.active_tab_idx is not None:
            self.tabs[self.active_tab_idx].color = self.active_color
            self.tabs[self.active_tab_idx].content_panel.set_visibility(True)

    def _get_actors(self):
        """Get the actors composing this UI component."""
        actors = []
        actors += self.parent_panel.actors
        for tab_panel in self.tabs:
            actors += tab_panel.actors

        return actors

    def _add_to_scene(self, _scene):
        """Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene

        """
        self.parent_panel.add_to_scene(_scene)
        for tab_panel in self.tabs:
            tab_panel.add_to_scene(_scene)

    def _set_position(self, _coords):
        """Set the lower-left corner position of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).

        """
        self.parent_panel.position = _coords

    def _get_size(self):
        return self.parent_panel.size

    def update_tabs(self):
        """Update position, size and callbacks for tab panels."""
        self.tab_panel_size = (self.size[0] // self.nb_tabs, int(0.1 * self.size[1]))
        if self.tab_bar_pos.lower() not in ['top', 'bottom']:
            warn("tab_bar_pos can only have value top/bottom")
            self.tab_bar_pos = "top"

        if self.tab_bar_pos.lower() == "top":
            tab_panel_pos = [0.0, 0.9]
        elif self.tab_bar_pos.lower() == "bottom":
            tab_panel_pos = [0.0, 0.0]

        for tab_panel in self.tabs:
            tab_panel.resize(self.tab_panel_size)
            tab_panel.content_panel.position = self.position

            content_panel = tab_panel.content_panel
            if self.draggable:
                tab_panel.panel.background.on_left_mouse_button_pressed = \
                    self.left_button_pressed
                content_panel.background.on_left_mouse_button_pressed = \
                    self.left_button_pressed
                tab_panel.text_block.on_left_mouse_button_pressed = \
                    self.left_button_pressed

                tab_panel.panel.background.on_left_mouse_button_dragged = \
                    self.left_button_dragged
                content_panel.background.on_left_mouse_button_dragged = \
                    self.left_button_dragged
                tab_panel.text_block.on_left_mouse_button_dragged = \
                    self.left_button_dragged
            else:
                tab_panel.panel.background.on_left_mouse_button_dragged = \
                    lambda i_ren, _obj, _comp: i_ren.force_render
                content_panel.background.on_left_mouse_button_dragged = \
                    lambda i_ren, _obj, _comp: i_ren.force_render

            tab_panel.text_block.on_left_mouse_button_clicked = self.select_tab_callback
            tab_panel.panel.background.on_left_mouse_button_clicked = (
                self.select_tab_callback
            )

            tab_panel.text_block.on_right_mouse_button_clicked = self.collapse_tab_ui
            tab_panel.panel.background.on_right_mouse_button_clicked = (
                self.collapse_tab_ui
            )

            tab_panel.content_panel.resize(self.content_size)
            self.parent_panel.add_element(tab_panel, tab_panel_pos)
            if self.tab_bar_pos.lower() == "top":
                self.parent_panel.add_element(tab_panel.content_panel,
                                              (0.0, 0.0))
            elif self.tab_bar_pos.lower() == "bottom":
                self.parent_panel.add_element(tab_panel.content_panel,
                                              (0.0, 0.1))
            tab_panel_pos[0] += 1 / self.nb_tabs

    def select_tab_callback(self, iren, _obj, _tab_comp):
        """Handle events when a tab is selected."""
        for idx, tab_panel in enumerate(self.tabs):
            if (
                tab_panel.text_block is not _tab_comp
                and tab_panel.panel.background is not _tab_comp
            ):
                tab_panel.color = self.inactive_color
                tab_panel.content_panel.set_visibility(False)
            else:
                current_visibility = tab_panel.content_panel.actors[0].GetVisibility()
                if not current_visibility:
                    tab_panel.color = self.active_color
                else:
                    tab_panel.color = self.inactive_color
                tab_panel.content_panel.set_visibility(not current_visibility)
                self.active_tab_idx = idx

        self.collapsed = False
        self.on_change(self)
        iren.force_render()
        iren.event.abort()

    def collapse_tab_ui(self, iren, _obj, _tab_comp):
        """Handle events when Tab UI is collapsed."""
        if self.active_tab_idx is not None:
            active_tab_panel = self.tabs[self.active_tab_idx]
            active_tab_panel.color = self.inactive_color
            active_tab_panel.content_panel.set_visibility(False)
        self.active_tab_idx = None
        self.collapsed = True
        self.on_collapse(self)
        iren.force_render()
        iren.event.abort()

    def add_element(self, tab_idx, element, coords, anchor='position'):
        """Add element to content panel after checking its existence."""
        if tab_idx < self.nb_tabs and tab_idx >= 0:
            self.tabs[tab_idx].add_element(element, coords, anchor)
            if tab_idx == self.active_tab_idx:
                element.set_visibility(True)
        else:
            raise IndexError('Tab with index ' '{} does not exist'.format(tab_idx))

    def remove_element(self, tab_idx, element):
        """Remove element from content panel after checking its existence."""
        if tab_idx < self.nb_tabs and tab_idx >= 0:
            self.tabs[tab_idx].remove_element(element)
        else:
            raise IndexError('Tab with index ' '{} does not exist'.format(tab_idx))

    def update_element(self, tab_idx, element, coords, anchor='position'):
        """Update element on content panel after checking its existence."""
        if tab_idx < self.nb_tabs and tab_idx >= 0:
            self.tabs[tab_idx].update_element(element, coords, anchor)
        else:
            raise IndexError('Tab with index ' '{} does not exist'.format(tab_idx))

    def left_button_pressed(self, i_ren, _obj, _sub_component):
        click_pos = np.array(i_ren.event.position)
        self._click_position = click_pos
        i_ren.event.abort()  # Stop propagating the event.

    def left_button_dragged(self, i_ren, _obj, _sub_component):
        click_position = np.array(i_ren.event.position)
        change = click_position - self._click_position
        self.parent_panel.position += change
        self._click_position = click_position
        i_ren.force_render()


class ImageContainer2D(UI):
    """A 2D container to hold an image.

    Currently Supports:
    - png and jpg/jpeg images

    Attributes
    ----------
    size: (float, float)
        Image size (width, height) in pixels.
    img : ImageData
        The image loaded from the specified path.

    """

    def __init__(self, img_path, position=(0, 0), size=(100, 100)):
        """Init class instance.

        Parameters
        ----------
        img_path : string
            URL or local path of the image
        position : (float, float), optional
            Absolute coordinates (x, y) of the lower-left corner of the image.
        size : (int, int), optional
            Width and height in pixels of the image.

        """
        super(ImageContainer2D, self).__init__(position)
        self.img = load_image(img_path, as_vtktype=True)
        self.set_img(self.img)
        self.resize(size)

    def _get_size(self):
        lower_left_corner = self.texture_points.GetPoint(0)
        upper_right_corner = self.texture_points.GetPoint(2)
        size = np.array(upper_right_corner) - np.array(lower_left_corner)
        return abs(size[:2])

    def _setup(self):
        """Setup this UI Component.

        Return an image as a 2D actor with a specific position.

        Returns
        -------
        :class:`vtkTexturedActor2D`

        """
        self.texture_polydata = PolyData()
        self.texture_points = Points()
        self.texture_points.SetNumberOfPoints(4)

        polys = CellArray()
        polys.InsertNextCell(4)
        polys.InsertCellPoint(0)
        polys.InsertCellPoint(1)
        polys.InsertCellPoint(2)
        polys.InsertCellPoint(3)
        self.texture_polydata.SetPolys(polys)

        tc = FloatArray()
        tc.SetNumberOfComponents(2)
        tc.SetNumberOfTuples(4)
        tc.InsertComponent(0, 0, 0.0)
        tc.InsertComponent(0, 1, 0.0)
        tc.InsertComponent(1, 0, 1.0)
        tc.InsertComponent(1, 1, 0.0)
        tc.InsertComponent(2, 0, 1.0)
        tc.InsertComponent(2, 1, 1.0)
        tc.InsertComponent(3, 0, 0.0)
        tc.InsertComponent(3, 1, 1.0)
        self.texture_polydata.GetPointData().SetTCoords(tc)

        texture_mapper = PolyDataMapper2D()
        texture_mapper = set_input(texture_mapper, self.texture_polydata)

        image = TexturedActor2D()
        image.SetMapper(texture_mapper)

        self.texture = Texture()
        image.SetTexture(self.texture)

        image_property = Property2D()
        image_property.SetOpacity(1.0)
        image.SetProperty(image_property)
        self.actor = image

        # Add default events listener to the VTK actor.
        self.handle_events(self.actor)

    def _get_actors(self):
        """Return the actors that compose this UI component."""
        return [self.actor]

    def _add_to_scene(self, scene):
        """Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene

        """
        scene.add(self.actor)

    def resize(self, size):
        """Resize the image.

        Parameters
        ----------
        size : (float, float)
            image size (width, height) in pixels.

        """
        # Update actor.
        self.texture_points.SetPoint(0, 0, 0, 0.0)
        self.texture_points.SetPoint(1, size[0], 0, 0.0)
        self.texture_points.SetPoint(2, size[0], size[1], 0.0)
        self.texture_points.SetPoint(3, 0, size[1], 0.0)
        self.texture_polydata.SetPoints(self.texture_points)

    def _set_position(self, coords):
        """Set the lower-left corner position of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).

        """
        self.actor.SetPosition(*coords)

    def scale(self, factor):
        """Scale the image.

        Parameters
        ----------
        factor : (float, float)
            Scaling factor (width, height) in pixels.

        """
        self.resize(self.size * factor)

    def set_img(self, img):
        """Modify the image used by the vtkTexturedActor2D.

        Parameters
        ----------
        img : imageData

        """
        self.texture = set_input(self.texture, img)


class GridUI(UI):
    """Add actors in a grid and interact with them individually."""

    def __init__(
        self,
        actors,
        captions=None,
        caption_offset=(0, -100, 0),
        cell_padding=0,
        cell_shape='rect',
        aspect_ratio=16 / 9.0,
        dim=None,
        rotation_speed=1,
        rotation_axis=(0, 1, 0),
    ):

        # TODO: add rotation axis None by default

        self.container = grid(
            actors,
            captions=captions,
            caption_offset=caption_offset,
            cell_padding=cell_padding,
            cell_shape=cell_shape,
            aspect_ratio=aspect_ratio,
            dim=dim,
        )
        self._actors = []
        self._actors_dict = {}
        self.rotation_speed = rotation_speed
        self.rotation_axis = rotation_axis

        for item in self.container._items:
            actor = item if captions is None else item._items[0]
            self._actors.append(actor)
            self._actors_dict[actor] = {'x': -np.inf, 'y': -np.inf}

        super(GridUI, self).__init__(position=(0, 0, 0))

    def _get_size(self):
        return

    @staticmethod
    def left_click_callback(istyle, _obj, _what):
        istyle.trackball_actor.OnLeftButtonDown()
        istyle.force_render()
        istyle.event.abort()

    @staticmethod
    def left_release_callback(istyle, _obj, _what):

        istyle.trackball_actor.OnLeftButtonUp()
        istyle.force_render()
        istyle.event.abort()

    @staticmethod
    def mouse_move_callback(istyle, _obj, _what):
        istyle.trackball_actor.OnMouseMove()
        istyle.force_render()
        istyle.event.abort()

    @staticmethod
    def left_click_callback2(istyle, obj, self):

        rx, ry, rz = self.rotation_axis
        clockwise_rotation = np.array([self.rotation_speed, rx, ry, rz])
        rotate(obj, clockwise_rotation)

        istyle.force_render()
        istyle.event.abort()

    @staticmethod
    def left_release_callback2(istyle, _obj, _what):

        istyle.force_render()
        istyle.event.abort()

    @staticmethod
    def mouse_move_callback2(istyle, obj, self):

        if self._actors_dict[obj]['y'] == -np.inf:

            iren = istyle.GetInteractor()
            event_pos = iren.GetEventPosition()
            self._actors_dict[obj]['y'] = event_pos[1]

        else:

            iren = istyle.GetInteractor()
            event_pos = iren.GetEventPosition()
            rx, ry, rz = self.rotation_axis

            if event_pos[1] >= self._actors_dict[obj]['y']:
                clockwise_rotation = np.array([-self.rotation_speed, rx, ry, rz])
                rotate(obj, clockwise_rotation)
            else:
                anti_clockwise_rotation = np.array([self.rotation_speed, rx, ry, rz])
                rotate(obj, anti_clockwise_rotation)

            self._actors_dict[obj]['y'] = event_pos[1]

            istyle.force_render()
            istyle.event.abort()

    ANTICLOCKWISE_ROTATION_Y = np.array([-10, 0, 1, 0])
    CLOCKWISE_ROTATION_Y = np.array([10, 0, 1, 0])
    ANTICLOCKWISE_ROTATION_X = np.array([-10, 1, 0, 0])
    CLOCKWISE_ROTATION_X = np.array([10, 1, 0, 0])

    def key_press_callback(self, istyle, obj, _what):
        has_changed = False
        if istyle.event.key == 'Left':
            has_changed = True
            for a in self._actors:
                rotate(a, self.ANTICLOCKWISE_ROTATION_Y)
        elif istyle.event.key == 'Right':
            has_changed = True
            for a in self._actors:
                rotate(a, self.CLOCKWISE_ROTATION_Y)
        elif istyle.event.key == 'Up':
            has_changed = True
            for a in self._actors:
                rotate(a, self.ANTICLOCKWISE_ROTATION_X)
        elif istyle.event.key == 'Down':
            has_changed = True
            for a in self._actors:
                rotate(a, self.CLOCKWISE_ROTATION_X)

        if has_changed:
            istyle.force_render()

    def _setup(self):
        """Set up this UI component and the events of its actor."""
        # Add default events listener to the VTK actor.
        for actor in self._actors:
            # self.handle_events(actor)

            if self.rotation_axis is None:
                self.add_callback(
                    actor, 'LeftButtonPressEvent', self.left_click_callback
                )
                self.add_callback(
                    actor, 'LeftButtonReleaseEvent', self.left_release_callback
                )
                self.add_callback(actor, 'MouseMoveEvent', self.mouse_move_callback)
            else:
                self.add_callback(
                    actor, 'LeftButtonPressEvent', self.left_click_callback2
                )
                # TODO: possibly add this too
                self.add_callback(
                    actor, 'LeftButtonReleaseEvent', self.left_release_callback2
                )
                self.add_callback(actor, 'MouseMoveEvent', self.mouse_move_callback2)

            # TODO: this is currently not running
            self.add_callback(actor, 'KeyPressEvent', self.key_press_callback)
        # self.on_key_press = self.key_press_callback2

    def _get_actors(self):
        """Get the actors composing this UI component."""
        return self._actors

    def _add_to_scene(self, scene):
        """Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene

        """
        self.container.add_to_scene(scene)

    def resize(self, size):
        """Resize the button.

        Parameters
        ----------
        size : (float, float)
            Button size (width, height) in pixels.

        """
        # Update actor.
        pass

    def _set_position(self, coords):
        """Set the lower-left corner position of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).

        """
        # coords = (0, 0, 0)
        pass
        # self.actor.SetPosition(*coords)
        # self.container.SetPosition(*coords)
