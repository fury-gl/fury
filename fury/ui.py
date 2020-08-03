from collections import OrderedDict
from warnings import warn
from numbers import Number

import numpy as np
import vtk
import os
import abc

from fury.data import read_viz_icons
from fury.interactor import CustomInteractorStyle
from fury.io import load_image
from fury.utils import set_input, rotate
from fury.actor import grid


TWO_PI = 2 * np.pi


def clip_overflow(textblock, width):
    """Clips overflowing text of TextBlock2D with respect to width.

    Parameters
    ----------
    textblock : TextBlock2D
        The textblock object whose text needs to be clipped.
    width : int
        Required width of the clipped text.

    Returns
    -------
    clipped text : str
        Clipped version of the text.
    """
    original_str = textblock.message
    start_ptr = 0
    end_ptr = len(original_str)
    prev_bg = textblock.have_bg
    textblock.have_bg = False

    if textblock.size[0] == width or textblock.size[0] <= width:
        textblock.have_bg = prev_bg
        return original_str

    while start_ptr < end_ptr:
        mid_ptr = (start_ptr + end_ptr)//2
        textblock.message = original_str[:mid_ptr] + "..."
        if textblock.size[0] < width:
            start_ptr = mid_ptr
        elif textblock.size[0] > width:
            end_ptr = mid_ptr

        if mid_ptr == (start_ptr + end_ptr)//2 or\
           textblock.size[0] == width:
            textblock.have_bg = prev_bg
            return textblock.message


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

    def __init__(self, position=(0, 0)):
        """Init scene.

        Parameters
        ----------
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of this
            UI component.

        """
        self._scene = object()
        self._position = np.array([0, 0])
        self._callbacks = []

        self._setup()  # Setup needed actors and sub UI components.
        self.position = position

        self.left_button_state = "released"
        self.right_button_state = "released"
        self.middle_button_state = "released"

        self.on_left_mouse_button_pressed = lambda i_ren, obj, element: None
        self.on_left_mouse_button_dragged = lambda i_ren, obj, element: None
        self.on_left_mouse_button_released = lambda i_ren, obj, element: None
        self.on_left_mouse_button_clicked = lambda i_ren, obj, element: None
        self.on_left_mouse_double_clicked = lambda i_ren, obj, element: None
        self.on_right_mouse_button_pressed = lambda i_ren, obj, element: None
        self.on_right_mouse_button_released = lambda i_ren, obj, element: None
        self.on_right_mouse_button_clicked = lambda i_ren, obj, element: None
        self.on_right_mouse_double_clicked = lambda i_ren, obj, element: None
        self.on_right_mouse_button_dragged = lambda i_ren, obj, element: None
        self.on_middle_mouse_button_pressed = lambda i_ren, obj, element: None
        self.on_middle_mouse_button_released = lambda i_ren, obj, element: None
        self.on_middle_mouse_button_clicked = lambda i_ren, obj, element: None
        self.on_middle_mouse_double_clicked = lambda i_ren, obj, element: None
        self.on_middle_mouse_button_dragged = lambda i_ren, obj, element: None
        self.on_key_press = lambda i_ren, obj, element: None

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
        """Actors composing this UI component."""
        return self._get_actors()

    @abc.abstractmethod
    def _add_to_scene(self, _scene):
        """Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        _scene : scene

        """
        msg = "Subclasses of UI must implement `_add_to_scene(self, scene)`."
        raise NotImplementedError(msg)

    def add_to_scene(self, scene):
        """Allow UI objects to add their own props to the scene.

        Parameters
        ----------
        scene : scene

        """
        self._add_to_scene(scene)

        # Get a hold on the current interactor style.
        iren = scene.GetRenderWindow().GetInteractor().GetInteractorStyle()

        for callback in self._callbacks:
            if not isinstance(iren, CustomInteractorStyle):
                msg = ("The ShowManager requires `CustomInteractorStyle` in"
                       " order to use callbacks.")
                raise TypeError(msg)

            if callback[0] == self._scene:

                iren.add_callback(iren, callback[1], callback[2], args=[self])
            else:
                iren.add_callback(*callback, args=[self])

    def add_callback(self, prop, event_type, callback, priority=0):
        """Add a callback to a specific event for this UI component.

        Parameters
        ----------
        prop : vtkProp
            The prop on which is callback is to be added.
        event_type : string
            The event code.
        callback : function
            The callback function.
        priority : int
            Higher number is higher priority.

        """
        # Actually since we need an interactor style we will add the callback
        # only when this UI component is added to the scene.
        self._callbacks.append((prop, event_type, callback, priority))

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, coords):
        coords = np.asarray(coords)
        self._set_position(coords)
        self._position = coords

    @abc.abstractmethod
    def _set_position(self, _coords):
        """Position the lower-left corner of this UI component.

        Parameters
        ----------
        _coords: (float, float)
            Absolute pixel coordinates (x, y).

        """
        msg = "Subclasses of UI must implement `_set_position(self, coords)`."
        raise NotImplementedError(msg)

    @property
    def size(self):
        return np.asarray(self._get_size(), dtype=int)

    @abc.abstractmethod
    def _get_size(self):
        msg = "Subclasses of UI must implement property `size`."
        raise NotImplementedError(msg)

    @property
    def center(self):
        return self.position + self.size / 2.

    @center.setter
    def center(self, coords):
        """Position the center of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).

        """
        if not hasattr(self, "size"):
            msg = "Subclasses of UI must implement the `size` property."
            raise NotImplementedError(msg)

        new_center = np.array(coords)
        size = np.array(self.size)
        new_lower_left_corner = new_center - size / 2.
        self.position = new_lower_left_corner

    def set_visibility(self, visibility):
        """Set visibility of this UI component."""
        for actor in self.actors:
            actor.SetVisibility(visibility)

    def handle_events(self, actor):
        self.add_callback(actor, "LeftButtonPressEvent",
                          self.left_button_click_callback)
        self.add_callback(actor, "LeftButtonReleaseEvent",
                          self.left_button_release_callback)
        self.add_callback(actor, "RightButtonPressEvent",
                          self.right_button_click_callback)
        self.add_callback(actor, "RightButtonReleaseEvent",
                          self.right_button_release_callback)
        self.add_callback(actor, "MiddleButtonPressEvent",
                          self.middle_button_click_callback)
        self.add_callback(actor, "MiddleButtonReleaseEvent",
                          self.middle_button_release_callback)
        self.add_callback(actor, "MouseMoveEvent", self.mouse_move_callback)
        self.add_callback(actor, "KeyPressEvent", self.key_press_callback)

    @staticmethod
    def left_button_click_callback(i_ren, obj, self):
        self.left_button_state = "pressing"
        self.on_left_mouse_button_pressed(i_ren, obj, self)
        i_ren.event.abort()

    @staticmethod
    def left_button_release_callback(i_ren, obj, self):
        if self.left_button_state == "pressing":
            self.on_left_mouse_button_clicked(i_ren, obj, self)
        self.left_button_state = "released"
        self.on_left_mouse_button_released(i_ren, obj, self)

    @staticmethod
    def right_button_click_callback(i_ren, obj, self):
        self.right_button_state = "pressing"
        self.on_right_mouse_button_pressed(i_ren, obj, self)
        i_ren.event.abort()

    @staticmethod
    def right_button_release_callback(i_ren, obj, self):
        if self.right_button_state == "pressing":
            self.on_right_mouse_button_clicked(i_ren, obj, self)
        self.right_button_state = "released"
        self.on_right_mouse_button_released(i_ren, obj, self)

    @staticmethod
    def middle_button_click_callback(i_ren, obj, self):
        self.middle_button_state = "pressing"
        self.on_middle_mouse_button_pressed(i_ren, obj, self)
        i_ren.event.abort()

    @staticmethod
    def middle_button_release_callback(i_ren, obj, self):
        if self.middle_button_state == "pressing":
            self.on_middle_mouse_button_clicked(i_ren, obj, self)
        self.middle_button_state = "released"
        self.on_middle_mouse_button_released(i_ren, obj, self)

    @staticmethod
    def mouse_move_callback(i_ren, obj, self):
        left_pressing_or_dragging = (self.left_button_state == "pressing" or
                                     self.left_button_state == "dragging")

        right_pressing_or_dragging = (self.right_button_state == "pressing" or
                                      self.right_button_state == "dragging")

        middle_pressing_or_dragging = \
            (self.middle_button_state == "pressing" or
             self.middle_button_state == "dragging")

        if left_pressing_or_dragging:
            self.left_button_state = "dragging"
            self.on_left_mouse_button_dragged(i_ren, obj, self)
        elif right_pressing_or_dragging:
            self.right_button_state = "dragging"
            self.on_right_mouse_button_dragged(i_ren, obj, self)
        elif middle_pressing_or_dragging:
            self.middle_button_state = "dragging"
            self.on_middle_mouse_button_dragged(i_ren, obj, self)

    @staticmethod
    def key_press_callback(i_ren, obj, self):
        self.on_key_press(i_ren, obj, self)


class Button2D(UI):
    """A 2D overlay button and is of type vtkTexturedActor2D.

    Currently supports::

        - Multiple icons.
        - Switching between icons.

    """

    def __init__(self, icon_fnames, position=(0, 0), size=(30, 30)):
        """Init class instance.

        Parameters
        ----------
        icon_fnames : List(string, string)
            ((iconname, filename), (iconname, filename), ....)
        position : (float, float), optional
            Absolute coordinates (x, y) of the lower-left corner of the button.
        size : (int, int), optional
            Width and height in pixels of the button.

        """
        super(Button2D, self).__init__(position)

        self.icon_extents = dict()
        self.icons = self._build_icons(icon_fnames)
        self.icon_names = [icon[0] for icon in self.icons]
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.set_icon(self.icons[self.current_icon_id][1])
        self.resize(size)

    def _get_size(self):
        lower_left_corner = self.texture_points.GetPoint(0)
        upper_right_corner = self.texture_points.GetPoint(2)
        size = np.array(upper_right_corner) - np.array(lower_left_corner)
        return abs(size[:2])

    def _build_icons(self, icon_fnames):
        """Convert file names to vtkImageDataGeometryFilters.

        A pre-processing step to prevent re-read of file names during every
        state change.

        Parameters
        ----------
        icon_fnames : List(string, string)
            ((iconname, filename), (iconname, filename), ....)

        Returns
        -------
        icons : List
            A list of corresponding vtkImageDataGeometryFilters.

        """
        icons = []
        for icon_name, icon_fname in icon_fnames:
            icons.append((icon_name, load_image(icon_fname, as_vtktype=True)))

        return icons

    def _setup(self):
        """Set up this UI component.

        Creating the button actor used internally.

        """
        # This is highly inspired by
        # https://github.com/Kitware/VTK/blob/c3ec2495b183e3327820e927af7f8f90d34c3474/Interaction/Widgets/vtkBalloonRepresentation.cxx#L47

        self.texture_polydata = vtk.vtkPolyData()
        self.texture_points = vtk.vtkPoints()
        self.texture_points.SetNumberOfPoints(4)

        polys = vtk.vtkCellArray()
        polys.InsertNextCell(4)
        polys.InsertCellPoint(0)
        polys.InsertCellPoint(1)
        polys.InsertCellPoint(2)
        polys.InsertCellPoint(3)
        self.texture_polydata.SetPolys(polys)

        tc = vtk.vtkFloatArray()
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

        texture_mapper = vtk.vtkPolyDataMapper2D()
        texture_mapper = set_input(texture_mapper, self.texture_polydata)

        button = vtk.vtkTexturedActor2D()
        button.SetMapper(texture_mapper)

        self.texture = vtk.vtkTexture()
        button.SetTexture(self.texture)

        button_property = vtk.vtkProperty2D()
        button_property.SetOpacity(1.0)
        button.SetProperty(button_property)
        self.actor = button

        # Add default events listener to the VTK actor.
        self.handle_events(self.actor)

    def _get_actors(self):
        """Get the actors composing this UI component."""
        return [self.actor]

    def _add_to_scene(self, scene):
        """Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene

        """
        scene.add(self.actor)

    def resize(self, size):
        """Resize the button.

        Parameters
        ----------
        size : (float, float)
            Button size (width, height) in pixels.

        """
        # Update actor.
        self.texture_points.SetPoint(0, 0, 0, 0.0)
        self.texture_points.SetPoint(1, size[0], 0, 0.0)
        self.texture_points.SetPoint(2, size[0], size[1], 0.0)
        self.texture_points.SetPoint(3, 0, size[1], 0.0)
        self.texture_polydata.SetPoints(self.texture_points)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.actor.SetPosition(*coords)

    @property
    def color(self):
        """ Gets the button's color.
        """
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """ Sets the button's color.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].
        """
        self.actor.GetProperty().SetColor(*color)

    def scale(self, factor):
        """ Scales the button.

        Parameters
        ----------
        factor : (float, float)
            Scaling factor (width, height) in pixels.
        """
        self.resize(self.size * factor)

    def set_icon_by_name(self, icon_name):
        """ Set the button icon using its name.

        Parameters
        ----------
        icon_name : str
        """
        icon_id = self.icon_names.index(icon_name)
        self.set_icon(self.icons[icon_id][1])

    def set_icon(self, icon):
        """ Modifies the icon used by the vtkTexturedActor2D.

        Parameters
        ----------
        icon : imageDataGeometryFilter
        """
        self.texture = set_input(self.texture, icon)

    def next_icon_id(self):
        """ Sets the next icon ID while cycling through icons.
        """
        self.current_icon_id += 1
        if self.current_icon_id == len(self.icons):
            self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]

    def next_icon(self):
        """ Increments the state of the Button.

            Also changes the icon.
        """
        self.next_icon_id()
        self.set_icon(self.icons[self.current_icon_id][1])


class Rectangle2D(UI):
    """ A 2D rectangle sub-classed from UI.
    """

    def __init__(self, size=(0, 0), position=(0, 0), color=(1, 1, 1),
                 opacity=1.0):
        """ Initializes a rectangle.

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
        super(Rectangle2D, self).__init__(position)
        self.color = color
        self.opacity = opacity
        self.resize(size)

    def _setup(self):
        """ Setup this UI component.

        Creating the polygon actor used internally.
        """
        # Setup four points
        size = (1, 1)
        self._points = vtk.vtkPoints()
        self._points.InsertNextPoint(0, 0, 0)
        self._points.InsertNextPoint(size[0], 0, 0)
        self._points.InsertNextPoint(size[0], size[1], 0)
        self._points.InsertNextPoint(0, size[1], 0)

        # Create the polygon
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
        polygon.GetPointIds().SetId(0, 0)
        polygon.GetPointIds().SetId(1, 1)
        polygon.GetPointIds().SetId(2, 2)
        polygon.GetPointIds().SetId(3, 3)

        # Add the polygon to a list of polygons
        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(polygon)

        # Create a PolyData
        self._polygonPolyData = vtk.vtkPolyData()
        self._polygonPolyData.SetPoints(self._points)
        self._polygonPolyData.SetPolys(polygons)

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper2D()
        mapper = set_input(mapper, self._polygonPolyData)

        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(mapper)

        # Add default events listener to the VTK actor.
        self.handle_events(self.actor)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return [self.actor]

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        scene.add(self.actor)

    def _get_size(self):
        # Get 2D coordinates of two opposed corners of the rectangle.
        lower_left_corner = np.array(self._points.GetPoint(0)[:2])
        upper_right_corner = np.array(self._points.GetPoint(2)[:2])
        size = abs(upper_right_corner - lower_left_corner)
        return size

    @property
    def width(self):
        return self._points.GetPoint(2)[0]

    @width.setter
    def width(self, width):
        self.resize((width, self.height))

    @property
    def height(self):
        return self._points.GetPoint(2)[1]

    @height.setter
    def height(self, height):
        self.resize((self.width, height))

    def resize(self, size):
        """ Sets the button size.

        Parameters
        ----------
        size : (float, float)
            Button size (width, height) in pixels.
        """
        self._points.SetPoint(0, 0, 0, 0.0)
        self._points.SetPoint(1, size[0], 0, 0.0)
        self._points.SetPoint(2, size[0], size[1], 0.0)
        self._points.SetPoint(3, 0, size[1], 0.0)
        self._polygonPolyData.SetPoints(self._points)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper = set_input(mapper, self._polygonPolyData)

        self.actor.SetMapper(mapper)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.actor.SetPosition(*coords)

    @property
    def color(self):
        """ Gets the rectangle's color.
        """
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """ Sets the rectangle's color.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].
        """
        self.actor.GetProperty().SetColor(*color)

    @property
    def opacity(self):
        """ Gets the rectangle's opacity.
        """
        return self.actor.GetProperty().GetOpacity()

    @opacity.setter
    def opacity(self, opacity):
        """ Sets the rectangle's opacity.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].
        """
        self.actor.GetProperty().SetOpacity(opacity)


class Disk2D(UI):
    """ A 2D disk UI component.
    """

    def __init__(self, outer_radius, inner_radius=0, center=(0, 0),
                 color=(1, 1, 1), opacity=1.0):
        """ Initializes a 2D Disk.

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
        super(Disk2D, self).__init__()
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.color = color
        self.opacity = opacity
        self.center = center

    def _setup(self):
        """ Setup this UI component.

        Creating the disk actor used internally.
        """
        # Setting up disk actor.
        self._disk = vtk.vtkDiskSource()
        self._disk.SetRadialResolution(10)
        self._disk.SetCircumferentialResolution(50)
        self._disk.Update()

        # Mapper
        mapper = vtk.vtkPolyDataMapper2D()
        mapper = set_input(mapper, self._disk.GetOutputPort())

        # Actor
        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(mapper)

        # Add default events listener to the VTK actor.
        self.handle_events(self.actor)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return [self.actor]

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        scene.add(self.actor)

    def _get_size(self):
        diameter = 2 * self.outer_radius
        size = (diameter, diameter)
        return size

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component's bounding box.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        # Disk actor are positioned with respect to their center.
        self.actor.SetPosition(*coords + self.outer_radius)

    @property
    def color(self):
        """ Gets the color of this UI component.
        """
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """ Sets the color of this UI component.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].
        """
        self.actor.GetProperty().SetColor(*color)

    @property
    def opacity(self):
        """ Gets the opacity of this UI component.
        """
        return self.actor.GetProperty().GetOpacity()

    @opacity.setter
    def opacity(self, opacity):
        """ Sets the opacity of this UI component.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].
        """
        self.actor.GetProperty().SetOpacity(opacity)

    @property
    def inner_radius(self):
        return self._disk.GetInnerRadius()

    @inner_radius.setter
    def inner_radius(self, radius):
        self._disk.SetInnerRadius(radius)
        self._disk.Update()

    @property
    def outer_radius(self):
        return self._disk.GetOuterRadius()

    @outer_radius.setter
    def outer_radius(self, radius):
        self._disk.SetOuterRadius(radius)
        self._disk.Update()


class Panel2D(UI):
    """ A 2D UI Panel.

    Can contain one or more UI elements.

    Attributes
    ----------
    alignment : [left, right]
        Alignment of the panel with respect to the overall screen.
    """

    def __init__(self, size, position=(0, 0), color=(0.1, 0.1, 0.1),
                 opacity=0.7, align="left"):
        """
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
        """
        super(Panel2D, self).__init__(position)
        self.resize(size)
        self.alignment = align
        self.color = color
        self.opacity = opacity
        self.position = position
        self._drag_offset = None

    def _setup(self):
        """ Setup this UI component.

        Create the background (Rectangle2D) of the panel.
        """
        self._elements = []
        self.element_offsets = []
        self.background = Rectangle2D()
        self.add_element(self.background, (0, 0))

        # Add default events listener for this UI component.
        self.background.on_left_mouse_button_pressed = self.left_button_pressed
        self.background.on_left_mouse_button_dragged = self.left_button_dragged

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        actors = []
        for element in self._elements:
            actors += element.actors

        return actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        for element in self._elements:
            element.add_to_scene(scene)

    def _get_size(self):
        return self.background.size

    def resize(self, size):
        """ Sets the panel size.

        Parameters
        ----------
        size : (float, float)
            Panel size (width, height) in pixels.
        """
        self.background.resize(size)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        coords = np.array(coords)
        for element, offset in self.element_offsets:
            element.position = coords + offset

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

    def add_element(self, element, coords, anchor="position"):
        """ Adds a UI component to the panel.

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
                raise ValueError("Normalized coordinates must be in [0,1].")

            coords = coords * self.size

        if anchor == "center":
            element.center = self.position + coords
        elif anchor == "position":
            element.position = self.position + coords
        else:
            msg = ("Unknown anchor {}. Supported anchors are 'position'"
                   " and 'center'.")
            raise ValueError(msg)

        self._elements.append(element)
        offset = element.position - self.position
        self.element_offsets.append((element, offset))

    def remove_element(self, element):
        """ Removes a UI component from the panel.

        Parameters
        ----------
        element : UI
            The UI item to be removed.
        """
        idx = self._elements.index(element)
        del self._elements[idx]
        del self.element_offsets[idx]

    def update_element(self, element, coords, anchor="position"):
        """ Updates the position of a UI component in the panel.

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
        self._drag_offset = click_pos - panel2d_object.position
        i_ren.event.abort()  # Stop propagating the event.

    def left_button_dragged(self, i_ren, _obj, _panel2d_object):
        if self._drag_offset is not None:
            click_position = np.array(i_ren.event.position)
            new_position = click_position - self._drag_offset
            self.position = new_position
        i_ren.force_render()

    def re_align(self, window_size_change):
        """ Re-organises the elements in case the window size is changed.

        Parameters
        ----------
        window_size_change : (int, int)
            New window size (width, height) in pixels.
        """
        if self.alignment == "left":
            pass
        elif self.alignment == "right":
            self.position += np.array(window_size_change)
        else:
            msg = "You can only left-align or right-align objects in a panel."
            raise ValueError(msg)


class TextBlock2D(UI):
    """ Wraps over the default vtkTextActor and helps setting the text.

    Contains member functions for text formatting.

    Attributes
    ----------
    actor : :class:`vtkTextActor`
        The text actor.
    message : str
        The initial text while building the actor.
    position : (float, float)
        (x, y) in pixels.
    color : (float, float, float)
        RGB: Values must be between 0-1.
    bg_color : (float, float, float)
        RGB: Values must be between 0-1.
    font_size : int
        Size of the text font.
    font_family : str
        Currently only supports Arial.
    justification : str
        left, right or center.
    vertical_justification : str
        bottom, middle or top.
    bold : bool
        Makes text bold.
    italic : bool
        Makes text italicised.
    shadow : bool
        Adds text shadow.
    size : (int, int)
        Size (width, height) in pixels of the text bounding box.
    """

    def __init__(self, text="Text Block", font_size=18, font_family='Arial',
                 justification='left', vertical_justification="bottom",
                 bold=False, italic=False, shadow=False, size=None,
                 color=(1, 1, 1), bg_color=None, position=(0, 0)):
        """
        Parameters
        ----------
        text : str
            The initial text while building the actor.
        position : (float, float)
            (x, y) in pixels.
        color : (float, float, float)
            RGB: Values must be between 0-1.
        bg_color : (float, float, float)
            RGB: Values must be between 0-1.
        font_size : int
            Size of the text font.
        font_family : str
            Currently only supports Arial.
        justification : str
            left, right or center.
        vertical_justification : str
            bottom, middle or top.
        bold : bool
            Makes text bold.
        italic : bool
            Makes text italicised.
        shadow : bool
            Adds text shadow.
        size : (int, int)
            Size (width, height) in pixels of the text bounding box.
        """
        super(TextBlock2D, self).__init__(position=position)
        self.scene = None
        self.have_bg = bool(bg_color)
        if size is not None:
            self.resize(size)
        else:
            self.font_size = font_size
        self.color = color
        self.background_color = bg_color
        self.font_family = font_family
        self.justification = justification
        self.bold = bold
        self.italic = italic
        self.shadow = shadow
        self.vertical_justification = vertical_justification
        self.message = text

    def _setup(self):
        self.actor = vtk.vtkTextActor()
        self.actor.GetPosition2Coordinate().SetCoordinateSystemToViewport()
        self.background = Rectangle2D()
        self.handle_events(self.actor)

    def resize(self, size):
        """Resize TextBlock2D.

        Parameters
        ----------
        size : (int, int)
            Text bounding box size(width, height) in pixels.
        """
        if self.have_bg:
            self.background.resize(size)
        self.actor.SetTextScaleModeToProp()
        self.actor.SetPosition2(*size)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return [self.actor] + self.background.actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.scene = scene
        if self.have_bg and not self.actor.GetTextScaleMode():
            size = np.zeros(2)
            self.actor.GetSize(scene, size)
            self.background.resize(size)
        scene.add(self.background, self.actor)

    @property
    def message(self):
        """ Gets message from the text.

        Returns
        -------
        str
            The current text message.
        """
        return self.actor.GetInput()

    @message.setter
    def message(self, text):
        """ Sets the text message.

        Parameters
        ----------
        text : str
            The message to be set.
        """
        self.actor.SetInput(text)

    @property
    def font_size(self):
        """ Gets text font size.

        Returns
        ----------
        int
            Text font size.
        """
        return self.actor.GetTextProperty().GetFontSize()

    @font_size.setter
    def font_size(self, size):
        """ Sets font size.

        Parameters
        ----------
        size : int
            Text font size.
        """
        self.actor.SetTextScaleModeToNone()
        self.actor.GetTextProperty().SetFontSize(size)

        if self.scene is not None and self.have_bg:
            bb_size = np.zeros(2)
            self.actor.GetSize(self.scene, bb_size)
            bg_size = self.background.size
            if bb_size[0] > bg_size[0] or bb_size[1] > bg_size[1]:
                warn("Font size exceeds background bounding box."
                     " Font Size will not be updated.", RuntimeWarning)
                self.actor.SetTextScaleModeToProp()
                self.actor.SetPosition2(*bg_size)

    @property
    def font_family(self):
        """ Gets font family.

        Returns
        ----------
        str
            Text font family.
        """
        return self.actor.GetTextProperty().GetFontFamilyAsString()

    @font_family.setter
    def font_family(self, family='Arial'):
        """ Sets font family.

        Currently Arial and Courier are supported.

        Parameters
        ----------
        family : str
            The font family.
        """
        if family == 'Arial':
            self.actor.GetTextProperty().SetFontFamilyToArial()
        elif family == 'Courier':
            self.actor.GetTextProperty().SetFontFamilyToCourier()
        else:
            raise ValueError("Font not supported yet: {}.".format(family))

    @property
    def justification(self):
        """ Gets text justification.

        Returns
        -------
        str
            Text justification.
        """
        justification = self.actor.GetTextProperty().GetJustificationAsString()
        if justification == 'Left':
            return "left"
        elif justification == 'Centered':
            return "center"
        elif justification == 'Right':
            return "right"

    @justification.setter
    def justification(self, justification):
        """ Justifies text.

        Parameters
        ----------
        justification : str
            Possible values are left, right, center.
        """
        text_property = self.actor.GetTextProperty()
        if justification == 'left':
            text_property.SetJustificationToLeft()
        elif justification == 'center':
            text_property.SetJustificationToCentered()
        elif justification == 'right':
            text_property.SetJustificationToRight()
        else:
            msg = "Text can only be justified left, right and center."
            raise ValueError(msg)

    @property
    def vertical_justification(self):
        """ Gets text vertical justification.

        Returns
        -------
        str
            Text vertical justification.
        """
        text_property = self.actor.GetTextProperty()
        vjustification = text_property.GetVerticalJustificationAsString()
        if vjustification == 'Bottom':
            return "bottom"
        elif vjustification == 'Centered':
            return "middle"
        elif vjustification == 'Top':
            return "top"

    @vertical_justification.setter
    def vertical_justification(self, vertical_justification):
        """ Justifies text vertically.

        Parameters
        ----------
        vertical_justification : str
            Possible values are bottom, middle, top.
        """
        text_property = self.actor.GetTextProperty()
        if vertical_justification == 'bottom':
            text_property.SetVerticalJustificationToBottom()
        elif vertical_justification == 'middle':
            text_property.SetVerticalJustificationToCentered()
        elif vertical_justification == 'top':
            text_property.SetVerticalJustificationToTop()
        else:
            msg = "Vertical justification must be: bottom, middle or top."
            raise ValueError(msg)

    @property
    def bold(self):
        """ Returns whether the text is bold.

        Returns
        -------
        bool
            Text is bold if True.
        """
        return self.actor.GetTextProperty().GetBold()

    @bold.setter
    def bold(self, flag):
        """ Bolds/un-bolds text.

        Parameters
        ----------
        flag : bool
            Sets text bold if True.
        """
        self.actor.GetTextProperty().SetBold(flag)

    @property
    def italic(self):
        """ Returns whether the text is italicised.

        Returns
        -------
        bool
            Text is italicised if True.
        """
        return self.actor.GetTextProperty().GetItalic()

    @italic.setter
    def italic(self, flag):
        """ Italicises/un-italicises text.

        Parameters
        ----------
        flag : bool
            Italicises text if True.
        """
        self.actor.GetTextProperty().SetItalic(flag)

    @property
    def shadow(self):
        """ Returns whether the text has shadow.

        Returns
        -------
        bool
            Text is shadowed if True.
        """
        return self.actor.GetTextProperty().GetShadow()

    @shadow.setter
    def shadow(self, flag):
        """ Adds/removes text shadow.

        Parameters
        ----------
        flag : bool
            Shadows text if True.
        """
        self.actor.GetTextProperty().SetShadow(flag)

    @property
    def color(self):
        """ Gets text color.

        Returns
        -------
        (float, float, float)
            Returns text color in RGB.
        """
        return self.actor.GetTextProperty().GetColor()

    @color.setter
    def color(self, color=(1, 0, 0)):
        """ Set text color.

        Parameters
        ----------
        color : (float, float, float)
            RGB: Values must be between 0-1.
        """
        self.actor.GetTextProperty().SetColor(*color)

    @property
    def background_color(self):
        """ Gets background color.

        Returns
        -------
        (float, float, float) or None
            If None, there no background color.
            Otherwise, background color in RGB.
        """
        if not self.have_bg:
            return None

        return self.background.color

    @background_color.setter
    def background_color(self, color):
        """ Set text color.

        Parameters
        ----------
        color : (float, float, float) or None
            If None, remove background.
            Otherwise, RGB values (must be between 0-1).
        """
        if color is None:
            # Remove background.
            self.have_bg = False
            self.background.set_visibility(False)

        else:
            self.have_bg = True
            self.background.set_visibility(True)
            self.background.color = color

    def _set_position(self, position):
        """ Set text actor position.

        Parameters
        ----------
        position : (float, float)
            The new position. (x, y) in pixels.
        """
        self.actor.SetPosition(*position)
        self.background.position = position

    def _get_size(self):
        if self.have_bg:
            return self.background.size

        if not self.actor.GetTextScaleMode():
            if self.scene is not None:
                size = np.zeros(2)
                self.actor.GetSize(self.scene, size)
                return size
            else:
                warn("TextBlock2D must be added to the scene before "
                     "querying its size while TextScaleMode is set to None.",
                     RuntimeWarning)

        return self.actor.GetPosition2()


class TextBox2D(UI):
    """ An editable 2D text box that behaves as a UI component.

    Currently supports:
    - Basic text editing.
    - Cursor movements.
    - Single and multi-line text boxes.
    - Pre text formatting (text needs to be formatted beforehand).

    Attributes
    ----------
    text : str
        The current text state.
    actor : :class:`vtkActor2d`
        The text actor.
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
    def __init__(self, width, height, text="Enter Text", position=(100, 10),
                 color=(0, 0, 0), font_size=18, font_family='Arial',
                 justification='left', bold=False,
                 italic=False, shadow=False):
        """
        Parameters
        ----------
        width : int
            The number of characters in a single line of text.
        height : int
            The number of lines in the textbox.
        text : str
            The initial text while building the actor.
        position : (float, float)
            (x, y) in pixels.
        color : (float, float, float)
            RGB: Values must be between 0-1.
        font_size : int
            Size of the text font.
        font_family : str
            Currently only supports Arial.
        justification : str
            left, right or center.
        bold : bool
            Makes text bold.
        italic : bool
            Makes text italicised.
        shadow : bool
            Adds text shadow.
        """
        super(TextBox2D, self).__init__(position=position)

        self.message = text
        self.text.message = text
        self.text.font_size = font_size
        self.text.font_family = font_family
        self.text.justification = justification
        self.text.bold = bold
        self.text.italic = italic
        self.text.shadow = shadow
        self.text.color = color
        self.text.background_color = (1, 1, 1)

        self.width = width
        self.height = height
        self.window_left = 0
        self.window_right = 0
        self.caret_pos = 0
        self.init = True

    def _setup(self):
        """ Setup this UI component.

        Create the TextBlock2D component used for the textbox.
        """
        self.text = TextBlock2D()

        # Add default events listener for this UI component.
        self.text.on_left_mouse_button_pressed = self.left_button_press
        self.text.on_key_press = self.key_press

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.text.actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.text.add_to_scene(scene)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.text.position = coords

    def _get_size(self):
        return self.text.size

    def set_message(self, message):
        """ Set custom text to textbox.

        Parameters
        ----------
        message: str
            The custom message to be set.
        """
        self.message = message
        self.text.message = message
        self.init = False
        self.window_right = len(self.message)
        self.window_left = 0
        self.caret_pos = self.window_right

    def width_set_text(self, text):
        """ Adds newlines to text where necessary.

        This is needed for multi-line text boxes.

        Parameters
        ----------
        text : str
            The final text to be formatted.

        Returns
        -------
        str
            A multi line formatted text.
        """
        multi_line_text = ""
        for i, t in enumerate(text):
            multi_line_text += t
            if (i + 1) % self.width == 0:
                multi_line_text += "\n"
        return multi_line_text.rstrip("\n")

    def handle_character(self, character):
        """ Main driving function that handles button events.

        # TODO: Need to handle all kinds of characters like !, +, etc.

        Parameters
        ----------
        character : str
        """
        if character.lower() == "return":
            self.render_text(False)
            return True
        if character.lower() == "backspace":
            self.remove_character()
        elif character.lower() == "left":
            self.move_left()
        elif character.lower() == "right":
            self.move_right()
        else:
            self.add_character(character)
        self.render_text()
        return False

    def move_caret_right(self):
        """ Moves the caret towards right.
        """
        self.caret_pos = min(self.caret_pos + 1, len(self.message))

    def move_caret_left(self):
        """ Moves the caret towards left.
        """
        self.caret_pos = max(self.caret_pos - 1, 0)

    def right_move_right(self):
        """ Moves right boundary of the text window right-wards.
        """
        if self.window_right <= len(self.message):
            self.window_right += 1

    def right_move_left(self):
        """ Moves right boundary of the text window left-wards.
        """
        if self.window_right > 0:
            self.window_right -= 1

    def left_move_right(self):
        """ Moves left boundary of the text window right-wards.
        """
        if self.window_left <= len(self.message):
            self.window_left += 1

    def left_move_left(self):
        """ Moves left boundary of the text window left-wards.
        """
        if self.window_left > 0:
            self.window_left -= 1

    def add_character(self, character):
        """ Inserts a character into the text and moves window and caret.

        Parameters
        ----------
        character : str
        """
        if len(character) > 1 and character.lower() != "space":
            return
        if character.lower() == "space":
            character = " "
        self.message = (self.message[:self.caret_pos] +
                        character +
                        self.message[self.caret_pos:])
        self.move_caret_right()
        if (self.window_right -
                self.window_left == self.height * self.width - 1):
            self.left_move_right()
        self.right_move_right()

    def remove_character(self):
        """ Removes a character and moves window and caret accordingly.
        """
        if self.caret_pos == 0:
            return
        self.message = (self.message[:self.caret_pos - 1] +
                        self.message[self.caret_pos:])
        self.move_caret_left()
        if len(self.message) < self.height * self.width - 1:
            self.right_move_left()
        if (self.window_right -
                self.window_left == self.height * self.width - 1):
            if self.window_left > 0:
                self.left_move_left()
                self.right_move_left()

    def move_left(self):
        """ Handles left button press.
        """
        self.move_caret_left()
        if self.caret_pos == self.window_left - 1:
            if (self.window_right -
                    self.window_left == self.height * self.width - 1):
                self.left_move_left()
                self.right_move_left()

    def move_right(self):
        """ Handles right button press.
        """
        self.move_caret_right()
        if self.caret_pos == self.window_right + 1:
            if (self.window_right -
                    self.window_left == self.height * self.width - 1):
                self.left_move_right()
                self.right_move_right()

    def showable_text(self, show_caret):
        """ Chops out text to be shown on the screen.

        Parameters
        ----------
        show_caret : bool
            Whether or not to show the caret.
        """
        if show_caret:
            ret_text = (self.message[:self.caret_pos] +
                        "_" +
                        self.message[self.caret_pos:])
        else:
            ret_text = self.message
        ret_text = ret_text[self.window_left:self.window_right + 1]
        return ret_text

    def render_text(self, show_caret=True):
        """ Renders text after processing.

        Parameters
        ----------
        show_caret : bool
            Whether or not to show the caret.
        """
        text = self.showable_text(show_caret)
        if text == "":
            text = "Enter Text"
        self.text.message = self.width_set_text(text)

    def edit_mode(self):
        """ Turns on edit mode.
        """
        if self.init:
            self.message = ""
            self.init = False
            self.caret_pos = 0
        self.render_text()

    def left_button_press(self, i_ren, _obj, _textbox_object):
        """ Left button press handler for textbox

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        _textbox_object: :class:`TextBox2D`
        """
        i_ren.add_active_prop(self.text.actor)
        self.edit_mode()
        i_ren.force_render()

    def key_press(self, i_ren, _obj, _textbox_object):
        """ Key press handler for textbox

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        _textbox_object: :class:`TextBox2D`
        """
        key = i_ren.event.key
        is_done = self.handle_character(key)
        if is_done:
            i_ren.remove_active_prop(self.text.actor)

        i_ren.force_render()


class LineSlider2D(UI):
    """ A 2D Line Slider.

    A sliding handle on a line with a percentage indicator.

    Attributes
    ----------
    line_width : int
        Width of the line on which the disk will slide.
    length : int
        Length of the slider.
    track : :class:`Rectangle2D`
        The line on which the slider's handle moves.
    handle : :class:`Disk2D`
        The moving part of the slider.
    text : :class:`TextBlock2D`
        The text that shows percentage.
    shape : string
        Describes the shape of the handle.
        Currently supports 'disk' and 'square'.
    default_color : (float, float, float)
        Color of the handle when in unpressed state.
    active_color : (float, float, float)
        Color of the handle when it is pressed.
    """
    def __init__(self, center=(0, 0),
                 initial_value=50, min_value=0, max_value=100,
                 length=200, line_width=5,
                 inner_radius=0, outer_radius=10, handle_side=20,
                 font_size=16, orientation="horizontal", text_alignment='',
                 text_template="{value:.1f} ({ratio:.0%})", shape="disk"):
        """
        Parameters
        ----------
        center : (float, float)
            Center of the slider's center.
        initial_value : float
            Initial value of the slider.
        min_value : float
            Minimum value of the slider.
        max_value : float
            Maximum value of the slider.
        length : int
            Length of the slider.
        line_width : int
            Width of the line on which the disk will slide.
        inner_radius : int
            Inner radius of the handles (if disk).
        outer_radius : int
            Outer radius of the handles (if disk).
        handle_side : int
            Side length of the handles (if sqaure).
        font_size : int
            Size of the text to display alongside the slider (pt).
        orientation : str
            horizontal or vertical
        text_alignment : str
            define text alignment on a slider. Left (default)/ right for the
            vertical slider or top/bottom (default) for an horizontal slider.
        text_template : str, callable
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`.
            If callable, this instance of `:class:LineSlider2D` will be
            passed as argument to the text template function.
        shape : string
            Describes the shape of the handle.
            Currently supports 'disk' and 'square'.
        """
        self.shape = shape
        self.orientation = orientation.lower().strip()
        self.align_dict = {'horizontal': ['top', 'bottom'],
                           'vertical': ['left', 'right']}
        self.default_color = (1, 1, 1)
        self.active_color = (0, 0, 1)
        self.alignment = text_alignment.lower()
        super(LineSlider2D, self).__init__()

        if self.orientation == "horizontal":
            self.alignment = 'bottom' if not self.alignment else self.alignment
            self.track.width = length
            self.track.height = line_width
        elif self.orientation == "vertical":
            self.alignment = 'left' if not self.alignment else self.alignment
            self.track.width = line_width
            self.track.height = length
        else:
            raise ValueError("Unknown orientation")

        if self.alignment not in self.align_dict[self.orientation]:
            raise ValueError("Unknown alignment: choose from '{}' or '{}'".
                             format(*self.align_dict[self.orientation]))

        if shape == "disk":
            self.handle.inner_radius = inner_radius
            self.handle.outer_radius = outer_radius
        elif shape == "square":
            self.handle.width = handle_side
            self.handle.height = handle_side
        self.center = center

        self.min_value = min_value
        self.max_value = max_value
        self.text.font_size = font_size
        self.text_template = text_template

        # Offer some standard hooks to the user.
        self.on_change = lambda ui: None

        self.value = initial_value
        self.update()

    def _setup(self):
        """ Setup this UI component.

        Create the slider's track (Rectangle2D), the handle (Disk2D) and
        the text (TextBlock2D).
        """
        # Slider's track
        self.track = Rectangle2D()
        self.track.color = (1, 0, 0)

        # Slider's handle
        if self.shape == "disk":
            self.handle = Disk2D(outer_radius=1)
        elif self.shape == "square":
            self.handle = Rectangle2D(size=(1, 1))
        self.handle.color = self.default_color

        # Slider Text
        self.text = TextBlock2D(justification="center",
                                vertical_justification="top")

        # Add default events listener for this UI component.
        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.track.on_left_mouse_button_released = \
            self.handle_release_callback
        self.handle.on_left_mouse_button_dragged = self.handle_move_callback
        self.handle.on_left_mouse_button_released = \
            self.handle_release_callback

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.track.actors + self.handle.actors + self.text.actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.track.add_to_scene(scene)
        self.handle.add_to_scene(scene)
        self.text.add_to_scene(scene)

    def _get_size(self):
        # Consider the handle's size when computing the slider's size.
        width = None
        height = None
        if self.orientation == "horizontal":
            width = self.track.width + self.handle.size[0]
            height = max(self.track.height, self.handle.size[1])
        else:
            width = max(self.track.width, self.handle.size[0])
            height = self.track.height + self.handle.size[1]

        return np.array([width, height])

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        # Offset the slider line by the handle's radius.
        track_position = coords + self.handle.size / 2.
        if self.orientation == "horizontal":
            # Offset the slider line height by half the slider line width.
            track_position[1] -= self.track.size[1] / 2.
        else:
            # Offset the slider line width by half the slider line height.
            track_position[0] += self.track.size[0] / 2.

        self.track.position = track_position
        self.handle.position = self.handle.position.astype('float64')
        self.handle.position += coords - self.position
        # Position the text below the handle.
        if self.orientation == "horizontal":
            align = 35 if self.alignment == 'top' else -10
            self.text.position = (self.handle.center[0],
                                  self.handle.position[1] + align)
        else:
            align = 70 if self.alignment == 'right' else -35
            self.text.position = (self.handle.position[0] + align,
                                  self.handle.center[1] + 2)

    @property
    def bottom_y_position(self):
        return self.track.position[1]

    @property
    def top_y_position(self):
        return self.track.position[1] + self.track.size[1]

    @property
    def left_x_position(self):
        return self.track.position[0]

    @property
    def right_x_position(self):
        return self.track.position[0] + self.track.size[0]

    def set_position(self, position):
        """ Sets the disk's position.

        Parameters
        ----------
        position : (float, float)
            The absolute position of the disk (x, y).
        """

        # Move slider disk.
        if self.orientation == "horizontal":
            x_position = position[0]
            x_position = max(x_position, self.left_x_position)
            x_position = min(x_position, self.right_x_position)
            self.handle.center = (x_position, self.track.center[1])
        else:
            y_position = position[1]
            y_position = max(y_position, self.bottom_y_position)
            y_position = min(y_position, self.top_y_position)
            self.handle.center = (self.track.center[0], y_position)
        self.update()  # Update information.

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        value_range = self.max_value - self.min_value
        self.ratio = (value - self.min_value) / value_range

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        position_x = self.left_x_position + ratio * self.track.width
        position_y = self.bottom_y_position + ratio * self.track.height
        self.set_position((position_x, position_y))

    def format_text(self):
        """ Returns formatted text to display along the slider. """
        if callable(self.text_template):
            return self.text_template(self)
        return self.text_template.format(ratio=self.ratio, value=self.value)

    def update(self):
        """ Updates the slider. """
        # Compute the ratio determined by the position of the slider disk.
        disk_position_x = None
        disk_position_y = None

        if self.orientation == "horizontal":
            length = float(self.right_x_position - self.left_x_position)
            if length != self.track.width:
                raise ValueError("Disk position outside the slider line")
            disk_position_x = self.handle.center[0]
            self._ratio = (disk_position_x - self.left_x_position) / length
        else:
            length = float(self.top_y_position - self.bottom_y_position)
            if length != self.track.height:
                raise ValueError("Disk position outside the slider line")
            disk_position_y = self.handle.center[1]
            self._ratio = (disk_position_y - self.bottom_y_position) / length

        # Compute the selected value considering min_value and max_value.
        value_range = self.max_value - self.min_value
        self._value = self.min_value + self.ratio * value_range

        # Update text.
        text = self.format_text()
        self.text.message = text

        # Move the text below the slider's handle.
        if self.orientation == "horizontal":
            self.text.position = (disk_position_x, self.text.position[1])
        else:
            self.text.position = (self.text.position[0], disk_position_y)

        self.on_change(self)

    def track_click_callback(self, i_ren, _vtkactor, _slider):
        """ Update disk position and grab the focus.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        _slider : :class:`LineSlider2D`
        """

        position = i_ren.event.position
        self.set_position(position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_move_callback(self, i_ren, _vtkactor, _slider):
        """ Actual handle movement.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        slider : :class:`LineSlider2D`
        """

        self.handle.color = self.active_color
        position = i_ren.event.position
        self.set_position(position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_release_callback(self, i_ren, _vtkactor, _slider):
        """ Change color when handle is released.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        slider : :class:`LineSlider2D`
        """
        self.handle.color = self.default_color
        i_ren.force_render()


class LineDoubleSlider2D(UI):
    """ A 2D Line Slider with two sliding rings.
    Useful for setting min and max values for something.

    Currently supports:
    - Setting positions of both disks.

    Attributes
    ----------
    line_width : int
        Width of the line on which the disk will slide.
    length : int
        Length of the slider.
    track : :class:`vtkActor`
        The line on which the handles move.
    handles : [:class:`vtkActor`, :class:`vtkActor`]
        The moving slider disks.
    text : [:class:`TextBlock2D`, :class:`TextBlock2D`]
        The texts that show the values of the disks.
    shape : string
        Describes the shape of the handle.
        Currently supports 'disk' and 'square'.
    default_color : (float, float, float)
        Color of the handles when in unpressed state.
    active_color : (float, float, float)
        Color of the handles when they are pressed.

    """
    def __init__(self, line_width=5, inner_radius=0, outer_radius=10,
                 handle_side=20, center=(450, 300), length=200,
                 initial_values=(0, 100), min_value=0, max_value=100,
                 font_size=16, text_template="{value:.1f}",
                 orientation="horizontal", shape="disk"):
        """
        Parameters
        ----------
        line_width : int
            Width of the line on which the disk will slide.
        inner_radius : int
            Inner radius of the handles (if disk).
        outer_radius : int
            Outer radius of the handles (if disk).
        handle_side : int
            Side length of the handles (if sqaure).
        center : (float, float)
            Center of the slider.
        length : int
            Length of the slider.
        initial_values : (float, float)
            Initial values of the two handles.
        min_value : float
            Minimum value of the slider.
        max_value : float
            Maximum value of the slider.
        font_size : int
            Size of the text to display alongside the slider (pt).
        text_template : str, callable
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`.
            If callable, this instance of `:class:LineDoubleSlider2D` will be
            passed as argument to the text template function.
        orientation : str
            horizontal or vertical
        shape : string
            Describes the shape of the handle.
            Currently supports 'disk' and 'square'.

        """
        self.shape = shape
        self.default_color = (1, 1, 1)
        self.active_color = (0, 0, 1)
        self.orientation = orientation.lower()
        super(LineDoubleSlider2D, self).__init__()

        if self.orientation == "horizontal":
            self.track.width = length
            self.track.height = line_width
        elif self.orientation == "vertical":
            self.track.width = line_width
            self.track.height = length
        else:
            raise ValueError("Unknown orientation")

        self.center = center
        if shape == "disk":
            self.handles[0].inner_radius = inner_radius
            self.handles[0].outer_radius = outer_radius
            self.handles[1].inner_radius = inner_radius
            self.handles[1].outer_radius = outer_radius
        elif shape == "square":
            self.handles[0].width = handle_side
            self.handles[0].height = handle_side
            self.handles[1].width = handle_side
            self.handles[1].height = handle_side

        self.min_value = min_value
        self.max_value = max_value
        self.text[0].font_size = font_size
        self.text[1].font_size = font_size
        self.text_template = text_template

        # Setting the handle positions will also update everything.
        self._values = [initial_values[0], initial_values[1]]
        self._ratio = [None, None]
        self.left_disk_value = initial_values[0]
        self.right_disk_value = initial_values[1]
        self.bottom_disk_value = initial_values[0]
        self.top_disk_value = initial_values[1]

    def _setup(self):
        """ Setup this UI component.

        Create the slider's track (Rectangle2D), the handles (Disk2D) and
        the text (TextBlock2D).
        """
        # Slider's track
        self.track = Rectangle2D()
        self.track.color = (1, 0, 0)

        # Handles
        self.handles = []
        if self.shape == "disk":
            self.handles.append(Disk2D(outer_radius=1))
            self.handles.append(Disk2D(outer_radius=1))
        elif self.shape == "square":
            self.handles.append(Rectangle2D(size=(1, 1)))
            self.handles.append(Rectangle2D(size=(1, 1)))
        self.handles[0].color = self.default_color
        self.handles[1].color = self.default_color

        # Slider Text
        self.text = [TextBlock2D(justification="center",
                                 vertical_justification="top"),
                     TextBlock2D(justification="center",
                                 vertical_justification="top")
                     ]

        # Add default events listener for this UI component.
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.handles[0].on_left_mouse_button_dragged = \
            self.handle_move_callback
        self.handles[1].on_left_mouse_button_dragged = \
            self.handle_move_callback
        self.handles[0].on_left_mouse_button_released = \
            self.handle_release_callback
        self.handles[1].on_left_mouse_button_released = \
            self.handle_release_callback

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return (self.track.actors + self.handles[0].actors +
                self.handles[1].actors + self.text[0].actors +
                self.text[1].actors)

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.track.add_to_scene(scene)
        self.handles[0].add_to_scene(scene)
        self.handles[1].add_to_scene(scene)
        self.text[0].add_to_scene(scene)
        self.text[1].add_to_scene(scene)

    def _get_size(self):
        # Consider the handle's size when computing the slider's size.
        width = None
        height = None
        if self.orientation == "horizontal":
            width = self.track.width + 2 * self.handles[0].size[0]
            height = max(self.track.height, self.handles[0].size[1])
        else:
            width = max(self.track.width, self.handles[0].size[0])
            height = self.track.height + 2 * self.handles[0].size[1]

        return np.array([width, height])

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        # Offset the slider line by the handle's radius.
        track_position = coords + self.handles[0].size / 2.
        if self.orientation == "horizontal":
            # Offset the slider line height by half the slider line width.
            track_position[1] -= self.track.size[1] / 2.
        else:
            # Offset the slider line width by half the slider line height.
            track_position[0] -= self.track.size[0] / 2.
        self.track.position = track_position

        self.handles[0].position = self.handles[0].position.astype('float64')
        self.handles[1].position = self.handles[1].position.astype('float64')

        self.handles[0].position += coords - self.position
        self.handles[1].position += coords - self.position

        if self.orientation == "horizontal":
            # Position the text below the handles.
            self.text[0].position = (self.handles[0].center[0],
                                     self.handles[0].position[1] - 20)
            self.text[1].position = (self.handles[1].center[0],
                                     self.handles[1].position[1] - 20)
        else:
            # Position the text to the left of the handles.
            self.text[0].position = (self.handles[0].center[0] - 35,
                                     self.handles[0].position[1])
            self.text[1].position = (self.handles[1].center[0] - 35,
                                     self.handles[1].position[1])

    @property
    def bottom_y_position(self):
        return self.track.position[1]

    @property
    def top_y_position(self):
        return self.track.position[1] + self.track.size[1]

    @property
    def left_x_position(self):
        return self.track.position[0]

    @property
    def right_x_position(self):
        return self.track.position[0] + self.track.size[0]

    def value_to_ratio(self, value):
        """ Converts the value of a disk to the ratio

        Parameters
        ----------
        value : float
        """
        value_range = self.max_value - self.min_value
        return (value - self.min_value) / value_range

    def ratio_to_coord(self, ratio):
        """ Converts the ratio to the absolute coordinate.

        Parameters
        ----------
        ratio : float
        """
        if self.orientation == "horizontal":
            return self.left_x_position + ratio * self.track.width
        return self.bottom_y_position + ratio * self.track.height

    def coord_to_ratio(self, coord):
        """ Converts the x coordinate of a disk to the ratio

        Parameters
        ----------
        coord : float
        """
        if self.orientation == "horizontal":
            return (coord - self.left_x_position) / float(self.track.width)
        return (coord - self.bottom_y_position) / float(self.track.height)

    def ratio_to_value(self, ratio):
        """ Converts the ratio to the value of the disk.

        Parameters
        ----------
        ratio : float
        """
        value_range = self.max_value - self.min_value
        return self.min_value + ratio * value_range

    def set_position(self, position, disk_number):
        """ Sets the disk's position.

        Parameters
        ----------
        position : (float, float)
            The absolute position of the disk (x, y).
        disk_number : int
            The index of disk being moved.
        """

        if self.orientation == "horizontal":
            x_position = position[0]

            if disk_number == 0 and x_position >= self.handles[1].center[0]:
                x_position = self.ratio_to_coord(
                    self.value_to_ratio(self._values[1] - 1))

            if disk_number == 1 and x_position <= self.handles[0].center[0]:
                x_position = self.ratio_to_coord(
                    self.value_to_ratio(self._values[0] + 1))

            x_position = max(x_position, self.left_x_position)
            x_position = min(x_position, self.right_x_position)

            self.handles[disk_number].center = \
                (x_position, self.track.center[1])
        else:
            y_position = position[1]

            if disk_number == 0 and y_position >= self.handles[1].center[1]:
                y_position = self.ratio_to_coord(
                    self.value_to_ratio(self._values[1] - 1))

            if disk_number == 1 and y_position <= self.handles[0].center[1]:
                y_position = self.ratio_to_coord(
                    self.value_to_ratio(self._values[0] + 1))

            y_position = max(y_position, self.bottom_y_position)
            y_position = min(y_position, self.top_y_position)

            self.handles[disk_number].center = \
                (self.track.center[0], y_position)
        self.update(disk_number)

    @property
    def bottom_disk_value(self):
        """ Returns the value of the bottom disk. """
        return self._values[0]

    @bottom_disk_value.setter
    def bottom_disk_value(self, bottom_disk_value):
        """ Sets the value of the bottom disk.

        Parameters
        ----------
        bottom_disk_value : float
            New value for the bottom disk.
        """
        self.bottom_disk_ratio = self.value_to_ratio(bottom_disk_value)

    @property
    def top_disk_value(self):
        """ Returns the value of the top disk. """
        return self._values[1]

    @top_disk_value.setter
    def top_disk_value(self, top_disk_value):
        """ Sets the value of the top disk.

        Parameters
        ----------
        top_disk_value : float
            New value for the top disk.
        """
        self.top_disk_ratio = self.value_to_ratio(top_disk_value)

    @property
    def left_disk_value(self):
        """ Returns the value of the left disk. """
        return self._values[0]

    @left_disk_value.setter
    def left_disk_value(self, left_disk_value):
        """ Sets the value of the left disk.

        Parameters
        ----------
        left_disk_value : New value for the left disk.
        """
        self.left_disk_ratio = self.value_to_ratio(left_disk_value)

    @property
    def right_disk_value(self):
        """ Returns the value of the right disk. """
        return self._values[1]

    @right_disk_value.setter
    def right_disk_value(self, right_disk_value):
        """ Sets the value of the right disk.

        Parameters
        ----------
        right_disk_value : New value for the right disk.
        """
        self.right_disk_ratio = self.value_to_ratio(right_disk_value)

    @property
    def bottom_disk_ratio(self):
        """ Returns the ratio of the bottom disk. """
        return self._ratio[0]

    @bottom_disk_ratio.setter
    def bottom_disk_ratio(self, bottom_disk_ratio):
        """ Sets the ratio of the bottom disk.

        Parameters
        ----------
        bottom_disk_ratio : float
            New ratio for the bottom disk.
        """
        position_x = self.ratio_to_coord(bottom_disk_ratio)
        position_y = self.ratio_to_coord(bottom_disk_ratio)
        self.set_position((position_x, position_y), 0)

    @property
    def top_disk_ratio(self):
        """ Returns the ratio of the top disk. """
        return self._ratio[1]

    @top_disk_ratio.setter
    def top_disk_ratio(self, top_disk_ratio):
        """ Sets the ratio of the top disk.

        Parameters
        ----------
        top_disk_ratio : float
            New ratio for the top disk.
        """
        position_x = self.ratio_to_coord(top_disk_ratio)
        position_y = self.ratio_to_coord(top_disk_ratio)
        self.set_position((position_x, position_y), 1)

    @property
    def left_disk_ratio(self):
        """ Returns the ratio of the left disk. """
        return self._ratio[0]

    @left_disk_ratio.setter
    def left_disk_ratio(self, left_disk_ratio):
        """ Sets the ratio of the left disk.

        Parameters
        ----------
        left_disk_ratio : New ratio for the left disk.
        """
        position_x = self.ratio_to_coord(left_disk_ratio)
        position_y = self.ratio_to_coord(left_disk_ratio)
        self.set_position((position_x, position_y), 0)

    @property
    def right_disk_ratio(self):
        """ Returns the ratio of the right disk. """
        return self._ratio[1]

    @right_disk_ratio.setter
    def right_disk_ratio(self, right_disk_ratio):
        """ Sets the ratio of the right disk.

        Parameters
        ----------
        right_disk_ratio : New ratio for the right disk.
        """
        position_x = self.ratio_to_coord(right_disk_ratio)
        position_y = self.ratio_to_coord(right_disk_ratio)
        self.set_position((position_x, position_y), 1)

    def format_text(self, disk_number):
        """ Returns formatted text to display along the slider.

        Parameters
        ----------
        disk_number : Index of the disk.
        """
        if callable(self.text_template):
            return self.text_template(self)

        return self.text_template.format(value=self._values[disk_number])

    def on_change(self, slider):
        pass

    def update(self, disk_number):
        """ Updates the slider.

        Parameters
        ----------
        disk_number : Index of the disk to be updated.
        """

        # Compute the ratio determined by the position of the slider disk.
        if self.orientation == "horizontal":
            self._ratio[disk_number] = self.coord_to_ratio(
                self.handles[disk_number].center[0])
        else:
            self._ratio[disk_number] = self.coord_to_ratio(
                self.handles[disk_number].center[1])

        # Compute the selected value considering min_value and max_value.
        self._values[disk_number] = self.ratio_to_value(
            self._ratio[disk_number])

        # Update text.
        text = self.format_text(disk_number)
        self.text[disk_number].message = text

        if self.orientation == "horizontal":
            self.text[disk_number].position = (
                self.handles[disk_number].center[0],
                self.text[disk_number].position[1])
        else:
            self.text[disk_number].position = (
                self.text[disk_number].position[0],
                self.handles[disk_number].center[1])
        self.on_change(self)

    def handle_move_callback(self, i_ren, vtkactor, _slider):
        """ Actual handle movement.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        _slider : :class:`LineDoubleSlider2D`
        """

        position = i_ren.event.position
        if vtkactor == self.handles[0].actors[0]:
            self.set_position(position, 0)
            self.handles[0].color = self.active_color
        elif vtkactor == self.handles[1].actors[0]:
            self.set_position(position, 1)
            self.handles[1].color = self.active_color
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_release_callback(self, i_ren, vtkactor, _slider):
        """ Change color when handle is released.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        _slider : :class:`LineDoubleSlider2D`
        """
        if vtkactor == self.handles[0].actors[0]:
            self.handles[0].color = self.default_color
        elif vtkactor == self.handles[1].actors[0]:
            self.handles[1].color = self.default_color
        i_ren.force_render()


class RingSlider2D(UI):
    """ A disk slider.

    A disk moves along the boundary of a ring.
    Goes from 0-360 degrees.

    Attributes
    ----------
    mid_track_radius: float
        Distance from the center of the slider to the middle of the track.
    previous_value: float
        Value of Rotation of the actor before the current value.
    track : :class:`Disk2D`
        The circle on which the slider's handle moves.
    handle : :class:`Disk2D`
        The moving part of the slider.
    text : :class:`TextBlock2D`
        The text that shows percentage.
    default_color : (float, float, float)
        Color of the handle when in unpressed state.
    active_color : (float, float, float)
        Color of the handle when it is pressed.
    """
    def __init__(self, center=(0, 0),
                 initial_value=180, min_value=0, max_value=360,
                 slider_inner_radius=40, slider_outer_radius=44,
                 handle_inner_radius=0, handle_outer_radius=10,
                 font_size=16,
                 text_template="{ratio:.0%}"):
        """
        Parameters
        ----------
        center : (float, float)
            Position (x, y) of the slider's center.
        initial_value : float
            Initial value of the slider.
        min_value : float
            Minimum value of the slider.
        max_value : float
            Maximum value of the slider.
        slider_inner_radius : int
            Inner radius of the base disk.
        slider_outer_radius : int
            Outer radius of the base disk.
        handle_outer_radius : int
            Outer radius of the slider's handle.
        handle_inner_radius : int
            Inner radius of the slider's handle.
        font_size : int
            Size of the text to display alongside the slider (pt).
        text_template : str, callable
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`, `{angle:}`.
            If callable, this instance of `:class:RingSlider2D` will be
            passed as argument to the text template function.
        """
        self.default_color = (1, 1, 1)
        self.active_color = (0, 0, 1)
        super(RingSlider2D, self).__init__()

        self.track.inner_radius = slider_inner_radius
        self.track.outer_radius = slider_outer_radius
        self.handle.inner_radius = handle_inner_radius
        self.handle.outer_radius = handle_outer_radius
        self.center = center

        self.min_value = min_value
        self.max_value = max_value
        self.text.font_size = font_size
        self.text_template = text_template

        # Offer some standard hooks to the user.
        self.on_change = lambda ui: None

        self._value = initial_value
        self.value = initial_value
        self._previous_value = initial_value
        self._angle = 0
        self._ratio = self.angle / TWO_PI

    def _setup(self):
        """ Setup this UI component.

        Create the slider's circle (Disk2D), the handle (Disk2D) and
        the text (TextBlock2D).
        """
        # Slider's track.
        self.track = Disk2D(outer_radius=1)
        self.track.color = (1, 0, 0)

        # Slider's handle.
        self.handle = Disk2D(outer_radius=1)
        self.handle.color = self.default_color

        # Slider Text
        self.text = TextBlock2D(justification="center",
                                vertical_justification="middle")

        # Add default events listener for this UI component.
        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.track.on_left_mouse_button_released = \
            self.handle_release_callback
        self.handle.on_left_mouse_button_dragged = self.handle_move_callback
        self.handle.on_left_mouse_button_released = \
            self.handle_release_callback

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.track.actors + self.handle.actors + self.text.actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.track.add_to_scene(scene)
        self.handle.add_to_scene(scene)
        self.text.add_to_scene(scene)

    def _get_size(self):
        return self.track.size + self.handle.size

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.track.position = coords + self.handle.size / 2.
        self.handle.position += coords - self.position
        # Position the text in the center of the slider's track.
        self.text.position = coords + self.size / 2.

    @property
    def mid_track_radius(self):
        return (self.track.inner_radius + self.track.outer_radius) / 2.

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        value_range = self.max_value - self.min_value
        self.ratio = (value - self.min_value) / value_range

    @property
    def previous_value(self):
        return self._previous_value

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        self.angle = ratio * TWO_PI

    @property
    def angle(self):
        """ Angle (in rad) the handle makes with x-axis """
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = angle % TWO_PI  # Wraparound
        self.update()

    def format_text(self):
        """ Returns formatted text to display along the slider. """
        if callable(self.text_template):
            return self.text_template(self)

        return self.text_template.format(ratio=self.ratio, value=self.value,
                                         angle=np.rad2deg(self.angle))

    def update(self):
        """ Updates the slider. """

        # Compute the ratio determined by the position of the slider disk.
        self._ratio = self.angle / TWO_PI

        # Compute the selected value considering min_value and max_value.
        value_range = self.max_value - self.min_value
        self._previous_value = self.value
        self._value = self.min_value + self.ratio * value_range

        # Update text disk actor.
        x = self.mid_track_radius * np.cos(self.angle) + self.center[0]
        y = self.mid_track_radius * np.sin(self.angle) + self.center[1]
        self.handle.center = (x, y)

        # Update text.
        text = self.format_text()
        self.text.message = text

        self.on_change(self)  # Call hook.

    def move_handle(self, click_position):
        """Moves the slider's handle.

        Parameters
        ----------
        click_position: (float, float)
            Position of the mouse click.
        """
        x, y = np.array(click_position) - self.center
        angle = np.arctan2(y, x)
        if angle < 0:
            angle += TWO_PI

        self.angle = angle

    def track_click_callback(self, i_ren, _obj, _slider):
        """ Update disk position and grab the focus.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        _slider : :class:`RingSlider2D`
        """
        click_position = i_ren.event.position
        self.move_handle(click_position=click_position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_move_callback(self, i_ren, _obj, _slider):
        """ Move the slider's handle.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        _slider : :class:`RingSlider2D`
        """
        click_position = i_ren.event.position
        self.handle.color = self.active_color
        self.move_handle(click_position=click_position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_release_callback(self, i_ren, _obj, _slider):
        """ Change color when handle is released.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        _slider : :class:`RingSlider2D`
        """
        self.handle.color = self.default_color
        i_ren.force_render()


class RangeSlider(UI):

    """ A set of a LineSlider2D and a LineDoubleSlider2D.
    The double slider is used to set the min and max value
    for the LineSlider2D

    Attributes
    ----------
    range_slider_center : (float, float)
        Center of the LineDoubleSlider2D object.
    value_slider_center : (float, float)
        Center of the LineSlider2D object.
    range_slider : :class:`LineDoubleSlider2D`
        The line slider which sets the min and max values
    value_slider : :class:`LineSlider2D`
        The line slider which sets the value

    """
    def __init__(self, line_width=5, inner_radius=0, outer_radius=10,
                 handle_side=20, range_slider_center=(450, 400),
                 value_slider_center=(450, 300), length=200, min_value=0,
                 max_value=100, font_size=16, range_precision=1,
                 orientation="horizontal", value_precision=2, shape="disk"):
        """
        Parameters
        ----------
        line_width : int
            Width of the slider tracks
        inner_radius : int
            Inner radius of the handles.
        outer_radius : int
            Outer radius of the handles.
        handle_side : int
            Side length of the handles (if square).
        range_slider_center : (float, float)
            Center of the LineDoubleSlider2D object.
        value_slider_center : (float, float)
            Center of the LineSlider2D object.
        length : int
            Length of the sliders.
        min_value : float
            Minimum value of the double slider.
        max_value : float
            Maximum value of the double slider.
        font_size : int
            Size of the text to display alongside the sliders (pt).
        range_precision : int
            Number of decimal places to show the min and max values set.
        orientation : str
            horizontal or vertical
        value_precision : int
            Number of decimal places to show the value set on slider.
        shape : string
            Describes the shape of the handle.
            Currently supports 'disk' and 'square'.
        """
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

        self.range_slider_text_template = \
            "{value:." + str(range_precision) + "f}"
        self.value_slider_text_template = \
            "{value:." + str(value_precision) + "f}"

        self.range_slider_center = range_slider_center
        self.value_slider_center = value_slider_center
        super(RangeSlider, self).__init__()

    def _setup(self):
        """ Setup this UI component.
        """
        self.range_slider = \
            LineDoubleSlider2D(line_width=self.line_width,
                               inner_radius=self.inner_radius,
                               outer_radius=self.outer_radius,
                               handle_side=self.handle_side,
                               center=self.range_slider_center,
                               length=self.length, min_value=self.min_value,
                               max_value=self.max_value,
                               initial_values=(self.min_value,
                                               self.max_value),
                               font_size=self.font_size, shape=self.shape,
                               orientation=self.orientation,
                               text_template=self.range_slider_text_template)

        self.value_slider = \
            LineSlider2D(line_width=self.line_width, length=self.length,
                         inner_radius=self.inner_radius,
                         outer_radius=self.outer_radius,
                         handle_side=self.handle_side,
                         center=self.value_slider_center,
                         min_value=self.min_value, max_value=self.max_value,
                         initial_value=(self.min_value + self.max_value) / 2,
                         font_size=self.font_size, shape=self.shape,
                         orientation=self.orientation,
                         text_template=self.value_slider_text_template)

        # Add default events listener for this UI component.
        self.range_slider.handles[0].on_left_mouse_button_dragged = \
            self.range_slider_handle_move_callback
        self.range_slider.handles[1].on_left_mouse_button_dragged = \
            self.range_slider_handle_move_callback

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.range_slider.actors + self.value_slider.actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.range_slider.add_to_scene(scene)
        self.value_slider.add_to_scene(scene)

    def _get_size(self):
        return self.range_slider.size + self.value_slider.size

    def _set_position(self, coords):
        pass

    def range_slider_handle_move_callback(self, i_ren, obj, _slider):
        """ Actual movement of range_slider's handles.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        _slider : :class:`RangeSlider`

        """
        position = i_ren.event.position
        if obj == self.range_slider.handles[0].actors[0]:
            self.range_slider.handles[0].color = \
                self.range_slider.active_color
            self.range_slider.set_position(position, 0)
            self.value_slider.min_value = self.range_slider.left_disk_value
            self.value_slider.update()
        elif obj == self.range_slider.handles[1].actors[0]:
            self.range_slider.handles[1].color = \
                self.range_slider.active_color
            self.range_slider.set_position(position, 1)
            self.value_slider.max_value = self.range_slider.right_disk_value
            self.value_slider.update()
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.


class ImageContainer2D(UI):
    """ A 2D container to hold an image.
    Currently Supports:
    - png and jpg/jpeg images

    Attributes
    ----------
    size: (float, float)
        Image size (width, height) in pixels.
    img : vtkImageDataGeometryFilters
        The image loaded from the specified path.

    """

    def __init__(self, img_path, position=(0, 0), size=(100, 100)):
        """
        Parameters
        ----------
        img_path : string
            Path of the image
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
        """ Setup this UI Component.
        Return an image as a 2D actor with a specific position.

        Returns
        -------
        :class:`vtkTexturedActor2D`
        """
        self.texture_polydata = vtk.vtkPolyData()
        self.texture_points = vtk.vtkPoints()
        self.texture_points.SetNumberOfPoints(4)

        polys = vtk.vtkCellArray()
        polys.InsertNextCell(4)
        polys.InsertCellPoint(0)
        polys.InsertCellPoint(1)
        polys.InsertCellPoint(2)
        polys.InsertCellPoint(3)
        self.texture_polydata.SetPolys(polys)

        tc = vtk.vtkFloatArray()
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

        texture_mapper = vtk.vtkPolyDataMapper2D()
        texture_mapper = set_input(texture_mapper, self.texture_polydata)

        image = vtk.vtkTexturedActor2D()
        image.SetMapper(texture_mapper)

        self.texture = vtk.vtkTexture()
        image.SetTexture(self.texture)

        image_property = vtk.vtkProperty2D()
        image_property.SetOpacity(1.0)
        image.SetProperty(image_property)
        self.actor = image

        # Add default events listener to the VTK actor.
        self.handle_events(self.actor)

    def _get_actors(self):
        """ Returns the actors that compose this UI component.
        """
        return [self.actor]

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        scene.add(self.actor)

    def resize(self, size):
        """ Resize the image.

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
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.actor.SetPosition(*coords)

    def scale(self, factor):
        """ Scales the image.

        Parameters
        ----------
        factor : (float, float)
            Scaling factor (width, height) in pixels.
        """
        self.resize(self.size * factor)

    def set_img(self, img):
        """ Modifies the image used by the vtkTexturedActor2D.

        Parameters
        ----------
        img : imageDataGeometryFilter

        """
        self.texture = set_input(self.texture, img)


class Option(UI):

    """
    A set of a Button2D and a TextBlock2D to act as a single option
    for checkboxes and radio buttons.
    Clicking the button toggles its checked/unchecked status.

    Attributes
    ----------
    label : str
        The label for the option.
    font_size : int
            Font Size of the label.
    """

    def __init__(self, label, position=(0, 0), font_size=18, checked=False):
        """
        Parameters
        ----------
        label : str
            Text to be displayed next to the option's button.
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of
            the button of the option.
        font_size : int
            Font size of the label.
        checked : bool, optional
            Boolean value indicates the initial state of the option
        """
        self.label = label
        self.font_size = font_size
        self.checked = checked
        self.button_size = (font_size * 1.2, font_size * 1.2)
        self.button_label_gap = 10
        super(Option, self).__init__(position)

        # Offer some standard hooks to the user.
        self.on_change = lambda obj: None

    def _setup(self):
        """ Setup this UI component.
        """
        # Option's button
        self.button_icons = []
        self.button_icons.append(('unchecked',
                                 read_viz_icons(fname="stop2.png")))
        self.button_icons.append(('checked',
                                 read_viz_icons(fname="checkmark.png")))
        self.button = Button2D(icon_fnames=self.button_icons,
                               size=self.button_size)

        self.text = TextBlock2D(text=self.label, font_size=self.font_size)

        # Display initial state
        if self.checked:
            self.button.set_icon_by_name("checked")

        # Add callbacks
        self.button.on_left_mouse_button_clicked = self.toggle
        self.text.on_left_mouse_button_clicked = self.toggle

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.button.actors + self.text.actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.button.add_to_scene(scene)
        self.text.add_to_scene(scene)

    def _get_size(self):
        width = self.button.size[0] + self.button_label_gap + self.text.size[0]
        height = max(self.button.size[1], self.text.size[1])
        return np.array([width, height])

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        num_newlines = self.label.count('\n')
        self.button.position = coords + \
            (0, num_newlines * self.font_size * 0.5)
        offset = (self.button.size[0] + self.button_label_gap, 0)
        self.text.position = coords + offset

    def toggle(self, i_ren, _obj, _element):
        if self.checked:
            self.deselect()
        else:
            self.select()

        self.on_change(self)
        i_ren.force_render()

    def select(self):
        self.checked = True
        self.button.set_icon_by_name("checked")

    def deselect(self):
        self.checked = False
        self.button.set_icon_by_name("unchecked")


class Checkbox(UI):

    """ A 2D set of :class:'Option' objects.
    Multiple options can be selected.

    Attributes
    ----------
    labels : list(string)
        List of labels of each option.
    options : dict(Option)
        Dictionary of all the options in the checkbox set.
    padding : float
        Distance between two adjacent options
    """

    def __init__(self, labels, checked_labels=(), padding=1, font_size=18,
                 font_family='Arial', position=(0, 0)):
        """
        Parameters
        ----------
        labels : list(str)
            List of labels of each option.
        checked_labels: list(str), optional
            List of labels that are checked on setting up.
        padding : float, optional
            The distance between two adjacent options
        font_size : int, optional
            Size of the text font.
        font_family : str, optional
            Currently only supports Arial.
        position : (float, float), optional
            Absolute coordinates (x, y) of the lower-left corner of
            the button of the first option.
        """

        self.labels = list(reversed(labels))
        self._padding = padding
        self._font_size = font_size
        self.font_family = font_family
        self.checked_labels = list(checked_labels)
        super(Checkbox, self).__init__(position)
        self.on_change = lambda checkbox: None

    def _setup(self):
        """ Setup this UI component.
        """
        self.options = OrderedDict()
        button_y = self.position[1]
        for label in self.labels:

            option = Option(label=label,
                            font_size=self.font_size,
                            position=(self.position[0], button_y),
                            checked=(label in self.checked_labels))

            line_spacing = option.text.actor.GetTextProperty().GetLineSpacing()
            button_y = button_y + self.font_size * \
                (label.count('\n') + 1) * (line_spacing + 0.1) + self.padding
            self.options[label] = option

            # Set callback
            option.on_change = self._handle_option_change

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        actors = []
        for option in self.options.values():
            actors = actors + option.actors
        return actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        for option in self.options.values():
            option.add_to_scene(scene)

    def _get_size(self):
        option_width, option_height = self.options.values()[0].get_size()
        height = len(self.labels) * (option_height + self.padding) \
            - self.padding
        return np.asarray([option_width, height])

    def _handle_option_change(self, option):
        """ Reacts whenever an option changes.

        Parameters
        ----------
        option : :class:`Option`
        """
        if option.checked:
            self.checked_labels.append(option.label)
        else:
            self.checked_labels.remove(option.label)

        self.on_change(self)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        button_y = coords[1]
        for option_no, option in enumerate(self.options.values()):
            option.position = (coords[0], button_y)
            line_spacing = option.text.actor.GetTextProperty().GetLineSpacing()
            button_y = (button_y + self.font_size
                        * (self.labels[option_no].count('\n') + 1)
                        * (line_spacing + 0.1) + self.padding)

    @property
    def font_size(self):
        """ Gets the font size of text.
        """
        return self._font_size

    @property
    def padding(self):
        """ Gets the padding between options.
        """
        return self._padding


class RadioButton(Checkbox):
    """ A 2D set of :class:'Option' objects.
    Only one option can be selected.

    Attributes
    ----------
    labels : list(string)
        List of labels of each option.
    options : dict(Option)
        Dictionary of all the options in the checkbox set.
    padding : float
        Distance between two adjacent options
    """

    def __init__(self, labels, checked_labels, padding=1, font_size=18,
                 font_family='Arial', position=(0, 0)):
        """
        Parameters
        ----------
        labels : list(str)
            List of labels of each option.
        checked_labels: list(str), optional
            List of labels that are checked on setting up.
        padding : float, optional
            The distance between two adjacent options
        font_size : int, optional
            Size of the text font.
        font_family : str, optional
            Currently only supports Arial.
        position : (float, float), optional
            Absolute coordinates (x, y) of the lower-left corner of
            the button of the first option.
        """
        if len(checked_labels) > 1:
            err_msg = "Only one option can be pre-selected for radio buttons."
            raise ValueError(err_msg)

        super(RadioButton, self).__init__(labels=labels, position=position,
                                          padding=padding,
                                          font_size=font_size,
                                          font_family=font_family,
                                          checked_labels=checked_labels)

    def _handle_option_change(self, option):
        for option_ in self.options.values():
            option_.deselect()

        option.select()
        self.checked_labels = [option.label]
        self.on_change(self)


class ListBox2D(UI):
    """ UI component that allows the user to select items from a list.

    Attributes
    ----------
    on_change: function
        Callback function for when the selected items have changed.
    """

    def __init__(self, values, position=(0, 0), size=(100, 300),
                 multiselection=True, reverse_scrolling=False,
                 font_size=20, line_spacing=1.4,
                 text_color=(0.2, 0.2, 0.2),
                 selected_color=(0.9, 0.6, 0.6),
                 unselected_color=(0.6, 0.6, 0.6),
                 scroll_bar_active_color=(0.6, 0.2, 0.2),
                 scroll_bar_inactive_color=(0.9, 0.0, 0.0),
                 background_opacity=1.):
        """

        Parameters
        ----------
        values: list of objects
            Values used to populate this listbox. Objects must be castable
            to string.
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of this
            UI component.
        size : (int, int)
            Width and height in pixels of this UI component.
        multiselection: {True, False}
            Whether multiple values can be selected at once.
        reverse_scrolling: {True, False}
            If True, scrolling up will move the list of files down.
        font_size: int
            The font size in pixels.
        line_spacing: float
            Distance between listbox's items in pixels.
        text_color : tuple of 3 floats
        selected_color : tuple of 3 floats
        unselected_color : tuple of 3 floats
        scroll_bar_active_color : tuple of 3 floats
        scroll_bar_inactive_color : tuple of 3 floats
        background_opacity : float
        """
        self.view_offset = 0
        self.slots = []
        self.selected = []

        self.panel_size = size
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.slot_height = int(self.font_size * self.line_spacing)

        self.text_color = text_color
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        self.background_opacity = background_opacity

        # self.panel.resize(size)
        self.values = values
        self.multiselection = multiselection
        self.last_selection_idx = 0
        self.reverse_scrolling = reverse_scrolling
        super(ListBox2D, self).__init__()

        denom = len(self.values) - self.nb_slots
        if not denom:
            denom += 1
        self.scroll_step_size = (self.slot_height * self.nb_slots -
                                 self.scroll_bar.height) / denom

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
        """ Setup this UI component.

        Create the ListBox (Panel2D) filled with empty slots (ListBoxItem2D).
        """
        self.margin = 10
        size = self.panel_size
        font_size = self.font_size
        # Calculating the number of slots.
        self.nb_slots = int((size[1] - 2 * self.margin) // self.slot_height)

        # This panel facilitates adding slots at the right position.
        self.panel = Panel2D(size=size, color=(1, 1, 1))

        # Add a scroll bar
        scroll_bar_height = self.nb_slots * (size[1] - 2 * self.margin) \
            / len(self.values)
        self.scroll_bar = Rectangle2D(size=(int(size[0]/20),
                                      scroll_bar_height))
        if len(self.values) <= self.nb_slots:
            self.scroll_bar.set_visibility(False)
        self.panel.add_element(
            self.scroll_bar, size - self.scroll_bar.size - self.margin)

        # Initialisation of empty text actors
        self.slot_width = size[0] - self.scroll_bar.size[0] - \
            2 * self.margin - self.margin
        x = self.margin
        y = size[1] - self.margin
        for _ in range(self.nb_slots):
            y -= self.slot_height
            item = ListBoxItem2D(list_box=self,
                                 size=(self.slot_width, self.slot_height),
                                 text_color=self.text_color,
                                 selected_color=self.selected_color,
                                 unselected_color=self.unselected_color,
                                 background_opacity=self.background_opacity)
            item.textblock.font_size = font_size
            self.slots.append(item)
            self.panel.add_element(item, (x, y + self.margin))

        # Add default events listener for this UI component.
        self.scroll_bar.on_left_mouse_button_pressed = \
            self.scroll_click_callback
        self.scroll_bar.on_left_mouse_button_released = \
            self.scroll_release_callback
        self.scroll_bar.on_left_mouse_button_dragged = \
            self.scroll_drag_callback

        # Handle mouse wheel events on the panel.
        up_event = "MouseWheelForwardEvent"
        down_event = "MouseWheelBackwardEvent"
        if self.reverse_scrolling:
            up_event, down_event = down_event, up_event  # Swap events

        self.add_callback(self.panel.background.actor, up_event,
                          self.up_button_callback)
        self.add_callback(self.panel.background.actor, down_event,
                          self.down_button_callback)

        # Handle mouse wheel events on the slots.
        for slot in self.slots:
            self.add_callback(slot.background.actor, up_event,
                              self.up_button_callback)
            self.add_callback(slot.background.actor, down_event,
                              self.down_button_callback)
            self.add_callback(slot.textblock.actor, up_event,
                              self.up_button_callback)
            self.add_callback(slot.textblock.actor, down_event,
                              self.down_button_callback)

    def resize(self, size):
        pass

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.panel.actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.panel.add_to_scene(scene)
        for slot in self.slots:
            clip_overflow(slot.textblock, self.slot_width)

    def _get_size(self):
        return self.panel.size

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.panel.position = coords

    def up_button_callback(self, i_ren, _obj, _list_box):
        """ Pressing up button scrolls up in the combo box.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        _list_box: :class:`ListBox2D`

        """
        if self.view_offset > 0:
            self.view_offset -= 1
            self.update()
            scroll_bar_idx = self.panel._elements.index(self.scroll_bar)
            self.scroll_bar.center = (self.scroll_bar.center[0],
                                      self.scroll_bar.center[1] +
                                      self.scroll_step_size)
            self.panel.element_offsets[scroll_bar_idx] = (
                self.scroll_bar,
                (self.scroll_bar.position - self.panel.position))

        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def down_button_callback(self, i_ren, _obj, _list_box):
        """ Pressing down button scrolls down in the combo box.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        _list_box: :class:`ListBox2D`

        """
        view_end = self.view_offset + self.nb_slots
        if view_end < len(self.values):
            self.view_offset += 1
            self.update()
            scroll_bar_idx = self.panel._elements.index(self.scroll_bar)
            self.scroll_bar.center = (self.scroll_bar.center[0],
                                      self.scroll_bar.center[1] -
                                      self.scroll_step_size)
            self.panel.element_offsets[scroll_bar_idx] = (
                self.scroll_bar,
                (self.scroll_bar.position - self.panel.position))

        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def scroll_click_callback(self, i_ren, _obj, _rect_obj):
        """ Callback to change the color of the bar when it is clicked.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        _rect_obj: :class:`Rectangle2D`

        """
        self.scroll_bar.color = self.scroll_bar_active_color
        self.scroll_init_position = i_ren.event.position[1]
        i_ren.force_render()
        i_ren.event.abort()

    def scroll_release_callback(self, i_ren, _obj, _rect_obj):
        """ Callback to change the color of the bar when it is released.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        rect_obj: :class:`Rectangle2D`

        """
        self.scroll_bar.color = self.scroll_bar_inactive_color
        i_ren.force_render()

    def scroll_drag_callback(self, i_ren, _obj, _rect_obj):
        """ Dragging scroll bar in the combo box.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        rect_obj: :class:`Rectangle2D`

        """
        position = i_ren.event.position
        offset = int((position[1] - self.scroll_init_position) /
                     self.scroll_step_size)
        if offset > 0 and self.view_offset > 0:
            offset = min(offset, self.view_offset)

        elif offset < 0 and (
                self.view_offset + self.nb_slots < len(self.values)):
            offset = min(-offset,
                         len(self.values) - self.nb_slots - self.view_offset)
            offset = - offset
        else:
            return

        self.view_offset -= offset
        self.update()
        scroll_bar_idx = self.panel._elements.index(self.scroll_bar)
        self.scroll_bar.center = (self.scroll_bar.center[0],
                                  self.scroll_bar.center[1] +
                                  offset * self.scroll_step_size)

        self.scroll_init_position += offset * self.scroll_step_size

        self.panel.element_offsets[scroll_bar_idx] = (
            self.scroll_bar, (self.scroll_bar.position - self.panel.position))
        i_ren.force_render()
        i_ren.event.abort()

    def update(self):
        """ Refresh listbox's content. """
        view_start = self.view_offset
        view_end = view_start + self.nb_slots
        values_to_show = self.values[view_start:view_end]

        # Populate slots according to the view.
        for i, choice in enumerate(values_to_show):
            slot = self.slots[i]
            slot.element = choice
            if slot.textblock.scene is not None:
                clip_overflow(slot.textblock, self.slot_width)
            slot.set_visibility(True)
            if slot.element in self.selected:
                slot.select()
            else:
                slot.deselect()

        # Flush remaining slots.
        for slot in self.slots[len(values_to_show):]:
            slot.element = None
            slot.set_visibility(False)
            slot.deselect()

    def update_scrollbar(self):
        """ Change the scroll-bar height when the values
        in the listbox change
        """
        self.scroll_bar.set_visibility(True)

        self.scroll_bar.height = self.nb_slots * \
            (self.panel_size[1] - 2 * self.margin) / len(self.values)

        self.scroll_step_size = (self.slot_height * self.nb_slots -
                                 self.scroll_bar.height) \
            / (len(self.values) - self.nb_slots)

        self.panel.update_element(
            self.scroll_bar, self.panel_size - self.scroll_bar.size -
            self.margin)

        if len(self.values) <= self.nb_slots:
            self.scroll_bar.set_visibility(False)

    def clear_selection(self):
        del self.selected[:]

    def select(self, item, multiselect=False, range_select=False):
        """ Select the item.

        Parameters
        ----------
        item: ListBoxItem2D's object
            Item to select.
        multiselect: {True, False}
            If True and multiselection is allowed, the item is added to the
            selection.
            Otherwise, the selection will only contain the provided item unless
            range_select is True.
        range_select: {True, False}
            If True and multiselection is allowed, all items between the last
            selected item and the current one will be added to the selection.
            Otherwise, the selection will only contain the provided item unless
            multi_select is True.

        """
        selection_idx = self.values.index(item.element)
        if self.multiselection and range_select:
            self.clear_selection()
            step = 1 if selection_idx >= self.last_selection_idx else -1
            for i in range(self.last_selection_idx,
                           selection_idx + step,
                           step):
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

        self.on_change()  # Call hook.
        self.update()


class ListBoxItem2D(UI):
    """ The text displayed in a listbox. """

    def __init__(self, list_box, size,
                 text_color=(1.0, 0.0, 0.0),
                 selected_color=(0.4, 0.4, 0.4),
                 unselected_color=(0.9, 0.9, 0.9),
                 background_opacity=1.):
        """ Single ListBox Item

        Parameters
        ----------
        list_box : :class:`ListBox`
            The ListBox reference this text belongs to.
        size : tuple of 2 ints
            The size of the listbox item.
        text_color : tuple of 3 floats
        unselected_color : tuple of 3 floats
        selected_color : tuple of 3 floats
        background_opacity : float
        """
        super(ListBoxItem2D, self).__init__()
        self._element = None
        self.list_box = list_box
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
        """ Setup this UI component.

        Create the ListBoxItem2D with its background (Rectangle2D) and its
        label (TextBlock2D).
        """
        self.background = Rectangle2D()
        self.textblock = TextBlock2D(justification="left",
                                     vertical_justification="middle")

        # Add default events listener for this UI component.
        self.add_callback(self.textblock.actor, "LeftButtonPressEvent",
                          self.left_button_clicked)
        self.add_callback(self.background.actor, "LeftButtonPressEvent",
                          self.left_button_clicked)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.background.actors + self.textblock.actors

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.background.add_to_scene(scene)
        self.textblock.add_to_scene(scene)

    def _get_size(self):
        return self.background.size

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.textblock.position = coords
        # Center background underneath the text.
        position = coords
        self.background.position = (position[0],
                                    position[1] - self.background.size[1] / 2.)

    def deselect(self):
        self.background.color = self.unselected_color
        self.textblock.bold = False
        self.selected = False

    def select(self):
        self.textblock.bold = True
        self.background.color = self.selected_color
        self.selected = True

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, element):
        self._element = element
        self.textblock.message = "" if self._element is None else str(element)

    def left_button_clicked(self, i_ren, _obj, _list_box_item):
        """ A callback to handle left click for this UI element.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        _list_box_item: :class:`ListBoxItem2D`

        """
        multiselect = i_ren.event.ctrl_key
        range_select = i_ren.event.shift_key
        self.list_box.select(self, multiselect, range_select)
        i_ren.force_render()


class FileMenu2D(UI):
    """ A menu to select files in the current folder.
    Can go to new folder, previous folder and select multiple files.

    Attributes
    ----------
    extensions: ['extension1', 'extension2', ....]
        To show all files, extensions=["*"] or [""]
        List of extensions to be shown as files.
    listbox : :class: 'ListBox2D'
        Container for the menu.

    """

    def __init__(self, directory_path, extensions=None, position=(0, 0),
                 size=(100, 300), multiselection=True, reverse_scrolling=False,
                 font_size=20, line_spacing=1.4):
        """

        Parameters
        ----------
        extensions: list(string)
            List of extensions to be shown as files.
        directory_path: string
            Path of the directory where this dialog should open.
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of this
            UI component.
        size : (int, int)
            Width and height in pixels of this UI component.
        multiselection: {True, False}
            Whether multiple values can be selected at once.
        reverse_scrolling: {True, False}
            If True, scrolling up will move the list of files down.
        font_size: int
            The font size in pixels.
        line_spacing: float
            Distance between listbox's items in pixels.
        """
        self.font_size = font_size
        self.multiselection = multiselection
        self.reverse_scrolling = reverse_scrolling
        self.line_spacing = line_spacing
        self.extensions = extensions or ["*"]
        self.current_directory = directory_path
        self.menu_size = size
        self.directory_contents = []

        super(FileMenu2D, self).__init__()
        self.position = position
        self.set_slot_colors()

    def _setup(self):
        """ Setup this UI component.
        Create the ListBox (Panel2D) filled with empty slots (ListBoxItem2D).
        """
        self.directory_contents = self.get_all_file_names()
        content_names = [x[0] for x in self.directory_contents]
        self.listbox = ListBox2D(
            values=content_names, multiselection=self.multiselection,
            font_size=self.font_size, line_spacing=self.line_spacing,
            reverse_scrolling=self.reverse_scrolling, size=self.menu_size)

        self.add_callback(self.listbox.scroll_bar.actor, "MouseMoveEvent",
                          self.scroll_callback)

        # Handle mouse wheel events on the panel.
        up_event = "MouseWheelForwardEvent"
        down_event = "MouseWheelBackwardEvent"
        if self.reverse_scrolling:
            up_event, down_event = down_event, up_event  # Swap events

        self.add_callback(self.listbox.panel.background.actor, up_event,
                          self.scroll_callback)
        self.add_callback(self.listbox.panel.background.actor, down_event,
                          self.scroll_callback)

        # Handle mouse wheel events on the slots.
        for slot in self.listbox.slots:
            self.add_callback(slot.background.actor, up_event,
                              self.scroll_callback)
            self.add_callback(slot.background.actor, down_event,
                              self.scroll_callback)
            self.add_callback(slot.textblock.actor, up_event,
                              self.scroll_callback)
            self.add_callback(slot.textblock.actor, down_event,
                              self.scroll_callback)
            slot.add_callback(slot.textblock.actor, "LeftButtonPressEvent",
                              self.directory_click_callback)
            slot.add_callback(slot.background.actor, "LeftButtonPressEvent",
                              self.directory_click_callback)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.listbox.actors

    def resize(self, size):
        pass

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.listbox.position = coords

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.listbox.add_to_scene(scene)

    def _get_size(self):
        return self.listbox.size

    def get_all_file_names(self):
        """ Gets file and directory names.

        Returns
        -------
        all_file_names: list((string, {"directory", "file"}))
            List of all file and directory names as string.
        """
        all_file_names = []

        directory_names = self.get_directory_names()
        for directory_name in directory_names:
            all_file_names.append((directory_name, "directory"))

        file_names = self.get_file_names()
        for file_name in file_names:
            all_file_names.append((file_name, "file"))

        return all_file_names

    def get_directory_names(self):
        """ Finds names of all directories in the current_directory

        Returns
        -------
        directory_names: list(string)
            List of all directory names as string.
        """
        # A list of directory names in the current directory
        directory_names = []
        for (_, dirnames, _) in os.walk(self.current_directory):
            directory_names += dirnames
            break
        directory_names.sort(key=lambda s: s.lower())
        directory_names.insert(0, "../")
        return directory_names

    def get_file_names(self):
        """ Finds names of all files in the current_directory

        Returns
        -------
        file_names: list(string)
            List of all file names as string.
        """
        # A list of file names with extension in the current directory
        for (_, _, files) in os.walk(self.current_directory):
            break

        file_names = []
        if "*" in self.extensions or "" in self.extensions:
            file_names = files
        else:
            for ext in self.extensions:
                for file in files:
                    if file.endswith("." + ext):
                        file_names.append(file)
        file_names.sort(key=lambda s: s.lower())
        return file_names

    def set_slot_colors(self):
        """ Sets the text color of the slots based on the type of element
        they show. Blue for directories and green for files.
        """
        for idx, slot in enumerate(self.listbox.slots):
            list_idx = min(self.listbox.view_offset + idx,
                           len(self.directory_contents)-1)
            if self.directory_contents[list_idx][1] == "directory":
                slot.textblock.color = (0, 0.6, 0)
            elif self.directory_contents[list_idx][1] == "file":
                slot.textblock.color = (0, 0, 0.7)

    def scroll_callback(self, i_ren, _obj, _filemenu_item):
        """ A callback to handle scroll and change the slot text colors.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        _filemenu_item: :class:`FileMenu2D`
        """
        self.set_slot_colors()
        i_ren.force_render()
        i_ren.event.abort()

    def directory_click_callback(self, i_ren, _obj, listboxitem):
        """ A callback to move into a directory if it has been clicked.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        listboxitem: :class:`ListBoxItem2D`
        """
        if (listboxitem.element, "directory") in self.directory_contents:
            new_directory_path = os.path.join(self.current_directory,
                                              listboxitem.element)
            if os.access(new_directory_path, os.R_OK):
                self.current_directory = new_directory_path
                self.directory_contents = self.get_all_file_names()
                content_names = [x[0] for x in self.directory_contents]
                self.listbox.clear_selection()
                self.listbox.values = content_names
                self.listbox.view_offset = 0
                self.listbox.update()
                self.listbox.update_scrollbar()
                self.set_slot_colors()
        i_ren.force_render()
        i_ren.event.abort()


class ComboBox2D(UI):
    """ UI element to create drop-down menus.

    Attributes
    ----------
    selection_box: :class: 'TextBox2D'
        Display selection and placeholder text.
    drop_down_button: :class: 'Button2D'
        Button to show or hide menu.
    drop_down_menu: :class: 'ListBox2D'
        Container for item list.
    """

    def __init__(self, items=[], position=(0, 0), size=(300, 200),
                 placeholder="Choose selection...", draggable=True,
                 selection_text_color=(0, 0, 0), selection_bg_color=(1, 1, 1),
                 menu_text_color=(0.2, 0.2, 0.2),
                 selected_color=(0.9, 0.6, 0.6),
                 unselected_color=(0.6, 0.6, 0.6),
                 scroll_bar_active_color=(0.6, 0.2, 0.2),
                 scroll_bar_inactive_color=(0.9, 0.0, 0.0), menu_opacity=1.,
                 reverse_scrolling=False, font_size=20, line_spacing=1.4):
        """

        Parameters
        ----------
        items: list(string)
            List of items to be displayed as choices.
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of this
            UI component.
        size : (int, int)
            Width and height in pixels of this UI component.
        placeholder : str
            Holds the default text to be displayed.
        draggable: {True, False}
            Whether the UI element is draggable or not.
        selection_text_color : tuple of 3 floats
            Color of the selected text to be displayed.
        selection_bg_color : tuple of 3 floats
            Background color of the selection text.
        menu_text_color : tuple of 3 floats.
            Color of the options displayed in drop down menu.
        selected_color : tuple of 3 floats.
            Background color of the selected option in drop down menu.
        unselected_color : tuple of 3 floats.
            Background color of the unselected option in drop down menu.
        scroll_bar_active_color : tuple of 3 floats.
            Color of the scrollbar when in active use.
        scroll_bar_inactive_color : tuple of 3 floats.
            Color of the scrollbar when inactive.
        reverse_scrolling: {True, False}
            If True, scrolling up will move the list of files down.
        font_size: int
            The font size of selected text in pixels.
        line_spacing: float
            Distance between drop down menu's items in pixels.
        """
        self.items = items.copy()
        self.font_size = font_size
        self.reverse_scrolling = reverse_scrolling
        self.line_spacing = line_spacing
        self.panel_size = size
        self._selection = placeholder
        self._menu_visibility = False
        self._selection_ID = None
        self.draggable = draggable
        self.sel_text_color = selection_text_color
        self.sel_bg_color = selection_bg_color
        self.menu_txt_color = menu_text_color
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        self.scroll_active_color = scroll_bar_active_color
        self.scroll_inactive_color = scroll_bar_inactive_color
        self.menu_opacity = menu_opacity

        # Define subcomponent sizes.
        self.text_block_size = (int(0.8*size[0]), int(0.3*size[1]))
        self.drop_menu_size = (size[0], int(0.7*size[1]))
        self.drop_button_size = (int(0.2*size[0]), int(0.3*size[1]))

        self._icon_files = [
            ('left', read_viz_icons(fname='circle-left.png')),
            ('down', read_viz_icons(fname='circle-down.png'))]

        super(ComboBox2D, self).__init__()
        self.position = position

    def _setup(self):
        """ Setup this UI component.
        Create the ListBox filled with empty slots (ListBoxItem2D).
        Create TextBox with placeholder text.
        Create Button for toggling drop down menu.
        """
        self.selection_box = TextBlock2D(
            size=self.text_block_size, color=self.sel_text_color,
            bg_color=self.sel_bg_color, text=self._selection)

        self.drop_down_button = Button2D(
            icon_fnames=self._icon_files, size=self.drop_button_size)

        self.drop_down_menu = ListBox2D(
            values=self.items, multiselection=False,
            font_size=self.font_size, line_spacing=self.line_spacing,
            text_color=self.menu_txt_color, selected_color=self.selected_color,
            unselected_color=self.unselected_color,
            scroll_bar_active_color=self.scroll_active_color,
            scroll_bar_inactive_color=self.scroll_inactive_color,
            background_opacity=self.menu_opacity,
            reverse_scrolling=self.reverse_scrolling, size=self.drop_menu_size)

        self.drop_down_menu.set_visibility(False)

        self.panel = Panel2D(self.panel_size, opacity=0.0)
        self.panel.add_element(self.selection_box, (0.001, 0.7))
        self.panel.add_element(self.drop_down_button, (0.8, 0.7))
        self.panel.add_element(self.drop_down_menu, (0, 0))

        if self.draggable:
            self.drop_down_button.on_left_mouse_button_dragged =\
                self.left_button_dragged
            self.drop_down_menu.panel.background.on_left_mouse_button_dragged\
                = self.left_button_dragged
            self.selection_box.on_left_mouse_button_dragged =\
                self.left_button_dragged
            self.selection_box.background.on_left_mouse_button_dragged =\
                self.left_button_dragged

            self.drop_down_button.on_left_mouse_button_pressed =\
                self.left_button_pressed
            self.drop_down_menu.panel.background.on_left_mouse_button_pressed\
                = self.left_button_pressed
            self.selection_box.on_left_mouse_button_pressed =\
                self.left_button_pressed
            self.selection_box.background.on_left_mouse_button_pressed =\
                self.left_button_pressed
        else:
            self.panel.background.on_left_mouse_button_dragged =\
                lambda i_ren, _obj, _comp: i_ren.force_render
            self.drop_down_menu.panel.background.on_left_mouse_button_dragged\
                = lambda i_ren, _obj, _comp: i_ren.force_render

        # Handle mouse wheel events on the slots.
        for slot in self.drop_down_menu.slots:
            slot.add_callback(
                slot.textblock.actor, "LeftButtonPressEvent",
                self.select_option_callback)

            slot.add_callback(
                slot.background.actor, "LeftButtonPressEvent",
                self.select_option_callback)

            self.drop_down_button.on_left_mouse_button_clicked = \
                self.menu_toggle_callback

        # Offer some standard hooks to the user.
        self.on_change = lambda ui: None

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.panel.actors

    def resize(self, size):
        """ Resizes ComboBox2D.

        Parameters
        ----------
        size : (int, int)
            ComboBox size(width, height) in pixels.
        """
        self.panel.resize(size)

        self.text_block_size = (int(0.8*size[0]), int(0.3*size[1]))
        self.drop_menu_size = (size[0], int(0.7*size[1]))
        self.drop_button_size = (int(0.2*size[0]), int(0.3*size[1]))

        self.panel.update_element(self.selection_box, (0.001, 0.7))
        self.panel.update_element(self.drop_down_button, (0.8, 0.7))
        self.panel.update_element(self.drop_down_menu, (0, 0))

        self.drop_down_button.resize(self.drop_button_size)
        self.drop_down_menu.resize(self.drop_menu_size)
        self.selection_box.resize(self.text_block_size)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.panel.position = coords

    def _add_to_scene(self, scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.panel.add_to_scene(scene)
        self.selection_box.font_size = self.font_size

    def _get_size(self):
        return self.panel.size

    @property
    def selected_text(self):
        return self._selection

    @property
    def selected_text_index(self):
        return self._selection_ID

    def append_item(self, *items):
        """ Append additional options to the menu.

        Parameters
        ----------
        items : n-d list, n-d tuple, Number or str
            Additional options.
        """
        for item in items:
            if isinstance(item, (list, tuple)):
                # Useful when n-d lists/tuples are used.
                self.append_item(*item)
            elif isinstance(item, (str, Number)):
                self.items.append(str(item))
            else:
                raise TypeError("Invalid item instance {}".format(type(item)))

        self.drop_down_menu.update_scrollbar()
        if not self._menu_visibility:
            self.drop_down_menu.scroll_bar.set_visibility(False)

    def select_option_callback(self, i_ren, _obj, listboxitem):
        """ Callback to select the appropriate option

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        listboxitem: :class:`ListBoxItem2D`
        """

        # Set the Text of TextBlock2D to the text of listboxitem
        self._selection = listboxitem.element
        self._selection_ID = self.items.index(self._selection)

        self.selection_box.message = self._selection
        clip_overflow(self.selection_box,
                      self.selection_box.background.size[0])
        self.drop_down_menu.set_visibility(False)
        self._menu_visibility = False

        self.drop_down_button.next_icon()

        self.on_change(self)

        i_ren.force_render()
        i_ren.event.abort()

    def menu_toggle_callback(self, i_ren, _vtkactor, _combobox):
        """ Callback to toggle visibility of drop down menu list.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        combobox : :class:`ComboBox2D`
        """

        self._menu_visibility = not self._menu_visibility
        self.drop_down_menu.set_visibility(self._menu_visibility)

        self.drop_down_button.next_icon()

        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def left_button_pressed(self, i_ren, _obj, _sub_component):
        click_pos = np.array(i_ren.event.position)
        self._click_position = click_pos
        i_ren.event.abort()  # Stop propagating the event.

    def left_button_dragged(self, i_ren, _obj, _sub_component):
        click_position = np.array(i_ren.event.position)
        change = click_position - self._click_position
        self.panel.position += change
        self._click_position = click_position
        i_ren.force_render()


class TabUI(UI):
    """ UI element to add multiple panels within a single window.

    Attributes
    ----------
    tabs: :class: List of 'TabPanel2D'
        Stores all the instances of 'TabPanel2D' that renderes the contents.
    """

    def __init__(self, position=(0, 0), size=(100, 100), nb_tabs=1,
                 active_color=(1, 1, 1), inactive_color=(0.5, 0.5, 0.5),
                 draggable=False):
        """

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
        """
        self.tabs = []
        self.nb_tabs = nb_tabs
        self.parent_size = size
        self.content_size = (size[0], int(0.9 * size[1]))
        self.draggable = draggable
        self.active_color = active_color
        self.inactive_color = inactive_color
        self.active_tab_idx = None
        self.collapsed = True

        super(TabUI, self).__init__()
        self.position = position

    def _setup(self):
        """ Setup this UI component.
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

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        actors = []
        actors += self.parent_panel.actors
        for tab_panel in self.tabs:
            actors += tab_panel.actors

        return actors

    def _add_to_scene(self, _scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.parent_panel.add_to_scene(_scene)
        for tab_panel in self.tabs:
            tab_panel.add_to_scene(_scene)

    def _set_position(self, _coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.parent_panel.position = _coords

    def _get_size(self):
        return self.parent_panel.size

    def update_tabs(self):
        """ Update position, size and callbacks for tab panels.
        """
        self.tab_panel_size =\
            (self.size[0] // self.nb_tabs, int(0.1*self.size[1]))

        tab_panel_pos = [0.0, 0.9]
        for tab_panel in self.tabs:
            tab_panel.resize(self.tab_panel_size)
            tab_panel.content_panel.position = self.position

            content_panel = tab_panel.content_panel
            if self.draggable:
                tab_panel.panel.background.on_left_mouse_button_pressed =\
                    self.left_button_pressed
                content_panel.background.on_left_mouse_button_pressed =\
                    self.left_button_pressed
                tab_panel.text_block.on_left_mouse_button_pressed =\
                    self.left_button_pressed

                tab_panel.panel.background.on_left_mouse_button_dragged =\
                    self.left_button_dragged
                content_panel.background.on_left_mouse_button_dragged =\
                    self.left_button_dragged
                tab_panel.text_block.on_left_mouse_button_dragged =\
                    self.left_button_dragged
            else:
                tab_panel.panel.background.on_left_mouse_button_dragged =\
                    lambda i_ren, _obj, _comp: i_ren.force_render
                content_panel.background.on_left_mouse_button_dragged =\
                    lambda i_ren, _obj, _comp: i_ren.force_render

            tab_panel.text_block.on_left_mouse_button_clicked =\
                self.select_tab_callback
            tab_panel.panel.background.on_left_mouse_button_clicked =\
                self.select_tab_callback

            tab_panel.text_block.on_right_mouse_button_clicked =\
                self.collapse_tab_ui
            tab_panel.panel.background.on_right_mouse_button_clicked =\
                self.collapse_tab_ui

            tab_panel.content_panel.resize(self.content_size)
            self.parent_panel.add_element(tab_panel, tab_panel_pos)
            self.parent_panel.add_element(tab_panel.content_panel, (0.0, 0.0))
            tab_panel_pos[0] += 1/self.nb_tabs

    def select_tab_callback(self, iren, _obj, _tab_comp):
        """ Handles events when a tab is selected.
        """
        for idx, tab_panel in enumerate(self.tabs):
            if tab_panel.text_block is not _tab_comp and\
               tab_panel.panel.background is not _tab_comp:
                tab_panel.color = self.inactive_color
                tab_panel.content_panel.set_visibility(False)
            else:
                tab_panel.color = self.active_color
                tab_panel.content_panel.set_visibility(True)
                self.active_tab_idx = idx

        self.collapsed = False
        self.on_change(self)
        iren.force_render()
        iren.event.abort()

    def collapse_tab_ui(self, iren, _obj, _tab_comp):
        """ Handles events when Tab UI is collapsed.
        """
        if self.active_tab_idx is not None:
            active_tab_panel = self.tabs[self.active_tab_idx]
            active_tab_panel.color = self.inactive_color
            active_tab_panel.content_panel.set_visibility(False)
        self.active_tab_idx = None
        self.collapsed = True
        self.on_collapse(self)
        iren.force_render()
        iren.event.abort()

    def add_element(self, tab_idx, element, coords, anchor="position"):
        """ Adds element to content panel after checking its existence.
        """
        if tab_idx < self.nb_tabs and tab_idx >= 0:
            self.tabs[tab_idx].add_element(element, coords, anchor)
        else:
            raise IndexError(
        "Tab with index {} does not exist".format(tab_idx))

    def remove_element(self, tab_idx, element):
        """ Removes element from content panel after checking its existence.
        """
        if tab_idx < self.nb_tabs and tab_idx >= 0:
            self.tabs[tab_idx].remove_element(element)
        else:
            raise IndexError(
        "Tab with index {} does not exist".format(tab_idx))

    def update_element(self, tab_idx, element, coords, anchor="position"):
        """ Updates element on content panel after checking its existence.
        """
        if tab_idx < self.nb_tabs and tab_idx >= 0:
            self.tabs[tab_idx].update_element(element, coords, anchor)
        else:
            raise IndexError(
        "Tab with index {} does not exist".format(tab_idx))

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


class TabPanel2D(UI):
    """ Renders content within a Tab.

    Attributes
    ----------
    content_panel: :class: 'Panel2D'
        Holds all the content UI components.
    text_block: :class: 'TextBlock2D'
        Renders the title of the tab.
    """

    def __init__(self, position=(0, 0), size=(100, 100),
                 title="New Tab", color=(0.5, 0.5, 0.5), content_panel=None):
        """

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
        """ Setup this UI component.
        Create parent panel.
        Create Text to hold tab information.
        Create Button to close tab.
        """
        self.panel = Panel2D(size=self.panel_size)
        self.text_block = TextBlock2D(size=self._text_size,
                                      color=(0, 0, 0))
        self.panel.add_element(self.text_block, (0, 0))

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.panel.actors + self.content_panel.actors

    def _add_to_scene(self, _scene):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene
        """
        self.panel.add_to_scene(_scene)
        self.content_panel.add_to_scene(_scene)

    def _set_position(self, _coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.panel.position = _coords

    def _get_size(self):
        self.panel.size

    def resize(self, size):
        """ Resizes Tab panel.

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
        """ Returns the background color of tab panel.
        """
        return self.panel.color

    @color.setter
    def color(self, color):
        """ Sets background color of tab panel.

        Parameters
        ----------
        color : list of 3 floats.
        """
        self.panel.color = color

    @property
    def title(self):
        """ Returns the title of tab panel.
        """
        return self.text_block.message

    @title.setter
    def title(self, text):
        """ Sets the title of tab panel.

        Parameters
        ----------
        text : str
            New title for tab panel.
        """
        self.text_block.message = text

    def add_element(self, element, coords, anchor="position"):
        """ Adds a UI component to the content panel.

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
        """ Removes a UI component from the content panel.

        Parameters
        ----------
        element : UI
            The UI item to be removed.
        """
        self.content_panel.remove_element(element)

    def update_element(self, element, coords, anchor="position"):
        """ Updates the position of a UI component in the content panel.

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
        self.content_panel.update_element(element, coords, anchor="position")


class GridUI(UI):
    """ Add actors in a grid and interact with them individually.
    """

    def __init__(self,
                 actors, captions=None, caption_offset=(0, -100, 0),
                 cell_padding=0,
                 cell_shape="rect", aspect_ratio=16/9., dim=None,
                 rotation_speed=1, rotation_axis=(0, 1, 0)):

        # TODO: add rotation axis None by default

        self.container = grid(actors, captions=captions,
                              caption_offset=caption_offset,
                              cell_padding=cell_padding,
                              cell_shape=cell_shape,
                              aspect_ratio=aspect_ratio, dim=dim)
        self._actors = []
        self._actors_dict = {}
        self.rotation_speed = rotation_speed
        self.rotation_axis = rotation_axis

        for item in self.container._items:
            self._actors.append(item._items[0])
            self._actors_dict[item._items[0]] = {'x': -np.inf, 'y': -np.inf}

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
        clockwise_rotation = np.array([self.rotation_speed,
                                       rx, ry, rz])
        rotate(obj, clockwise_rotation)

        istyle.force_render()
        istyle.event.abort()

    @staticmethod
    def left_release_callback2(istyle, _obj, _what):

        istyle.force_render()
        istyle.event.abort()

    @staticmethod
    def mouse_move_callback2(istyle, obj, self):

        if self._actors_dict[obj]['y'] == - np.inf:

            iren = istyle.GetInteractor()
            event_pos = iren.GetEventPosition()
            self._actors_dict[obj]['y'] = event_pos[1]

        else:

            iren = istyle.GetInteractor()
            event_pos = iren.GetEventPosition()
            rx, ry, rz = self.rotation_axis

            if event_pos[1] >= self._actors_dict[obj]['y']:
                clockwise_rotation = np.array([-self.rotation_speed,
                                               rx, ry, rz])
                rotate(obj, clockwise_rotation)
            else:
                anti_clockwise_rotation = np.array(
                    [self.rotation_speed, rx, ry, rz])
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
        if istyle.event.key == "Left":
            has_changed = True
            for a in self._actors:
                rotate(a, self.ANTICLOCKWISE_ROTATION_Y)
        elif istyle.event.key == "Right":
            has_changed = True
            for a in self._actors:
                rotate(a, self.CLOCKWISE_ROTATION_Y)
        elif istyle.event.key == "Up":
            has_changed = True
            for a in self._actors:
                rotate(a, self.ANTICLOCKWISE_ROTATION_X)
        elif istyle.event.key == "Down":
            has_changed = True
            for a in self._actors:
                rotate(a, self.CLOCKWISE_ROTATION_X)

        if has_changed:
            istyle.force_render()

    def _setup(self):
        """Set up this UI component and the events of its actor
        """
        # Add default events listener to the VTK actor.
        for actor in self._actors:
            # self.handle_events(actor)

            if self.rotation_axis is None:
                self.add_callback(actor, "LeftButtonPressEvent",
                                  self.left_click_callback)
                self.add_callback(actor, "LeftButtonReleaseEvent",
                                  self.left_release_callback)
                self.add_callback(actor, "MouseMoveEvent",
                                  self.mouse_move_callback)
            else:
                self.add_callback(actor, "LeftButtonPressEvent",
                                  self.left_click_callback2)
                # TODO: possibly add this too
                self.add_callback(actor, "LeftButtonReleaseEvent",
                                  self.left_release_callback2)
                self.add_callback(actor, "MouseMoveEvent",
                                  self.mouse_move_callback2)

            # TODO: this is currently not running
            self.add_callback(actor, "KeyPressEvent",
                              self.key_press_callback)
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
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        # coords = (0, 0, 0)
        pass
        # self.actor.SetPosition(*coords)
        # self.container.SetPosition(*coords)
