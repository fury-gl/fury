"""UI core module that describe UI abstract class."""

__all__ = ["Rectangle2D", "Disk2D", "TextBlock2D", "Button2D"]

import abc
from warnings import warn

import numpy as np
import vtk

from fury.interactor import CustomInteractorStyle
from fury.io import load_image
from fury.utils import set_input


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
        _scene : Scene

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


class Rectangle2D(UI):
    """A 2D rectangle sub-classed from UI."""

    def __init__(self, size=(0, 0), position=(0, 0), color=(1, 1, 1),
                 opacity=1.0):
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
        super(Rectangle2D, self).__init__(position)
        self.color = color
        self.opacity = opacity
        self.resize(size)

    def _setup(self):
        """Setup this UI component.

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
        """Get the actors composing this UI component."""
        return [self.actor]

    def _add_to_scene(self, scene):
        """Add all subcomponents or VTK props that compose this UI component.

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
        """Set the button size.

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
        """Set the lower-left corner position of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).

        """
        self.actor.SetPosition(*coords)

    @property
    def color(self):
        """Get the rectangle's color."""
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """Set the rectangle's color.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].

        """
        self.actor.GetProperty().SetColor(*color)

    @property
    def opacity(self):
        """Get the rectangle's opacity."""
        return self.actor.GetProperty().GetOpacity()

    @opacity.setter
    def opacity(self, opacity):
        """Set the rectangle's opacity.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].

        """
        self.actor.GetProperty().SetOpacity(opacity)


class Disk2D(UI):
    """A 2D disk UI component."""

    def __init__(self, outer_radius, inner_radius=0, center=(0, 0),
                 color=(1, 1, 1), opacity=1.0):
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
        super(Disk2D, self).__init__()
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.color = color
        self.opacity = opacity
        self.center = center

    def _setup(self):
        """Setup this UI component.

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
        """Get the actors composing this UI component."""
        return [self.actor]

    def _add_to_scene(self, scene):
        """Add all subcomponents or VTK props that compose this UI component.

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
        """Set the lower-left corner position of this UI bounding box.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        # Disk actor are positioned with respect to their center.
        self.actor.SetPosition(*coords + self.outer_radius)

    @property
    def color(self):
        """Get the color of this UI component."""
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """Set the color of this UI component.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].

        """
        self.actor.GetProperty().SetColor(*color)

    @property
    def opacity(self):
        """Get the opacity of this UI component."""
        return self.actor.GetProperty().GetOpacity()

    @opacity.setter
    def opacity(self, opacity):
        """Set the opacity of this UI component.

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


class TextBlock2D(UI):
    """Wrap over the default vtkTextActor and helps setting the text.

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
        """Init class instance.

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
        """Get the actors composing this UI component."""
        return [self.actor] + self.background.actors

    def _add_to_scene(self, scene):
        """Add all subcomponents or VTK props that compose this UI component.

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
        """Get message from the text.

        Returns
        -------
        str
            The current text message.
        """
        return self.actor.GetInput()

    @message.setter
    def message(self, text):
        """Set the text message.

        Parameters
        ----------
        text : str
            The message to be set.
        """
        self.actor.SetInput(text)

    @property
    def font_size(self):
        """Get text font size.

        Returns
        ----------
        int
            Text font size.
        """
        return self.actor.GetTextProperty().GetFontSize()

    @font_size.setter
    def font_size(self, size):
        """Set font size.

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
        """Get font family.

        Returns
        ----------
        str
            Text font family.
        """
        return self.actor.GetTextProperty().GetFontFamilyAsString()

    @font_family.setter
    def font_family(self, family='Arial'):
        """Set font family.

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
        """Get text justification.

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
        """Justify text.

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
        """Get text vertical justification.

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
        """Justify text vertically.

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
        """Return whether the text is bold.

        Returns
        -------
        bool
            Text is bold if True.

        """
        return self.actor.GetTextProperty().GetBold()

    @bold.setter
    def bold(self, flag):
        """Bold/un-bold text.

        Parameters
        ----------
        flag : bool
            Sets text bold if True.

        """
        self.actor.GetTextProperty().SetBold(flag)

    @property
    def italic(self):
        """Return whether the text is italicised.

        Returns
        -------
        bool
            Text is italicised if True.

        """
        return self.actor.GetTextProperty().GetItalic()

    @italic.setter
    def italic(self, flag):
        """Italicise/un-italicise text.

        Parameters
        ----------
        flag : bool
            Italicises text if True.
        """
        self.actor.GetTextProperty().SetItalic(flag)

    @property
    def shadow(self):
        """Return whether the text has shadow.

        Returns
        -------
        bool
            Text is shadowed if True.
        """
        return self.actor.GetTextProperty().GetShadow()

    @shadow.setter
    def shadow(self, flag):
        """Add/remove text shadow.

        Parameters
        ----------
        flag : bool
            Shadows text if True.
        """
        self.actor.GetTextProperty().SetShadow(flag)

    @property
    def color(self):
        """Get text color.

        Returns
        -------
        (float, float, float)
            Returns text color in RGB.
        """
        return self.actor.GetTextProperty().GetColor()

    @color.setter
    def color(self, color=(1, 0, 0)):
        """Set text color.

        Parameters
        ----------
        color : (float, float, float)
            RGB: Values must be between 0-1.

        """
        self.actor.GetTextProperty().SetColor(*color)

    @property
    def background_color(self):
        """Get background color.

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
        """Set text color.

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
        """Set text actor position.

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
        """Set the lower-left corner position of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.actor.SetPosition(*coords)

    @property
    def color(self):
        """Get the button's color."""
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """Set the button's color.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].

        """
        self.actor.GetProperty().SetColor(*color)

    def scale(self, factor):
        """Scale the button.

        Parameters
        ----------
        factor : (float, float)
            Scaling factor (width, height) in pixels.

        """
        self.resize(self.size * factor)

    def set_icon_by_name(self, icon_name):
        """Set the button icon using its name.

        Parameters
        ----------
        icon_name : str

        """
        icon_id = self.icon_names.index(icon_name)
        self.set_icon(self.icons[icon_id][1])

    def set_icon(self, icon):
        """Modify the icon used by the vtkTexturedActor2D.

        Parameters
        ----------
        icon : imageDataGeometryFilter

        """
        self.texture = set_input(self.texture, icon)

    def next_icon_id(self):
        """Set the next icon ID while cycling through icons."""
        self.current_icon_id += 1
        if self.current_icon_id == len(self.icons):
            self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]

    def next_icon(self):
        """Increment the state of the Button.

        Also changes the icon.
        """
        self.next_icon_id()
        self.set_icon(self.icons[self.current_icon_id][1])
