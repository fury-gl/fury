"""UI context module."""

import numpy as np


class UIContextState:
    """
    Hold the interaction and canvas state of one window.

    Attributes
    ----------
    hot_ui : UI or None
        UI element currently under the pointer.
    active_ui : UI or None
        UI element currently receiving keyboard input.
    canvas_size : numpy.ndarray
        Canvas `(width, height)` dimensions in pixels.
    """

    def __init__(self):
        """Initialize an empty state with the default canvas size."""
        self.hot_ui = None
        self.active_ui = None
        self.canvas_size = np.array([800, 800])


class UIContextClass:
    """
    Expose the UI state of the active window.

    Notes
    -----
    Each `ShowManager` owns a `UIContextState` and activates it whenever it
    handles input, resizes, draws, or runs its callbacks. Focus, hover, and
    canvas size therefore never leak between windows. Z-order bounds are shared.
    """

    _instance = None

    def __new__(cls):
        """
        Handle instance creation for the UI context singleton.

        Returns
        -------
        UIContextClass
            The single, shared instance of `UIContextClass`.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the UIContext singleton."""
        if self._initialized:
            return
        self._state = UIContextState()
        self._z_max = 1
        self._z_min = -1
        self._initialized = True

    def activate(self, state):
        """
        Make a window's state the one exposed by this context.

        Parameters
        ----------
        state : UIContextState
            State owned by the window that is about to run UI code.
        """
        self._state = state

    @property
    def hot_ui(self):
        """
        Get the currently hot UI element.

        Returns
        -------
        UI or None
            UI element that is currently hot, or `None` if no element is hot.
        """
        return self._state.hot_ui

    @hot_ui.setter
    def hot_ui(self, element):
        """
        Set the currently hot UI element.

        Parameters
        ----------
        element : UI or None
            UI element that is currently hot, or `None` if no element is hot.
        """
        self._state.hot_ui = element

    @property
    def active_ui(self):
        """
        Get the currently active UI element.

        Returns
        -------
        UI or None
            UI element that is currently active, or `None` if no element is active.
        """
        return self._state.active_ui

    @active_ui.setter
    def active_ui(self, element):
        """
        Set the currently active UI element.

        Parameters
        ----------
        element : UI or None
            UI element that is currently active, or `None` if no element is active.
        """
        self._state.active_ui = element

    @property
    def canvas_size(self):
        """
        Get the current size of the rendering canvas in pixels.

        Returns
        -------
        numpy.ndarray
            Canvas `(width, height)` dimensions.
        """
        return self._state.canvas_size

    @canvas_size.setter
    def canvas_size(self, size):
        """
        Set the current size of the rendering canvas in pixels.

        Parameters
        ----------
        size : (int, int)
            Canvas `(width, height)` dimensions.
        """
        self._state.canvas_size = np.array(size)

    @property
    def z_order_bounds(self):
        """
        Get the minimum and maximum Z-order values in the UI.

        Returns
        -------
        numpy.ndarray
            Z-order bounds `[z_min, z_max]`.
        """
        return np.array([self._z_min, self._z_max])

    @z_order_bounds.setter
    def z_order_bounds(self, z_order):
        """
        Update the minimum or maximum active Z-order values in the UI.

        Parameters
        ----------
        z_order : int
            Z-order value of a UI.
        """
        self._z_min = min(self._z_min, z_order)
        self._z_max = max(self._z_max, z_order)


UIContext = UIContextClass()
