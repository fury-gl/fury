"""UI context module."""

import numpy as np


class UIContextClass:
    """Manage global UI context."""

    _instance = None

    def __new__(cls):
        """Handle instance creation for the UI context singleton.

        Returns
        -------
        UIContextClass
            The single, shared instance of `UIContextClass`.
        """
        if cls._instance is None:
            cls._instance = super(UIContextClass, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the UIContext singleton."""
        if self._initialized:
            return
        self._hot_ui = None
        self._active_ui = None
        self._canvas_size = np.array([800, 800])
        self._z_max = 1
        self._z_min = -1
        self._initialized = True

    @property
    def hot_ui(self):
        """Get the currently hot UI element.

        Returns
        -------
        UI or None
            UI element that is currently hot, or `None` if no element is hot.
        """
        return self._hot_ui

    @hot_ui.setter
    def hot_ui(self, element):
        """Set the currently hot UI element.

        Parameters
        ----------
        element : UI or None
            UI element that is currently hot, or `None` if no element is hot.
        """
        self._hot_ui = element

    @property
    def active_ui(self):
        """Get the currently active UI element.

        Returns
        -------
        UI or None
            UI element that is currently active, or `None` if no element is active.
        """
        return self._active_ui

    @active_ui.setter
    def active_ui(self, element):
        """Set the currently active UI element.

        Parameters
        ----------
        element : UI or None
            UI element that is currently active, or `None` if no element is active.
        """
        self._active_ui = element

    @property
    def canvas_size(self):
        """Get the current size of the rendering canvas in pixels.

        Returns
        -------
        numpy.ndarray
            Canvas `(width, height)` dimensions.
        """
        return self._canvas_size

    @canvas_size.setter
    def canvas_size(self, size):
        """Set the current size of the rendering canvas in pixels.

        Parameters
        ----------
        size : (int, int)
            Canvas `(width, height)` dimensions.
        """
        size = np.array(size)
        if not np.array_equal(self._canvas_size, size):
            self._canvas_size = size

    @property
    def z_order_bounds(self):
        """Get the minimum and maximum Z-order values in the UI.

        Returns
        -------
        numpy.ndarray
            Z-order bounds `[z_min, z_max]`.
        """
        return np.array([self._z_min, self._z_max])

    @z_order_bounds.setter
    def z_order_bounds(self, z_order):
        """Update the minimum or maximum active Z-order values in the UI.

        Parameters
        ----------
        z_order : int
            Z-order value of a UI.
        """
        self._z_min = min(self._z_min, z_order)
        self._z_max = max(self._z_max, z_order)


UIContext = UIContextClass()
