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
        self._use_v2 = False
        self._hot_ui = None
        self._active_ui = None
        self._canvas_size = np.zeros(2)
        self._initialized = True

    def set_hot_ui(self, element):
        """Set the currently 'hot' UI element.

        Parameters
        ----------
        element : UI or None
            UI element that is currently 'hot', or `None` if no element is hot.
        """
        self._hot_ui = element

    def get_hot_ui(self):
        """Get the currently 'hot' UI element.

        Returns
        -------
        UI or None
            UI element that is currently 'hot', or `None` if no element is hot.
        """
        return self._hot_ui

    def set_active_ui(self, element):
        """Set the currently 'active' UI element.

        Parameters
        ----------
        element : UI or None
            UI element that is currently `active`, or `None` if no element is active.
        """
        self._active_ui = element

    def get_active_ui(self):
        """Get the currently 'active' UI element.

        Returns
        -------
        UI or None
            UI element that is currently 'active', or `None` if no element is active.
        """
        return self._active_ui

    def set_canvas_size(self, size):
        """Set the current size of the rendering canvas in pixels.

        Parameters
        ----------
        size : (int, int)
            Canvas `(width, height)` dimensions.
        """
        size = np.array(size)
        if not np.array_equal(self._canvas_size, size):
            self._canvas_size = size

    def get_canvas_size(self):
        """Get the current size of the rendering canvas in pixels.

        Returns
        -------
        numpy.ndarray
            Canvas `(width, height)` dimensions.
        """
        return self._canvas_size

    def get_is_v2_ui(self):
        """Get the currently active UI mode.

        Returns
        -------
        bool
            `True` if UI v2 mode is active, `False` otherwise (V1 mode).
        """
        return self._use_v2

    def set_is_v2_ui(self, use_v2_ui):
        """Set the UI mode to V1 or V2.

        Parameters
        ----------
        use_v2_ui : bool
            Set to `True` to enable V2 UI mode (top-down Y-axis interpretation).
            Set to `False` for V1 UI mode (bottom-up Y-axis interpretation).
        """
        self._use_v2 = use_v2_ui


UIContext = UIContextClass()
