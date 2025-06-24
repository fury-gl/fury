import numpy as np


class UIContextClass:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UIContextClass, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._use_v2 = False
        self._hot_ui = None
        self._active_ui = None
        self._canvas_size = np.zeros(2)
        self._initialized = True

    def set_hot_ui(self, element):
        self._hot_ui = element

    def get_hot_ui(self):
        return self._hot_ui

    def set_active_ui(self, element):
        self._active_ui = element

    def get_active_ui(self):
        return self._active_ui

    def set_canvas_size(self, size):
        size = np.array(size)
        if not np.array_equal(self._canvas_size, size):
            self._canvas_size = size

    def get_canvas_size(self):
        return self._canvas_size

    def get_is_v2_ui(self):
        return self._use_v2

    def set_is_v2_ui(self, use_v2_ui):
        self._use_v2 = use_v2_ui


UIContext = UIContextClass()
