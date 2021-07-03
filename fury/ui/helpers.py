"""Helper variable or function for UI Elements."""

import numpy as np


TWO_PI = 2 * np.pi


def clip_overflow(textblock, width, side='right'):
    """Clips overflowing text of TextBlock2D with respect to width.

    Parameters
    ----------
    textblock : TextBlock2D
        The textblock object whose text needs to be clipped.
    width : int
        Required width of the clipped text.
    side : str, optional
        Clips the overflowing text according to side.
        It takes values "left" or "right".

    Returns
    -------
    clipped text : str
        Clipped version of the text.
    """
    side = side.lower()
    if side not in ['left', 'right']:
        raise ValueError("side can only take values 'left' or 'right'")

    original_str = textblock.message
    start_ptr = 0
    end_ptr = len(original_str)
    prev_bg = textblock.have_bg
    textblock.have_bg = False

    if textblock.size[0] == width or textblock.size[0] <= width:
        textblock.have_bg = prev_bg
        return original_str

    if side == 'left':
        original_str = original_str[::-1]

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
            if side == 'left':
                textblock.message = textblock.message[::-1]
            return textblock.message


class Watcher:
    """Class to monitor a UI element in runtime

    Attributes
    ----------
    instance: :class: `UI`
        UI element instance
    show_m: :class: `ShowManager`
        Show Manager
    is_running: bool
        Current running state of the watcher
    i_ren: :class: `CustomInteractorStyle`
        CustomInteractorStyle
    """

    def __init__(self, object):
        """Initialize the watcher class

        Parameters
        ----------
        object: :class: `UI`
            Instance of the UI element
        """
        self.instance = object
        self.show_m = None
        self.i_ren = None
        self.attr = None
        self.original_attr = None
        self.updated_attr = None
        self.is_running = False
        self.callback = None

    def start(self, delay, show_m, attr):
        """Start the watcher

        Parameters
        ----------
        delay: int
            delay between each update call
        show_m: :class: `window.ShowManager`
            show manager
        attr: str
            attribute to watch
        """
        self.show_m = show_m
        self.attr = attr
        self.i_ren = self.show_m.scene.GetRenderWindow()\
            .GetInteractor().GetInteractorStyle()

        if hasattr(self.instance, self.attr):
            self.original_attr = getattr(self.instance, attr)

            if type(self.original_attr) == np.array:
                self.original_attr = self.original_attr.tolist()
        else:
            raise(AttributeError(
                f'{self.instance} has no attribute {self.attr}')
            )

        if delay > 0:
            self.id_timer = self.show_m\
                .add_timer_callback(True, delay, self.update)
        else:
            self.id_observer = self.i_ren.AddObserver('RenderEvent',
                                                      self.update)

        self.is_running = True

    def stop(self):
        """Stop the watcher
        """
        if self.id_timer:
            self.show_m.destroy_timer(self.id_timer)

        if self.id_observer:
            self.i_ren.RemoveObserver(self.id_observer)

        self.is_running = False

    def update(self, _obj, _evt):
        """ Update the instance of UI element.
        """
        self.updated_attr = getattr(self.instance, self.attr)

        if self.original_attr != self.updated_attr:
            self.callback(self.i_ren, _obj, self.instance)
            self.original_attr = self.updated_attr
            self.i_ren.force_render()
