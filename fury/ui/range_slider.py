from fury.ui.core import UI
from fury.ui.elements import LineSlider2D


class RangeSlider(UI):

    def __init__(
        self,
        position=(300, 300),
        min_value=0,
        max_value=100,
        initial_min=25,
        initial_max=75,
        length=200,
    ):

        self.position = position
        self.length = length

        self.min_value = min_value
        self.max_value = max_value

        self.initial_min = initial_min
        self.initial_max = initial_max

        super().__init__()

    def _setup(self):

        x, y = self.position

        self.min_slider = LineSlider2D(
            position=(x, y),
            min_value=self.min_value,
            max_value=self.max_value,
            initial_value=self.initial_min,
            length=self.length,
        )

        self.max_slider = LineSlider2D(
            position=(x, y - 40),
            min_value=self.min_value,
            max_value=self.max_value,
            initial_value=self.initial_max,
            length=self.length,
        )

        self.min_slider.on_change = self._range_changed
        self.max_slider.on_change = self._range_changed

    def _range_changed(self, slider):

        min_val = self.min_slider.value
        max_val = self.max_slider.value

        if min_val > max_val:
            if slider == self.min_slider:
                self.min_slider.value = max_val
            else:
                self.max_slider.value = min_val

        print("Selected Range:", self.get_range())

    def get_range(self):
        return (self.min_slider.value, self.max_slider.value)

    def _get_actors(self):
        return self.min_slider._get_actors() + self.max_slider._get_actors()

    def _get_size(self):
        return (self.length, 80)

    def _update_actors_position(self):
        pass
