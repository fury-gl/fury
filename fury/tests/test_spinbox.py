import numpy as np
import pytest

from fury.ui.elements import SpinBox


def test_spinbox_default_initialization():
    """Test SpinBox initializes with correct default values."""
    sb = SpinBox()
    assert sb.value == 50.0
    assert sb.min_value == 0
    assert sb.max_value == 100
    assert sb.step == 1


def test_spinbox_custom_initialization():
    """Test SpinBox initializes correctly with custom parameters."""
    sb = SpinBox(
        position=(100, 100),
        size=(180, 40),
        min_value=0,
        max_value=10,
        initial_value=5,
        step=1,
    )
    assert sb.value == 5.0
    assert sb.min_value == 0
    assert sb.max_value == 10
    assert sb.step == 1


def test_spinbox_initial_value_clamped_to_min():
    """Test that initial_value below min_value is clamped to min_value."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=-5)
    assert sb.value == 0.0


def test_spinbox_initial_value_clamped_to_max():
    """Test that initial_value above max_value is clamped to max_value."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=99)
    assert sb.value == 10.0


def test_spinbox_value_setter_clamps_below_min():
    """Test that setting value below min_value clamps to min_value."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=5)
    sb.value = -10
    assert sb.value == 0.0


def test_spinbox_value_setter_clamps_above_max():
    """Test that setting value above max_value clamps to max_value."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=5)
    sb.value = 100
    assert sb.value == 10.0


def test_spinbox_value_setter_valid():
    """Test that setting a valid value updates correctly."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=5)
    sb.value = 7
    assert sb.value == 7.0


def test_spinbox_increment():
    """Test that increment button callback increases value by step."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=5, step=1)
    sb._on_increment(event=None)
    assert sb.value == 6.0


def test_spinbox_decrement():
    """Test that decrement button callback decreases value by step."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=5, step=1)
    sb._on_decrement(event=None)
    assert sb.value == 4.0


def test_spinbox_increment_clamps_at_max():
    """Test that incrementing at max_value does not exceed max_value."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=10, step=1)
    sb._on_increment(event=None)
    assert sb.value == 10.0


def test_spinbox_decrement_clamps_at_min():
    """Test that decrementing at min_value does not go below min_value."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=0, step=1)
    sb._on_decrement(event=None)
    assert sb.value == 0.0


def test_spinbox_custom_step():
    """Test SpinBox increments and decrements by the correct custom step."""
    sb = SpinBox(min_value=0, max_value=100, initial_value=50, step=10)
    sb._on_increment(event=None)
    assert sb.value == 60.0
    sb._on_decrement(event=None)
    sb._on_decrement(event=None)
    assert sb.value == 40.0


def test_spinbox_float_step():
    """Test SpinBox works correctly with float step values."""
    sb = SpinBox(min_value=0.0, max_value=1.0, initial_value=0.5, step=0.1)
    sb._on_increment(event=None)
    assert pytest.approx(sb.value, abs=1e-6) == 0.6
    sb._on_decrement(event=None)
    sb._on_decrement(event=None)
    assert pytest.approx(sb.value, abs=1e-6) == 0.4


def test_spinbox_on_change_callback_fires_on_increment():
    """Test that on_change callback is called when value increases."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=5, step=1)
    callback_values = []
    sb.on_change = lambda ui: callback_values.append(ui.value)
    sb._on_increment(event=None)
    assert len(callback_values) == 1
    assert callback_values[0] == 6.0


def test_spinbox_on_change_callback_fires_on_decrement():
    """Test that on_change callback is called when value decreases."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=5, step=1)
    callback_values = []
    sb.on_change = lambda ui: callback_values.append(ui.value)
    sb._on_decrement(event=None)
    assert len(callback_values) == 1
    assert callback_values[0] == 4.0


def test_spinbox_on_change_not_fired_at_max():
    """Test that on_change is NOT called when already at max and incrementing."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=10, step=1)
    callback_values = []
    sb.on_change = lambda ui: callback_values.append(ui.value)
    sb._on_increment(event=None)
    assert len(callback_values) == 0


def test_spinbox_on_change_not_fired_at_min():
    """Test that on_change is NOT called when already at min and decrementing."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=0, step=1)
    callback_values = []
    sb.on_change = lambda ui: callback_values.append(ui.value)
    sb._on_decrement(event=None)
    assert len(callback_values) == 0


def test_spinbox_label_updates_on_increment():
    """Test that the label text updates correctly after increment."""
    sb = SpinBox(
        min_value=0, max_value=10, initial_value=5,
        step=1, text_template="{value:.0f}"
    )
    sb._on_increment(event=None)
    assert sb.label.message == "6"


def test_spinbox_label_updates_on_decrement():
    """Test that the label text updates correctly after decrement."""
    sb = SpinBox(
        min_value=0, max_value=10, initial_value=5,
        step=1, text_template="{value:.0f}"
    )
    sb._on_decrement(event=None)
    assert sb.label.message == "4"


def test_spinbox_text_template():
    """Test that text_template formats the label correctly."""
    sb = SpinBox(
        min_value=0.0, max_value=1.0, initial_value=0.5,
        step=0.1, text_template="{value:.2f}"
    )
    assert sb.label.message == "0.50"


def test_spinbox_get_size():
    """Test that _get_size returns the correct total size."""
    sb = SpinBox(size=(200, 50))
    size = sb._get_size()
    assert np.array_equal(size, np.array([200, 50]))


def test_spinbox_child_components_exist():
    """Test that all three child components are created after init."""
    sb = SpinBox()
    assert hasattr(sb, "btn_decrement")
    assert hasattr(sb, "btn_increment")
    assert hasattr(sb, "label")


def test_spinbox_get_actors_returns_all():
    """Test that _get_actors returns a non-empty combined list."""
    sb = SpinBox()
    actors = sb._get_actors()
    assert isinstance(actors, list)
    assert len(actors) > 0


def test_spinbox_multiple_increments():
    """Test multiple sequential increments accumulate correctly."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=0, step=1)
    for _ in range(5):
        sb._on_increment(event=None)
    assert sb.value == 5.0


def test_spinbox_multiple_decrements():
    """Test multiple sequential decrements accumulate correctly."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=10, step=1)
    for _ in range(5):
        sb._on_decrement(event=None)
    assert sb.value == 5.0


def test_spinbox_negative_range():
    """Test SpinBox works correctly with a negative value range."""
    sb = SpinBox(min_value=-10, max_value=0, initial_value=-5, step=1)
    assert sb.value == -5.0
    sb._on_increment(event=None)
    assert sb.value == -4.0
    sb._on_decrement(event=None)
    sb._on_decrement(event=None)
    assert sb.value == -6.0


def test_spinbox_value_is_float():
    """Test that value is always stored as float."""
    sb = SpinBox(min_value=0, max_value=10, initial_value=5)
    assert isinstance(sb.value, float)
