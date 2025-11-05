"""Utilities for testing."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from distutils.version import LooseVersion
from functools import partial
import io
import json
import operator
import sys
from typing import List, Optional, Tuple, Union
import warnings

from PIL import Image
import numpy as np
from numpy.testing import assert_array_equal
import scipy  # type: ignore
from scipy import ndimage
from scipy.cluster.vq import kmeans
from scipy.ndimage import gaussian_filter


@contextmanager
def captured_output():
    """Capture stdout and stderr from print or logging.

    This context manager temporarily replaces sys.stdout and sys.stderr
    to capture printed output and return it for testing.

    Returns
    -------
    out : StringIO
        Object containing captured stdout.
    err : StringIO
        Object containing captured stderr.

    Examples
    --------
    >>> def foo():
    ...    print('hello world!')
    >>> with captured_output() as (out, err):
    ...    foo()
    >>> print(out.getvalue().strip())
    hello world!
    """
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def assert_operator(value1, value2, *, msg="", op=operator.eq):
    """Check boolean statement using the given operator.

    Compares two values using the provided operator and raises
    an AssertionError if the comparison is false.

    Parameters
    ----------
    value1 : object
        First value to be compared.
    value2 : object
        Second value to be compared.
    msg : str, optional
        Error message to be displayed if the assertion fails.
        Can contain format placeholders for values.
    op : callable, optional
        Operator to compare values. Default is equality operator.

    Raises
    ------
    AssertionError
        If the comparison between value1 and value2 using op returns False.
    """
    if not op(value1, value2):
        raise AssertionError(msg.format(str(value2), str(value1)))


assert_greater_equal = partial(
    assert_operator,
    op=operator.ge,
    msg="{0} >= {1}",
)
assert_greater = partial(assert_operator, op=operator.gt, msg="{0} > {1}")
assert_less_equal = partial(assert_operator, op=operator.le, msg="{0} =< {1}")
assert_less = partial(assert_operator, op=operator.lt, msg="{0} < {1}")
assert_true = partial(
    assert_operator, value2=True, op=operator.eq, msg="False is not true"
)
assert_false = partial(
    assert_operator, value2=False, op=operator.eq, msg="True is not false"
)
assert_not_equal = partial(assert_operator, op=operator.ne)
assert_equal = partial(assert_operator, op=operator.eq)


def assert_arrays_equal(arrays1, arrays2):
    """Check that all arrays in arrays1 equal the corresponding arrays in arrays2.

    Parameters
    ----------
    arrays1 : sequence of ndarray
        First sequence of arrays to be compared.
    arrays2 : sequence of ndarray
        Second sequence of arrays to be compared.

    Raises
    ------
    AssertionError
        If any corresponding arrays are not equal.
    """
    for arr1, arr2 in zip(arrays1, arrays2, strict=False):
        assert_array_equal(arr1, arr2)


class EventCounter:
    """Count and record UI events for testing.

    This class provides functionality to count event occurrences for UI testing
    and verification. It can record counts, save them to a file, and compare them
    with expected counts.

    Parameters
    ----------
    events_names : list of str, optional
        List of event names to count. If None, defaults to common VTK events.
    """

    def __init__(self, *, events_names=None):
        """Initialize the EventCounter.

        Parameters
        ----------
        events_names : list of str, optional
            List of event names to count. If None, defaults to common VTK events.
        """
        if events_names is None:
            events_names = [
                "CharEvent",
                "MouseMoveEvent",
                "KeyPressEvent",
                "KeyReleaseEvent",
                "LeftButtonPressEvent",
                "LeftButtonReleaseEvent",
                "RightButtonPressEvent",
                "RightButtonReleaseEvent",
                "MiddleButtonPressEvent",
                "MiddleButtonReleaseEvent",
            ]

        # Events to count
        self.events_counts = dict.fromkeys(events_names, 0)

    def count(self, i_ren, _obj, _element):
        """Count events occurrences.

        Parameters
        ----------
        i_ren : object
            The interaction renderer with event data.
        _obj : object
            The object that received the event.
        _element : object
            UI element that received the event.
        """
        self.events_counts[i_ren.event.name] += 1

    def monitor(self, ui_component):
        """Add callbacks to monitor events on a UI component.

        Parameters
        ----------
        ui_component : object
            UI component with actors to monitor for events.
        """
        for event in self.events_counts:
            for obj_actor in ui_component.actors:
                ui_component.add_callback(obj_actor, event, self.count)

    def save(self, filename):
        """Save event counts to a JSON file.

        Parameters
        ----------
        filename : str
            Path to save the event counts.
        """
        with open(filename, "w") as f:
            json.dump(self.events_counts, f)

    @classmethod
    def load(cls, filename):
        """Load event counts from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file with saved event counts.

        Returns
        -------
        EventCounter
            A new EventCounter instance with loaded counts.
        """
        event_counter = cls()
        with open(filename) as f:
            event_counter.events_counts = json.load(f)

        return event_counter

    def check_counts(self, expected):
        """Compare current event counts with expected counts.

        Parameters
        ----------
        expected : EventCounter
            EventCounter instance with expected event counts.

        Raises
        ------
        AssertionError
            If the counts don't match the expected counts.
        """
        assert_equal(len(self.events_counts), len(expected.events_counts))

        # Useful loop for debugging.
        msg = "{}: {} vs. {} (expected)"
        for event, count in expected.events_counts.items():
            if self.events_counts[event] != count:
                print(msg.format(event, self.events_counts[event], count))

        msg = "Wrong count for '{}'."
        for event, count in expected.events_counts.items():
            assert_equal(
                self.events_counts[event],
                count,
                msg=msg.format(event),
            )


class clear_and_catch_warnings(warnings.catch_warnings):
    """Context manager that resets warning registry for catching warnings.

    Warnings can be slippery, because whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module. This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters. This context manager accepts a sequence of `modules`
    as a keyword argument to its constructor and:

    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.

    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.

    Parameters
    ----------
    record : bool, optional
        Specifies whether warnings should be captured by a custom
        implementation of ``warnings.showwarning()`` and be appended to a list
        returned by the context manager. Otherwise None is returned by the
        context manager. Default is True.
    modules : sequence, optional
        Sequence of modules for which to reset warnings registry on entry and
        restore on exit.

    Notes
    -----
    This class is copied (with minor modifications) from the Nibabel package.
    https://github.com/nipy/nibabel. See COPYING file distributed along with
    the Nibabel package for the copyright and license terms.

    Examples
    --------
    >>> import warnings
    >>> with clear_and_catch_warnings(modules=[np.random.rand]):
    ...     warnings.simplefilter('always')
    ...     # do something that raises a warning in np.random.rand
    """

    class_modules = ()

    def __init__(self, *, record=True, modules=()):
        """Initialize the context manager.

        Parameters
        ----------
        record : bool, optional
            Specifies whether warnings should be captured by a custom
            implementation of ``warnings.showwarning()`` and be appended to a list
            returned by the context manager. Otherwise None is returned by the
            context manager. Default is True.
        modules : sequence, optional
            Sequence of modules for which to reset warnings registry on entry and
            restore on exit.
        """
        self.modules = set(modules).union(self.class_modules)
        self._warnreg_copies = {}
        super(clear_and_catch_warnings, self).__init__(record=record)

    def __enter__(self):
        """Clear warning registry for given modules.

        Returns
        -------
        clear_and_catch_warnings
            The context manager instance.
        """
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super(clear_and_catch_warnings, self).__enter__()

    def __exit__(self, *exc_info):
        """Restore warning registry to its previous state.

        Parameters
        ----------
        *exc_info : tuple
            Exception information, if any, raised in the context.
        """
        super(clear_and_catch_warnings, self).__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])


def setup_test():
    """Set numpy print options to "legacy" for new versions of numpy.

    Configure numpy print options to maintain compatibility with older versions.
    If imported into a file, nosetest will run this before any doctests.

    References
    ----------
    https://github.com/numpy/numpy/commit/710e0327687b9f7653e5ac02d222ba62c657a718
    https://github.com/numpy/numpy/commit/734b907fc2f7af6e40ec989ca49ee6d87e21c495
    https://github.com/nipy/nibabel/pull/556
    """
    if LooseVersion(np.__version__) >= LooseVersion("1.14"):
        np.set_printoptions(legacy="1.13")

    # Temporary fix until scipy release in October 2018
    # must be removed after that
    # print the first occurrence of matching warnings for each location
    # (module + line number) where the warning is issued
    if (
        LooseVersion(np.__version__) >= LooseVersion("1.15")
        and LooseVersion(scipy.version.short_version) <= "1.1.0"
    ):
        warnings.simplefilter("default")


def check_for_warnings(warn_printed, w_msg):
    """Check for specific warnings in the warning registry.

    Parameters
    ----------
    warn_printed : list
        List of captured warnings.
    w_msg : str
        Warning message to check for.
    """
    selected_w = [w for w in warn_printed if issubclass(w.category, UserWarning)]
    assert len(selected_w) >= 1
    msg = [str(m.message) for m in selected_w]
    assert_equal(w_msg in msg, True)


# =============================================================================
# Snapshot Testing Framework
# =============================================================================
"""
Comprehensive snapshot testing utilities for FURY.

This section provides advanced image analysis tools for validating rendered
scenes, including color detection, shading analysis, opacity verification,
and component identification with shading-aware color matching.
"""


@dataclass
class SnapshotReport:
    """Container for snapshot analysis results.

    Attributes
    ----------
    objects : int
        Number of distinct objects detected
    colors_found : List[bool]
        Whether each specified color was found
    colors_detected : List[Tuple[int, int, int]]
        All unique colors detected (RGB)
    color_coverage : List[float]
        Percentage coverage of each specified color
    brightness_mean : float
        Mean brightness across the image
    brightness_std : float
        Standard deviation of brightness
    contrast : float
        Image contrast measure
    has_shading : bool
        Whether shading gradients were detected
    shading_quality : float
        Quality score for shading (0-1)
    opacity_detected : bool
        Whether semi-transparent regions were found
    opacity_levels : List[float]
        List of detected opacity values
    edge_density : float
        Density of edges in the image
    histogram : dict
        Color histogram data
    gradient_magnitude : float
        Average gradient magnitude (for shading detection)
    component_stats : List[dict]
        Statistics for each detected component
    """

    objects: int = 0
    colors_found: List[bool] = field(default_factory=list)
    colors_detected: List[Tuple[int, int, int]] = field(default_factory=list)
    color_coverage: List[float] = field(default_factory=list)
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    contrast: float = 0.0
    has_shading: bool = False
    shading_quality: float = 0.0
    opacity_detected: bool = False
    opacity_levels: List[float] = field(default_factory=list)
    edge_density: float = 0.0
    histogram: dict = field(default_factory=dict)
    gradient_magnitude: float = 0.0
    component_stats: List[dict] = field(default_factory=list)


def analyze_snapshot(
    im_arr: Union[np.ndarray, str],
    *,
    colors: Optional[Union[List, np.ndarray]] = None,
    bg_color: Optional[Union[Tuple, List, np.ndarray]] = None,
    find_objects: bool = True,
    color_tolerance: float = 20.0,
    min_object_size: int = 10,
    analyze_shading: bool = True,
    analyze_opacity: bool = True,
) -> SnapshotReport:
    """Analyze a rendered snapshot for colors, objects, shading, and opacity.

    Parameters
    ----------
    im_arr : ndarray or str
        Image array (H, W, C) or path to image file. Can be RGB or RGBA.
    colors : list or ndarray, optional
        List of colors to search for. Each color can be (R,G,B) or (R,G,B,A).
        Values should be in range [0, 255] or [0, 1] (will be auto-detected).
    bg_color : tuple or list or ndarray, optional
        Background color to exclude from analysis.
    find_objects : bool, optional
        Whether to detect and count distinct objects. Default: True.
    color_tolerance : float, optional
        Color matching tolerance (Euclidean distance). Default: 20.0.
    min_object_size : int, optional
        Minimum number of pixels for an object. Default: 10.
    analyze_shading : bool, optional
        Whether to analyze shading and gradients. Default: True.
    analyze_opacity : bool, optional
        Whether to analyze alpha channel/opacity. Default: True.

    Returns
    -------
    SnapshotReport
        Comprehensive analysis results.

    Examples
    --------
    >>> import numpy as np
    >>> from fury.testing.snapshot import analyze_snapshot
    >>> # Create a simple test image
    >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> img[25:75, 25:75] = [255, 0, 0]  # Red square
    >>> report = analyze_snapshot(img, colors=[[255, 0, 0]])
    >>> print(f"Objects found: {report.objects}")
    >>> print(f"Red found: {report.colors_found[0]}")
    """
    if isinstance(im_arr, str):
        img_pil = Image.open(im_arr)
        im_arr = np.array(img_pil)

    if not isinstance(im_arr, np.ndarray):
        raise TypeError("im_arr must be a numpy array or file path")

    if im_arr.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got {im_arr.ndim}D")

    if im_arr.ndim == 2:
        im_arr = np.stack([im_arr] * 3, axis=-1)

    has_alpha = im_arr.shape[2] == 4 if im_arr.ndim == 3 else False
    rgb = im_arr[..., :3] if has_alpha else im_arr.copy()

    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)

    report = SnapshotReport()

    if colors is not None:
        colors = _normalize_colors(colors)

    if bg_color is not None:
        bg_color = _normalize_color(bg_color)

    if bg_color is not None:
        bg_mask = _create_color_mask(rgb, bg_color, color_tolerance)
        fg_mask = ~bg_mask
    else:
        fg_mask = np.any(rgb > 10, axis=-1)

    if colors is not None:
        report.colors_found, report.color_coverage = _find_colors(
            rgb, colors, fg_mask, color_tolerance
        )

    report.colors_detected = _detect_unique_colors(rgb, fg_mask)

    if find_objects:
        report.objects, report.component_stats = _count_objects(
            fg_mask, min_object_size
        )

    report.brightness_mean, report.brightness_std = _analyze_brightness(rgb, fg_mask)
    report.contrast = _compute_contrast(rgb, fg_mask)

    if analyze_shading:
        report.has_shading, report.shading_quality, report.gradient_magnitude = (
            _analyze_shading(rgb, fg_mask)
        )

    if analyze_opacity and has_alpha:
        report.opacity_detected, report.opacity_levels = _analyze_opacity(
            im_arr[..., 3], fg_mask
        )

    report.edge_density = _compute_edge_density(rgb, fg_mask)
    report.histogram = _compute_histogram(rgb, fg_mask)

    return report


def _normalize_colors(colors):
    """Normalize color list to numpy array in 0-255 range.

    Parameters
    ----------
    colors : array-like
        Colors to normalize.

    Returns
    -------
    ndarray
        Normalized colors in 0-255 range.
    """
    colors = np.atleast_2d(colors)

    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    if colors.shape[1] >= 3:
        colors = colors[..., :3]

    return colors


def _normalize_color(color):
    """Normalize single color to numpy array in 0-255 range.

    Parameters
    ----------
    color : array-like
        Single color to normalize.

    Returns
    -------
    ndarray
        Normalized color in 0-255 range.
    """
    color = np.atleast_1d(color)

    if color.max() <= 1.0:
        color = (color * 255).astype(np.uint8)
    else:
        color = color.astype(np.uint8)

    return color[:3]


def _rgb_to_hsv(rgb):
    """Convert RGB to HSV color space for better color matching with shading.

    HSV separates hue (color) from value (brightness), making it more robust
    to lighting variations.

    Parameters
    ----------
    rgb : ndarray
        RGB color array.

    Returns
    -------
    ndarray
        HSV color array.
    """
    rgb_normalized = rgb.astype(float) / 255.0

    r, g, b = rgb_normalized[..., 0], rgb_normalized[..., 1], rgb_normalized[..., 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c

    # Hue calculation
    h = np.zeros_like(max_c)

    mask_r = (max_c == r) & (diff > 0)
    mask_g = (max_c == g) & (diff > 0)
    mask_b = (max_c == b) & (diff > 0)

    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

    # Saturation
    s = np.zeros_like(max_c)
    s[max_c > 0] = diff[max_c > 0] / max_c[max_c > 0]

    # Value
    v = max_c

    return np.stack([h, s, v], axis=-1)


def _extract_dominant_colors(rgb, mask, n_colors=5):
    """Extract dominant colors using k-means clustering.

    This finds the most representative colors in the image, accounting for
    shading variations by clustering similar colors together.

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    mask : ndarray
        Boolean mask of pixels to analyze.
    n_colors : int, optional
        Number of dominant colors to extract.

    Returns
    -------
    list
        List of dominant colors as tuples.
    """
    if not np.any(mask):
        return []

    masked_pixels = rgb[mask].astype(float)

    if len(masked_pixels) < n_colors:
        return [tuple(c.astype(int)) for c in masked_pixels]

    # Sample if too many pixels (for performance)
    if len(masked_pixels) > 10000:
        indices = np.random.choice(len(masked_pixels), 10000, replace=False)
        sample = masked_pixels[indices]
    else:
        sample = masked_pixels

    try:
        # Use k-means to find dominant colors
        centroids, _ = kmeans(sample, n_colors)
        return [tuple(c.astype(int)) for c in centroids]
    except Exception:
        # Fallback to unique colors if k-means fails
        unique_colors = np.unique(masked_pixels.reshape(-1, 3), axis=0)
        if len(unique_colors) > n_colors:
            # Return most common colors
            from collections import Counter

            pixel_tuples = [tuple(p) for p in masked_pixels.astype(int)]
            counter = Counter(pixel_tuples)
            return [color for color, _ in counter.most_common(n_colors)]
        return [tuple(c.astype(int)) for c in unique_colors]


def _color_family_match(rgb_pixel, target_color, shading_aware=True):
    """Check if a pixel belongs to the same color family as target.

    Uses HSV color space to match hue while being tolerant to brightness
    and saturation variations from shading.

    Parameters
    ----------
    rgb_pixel : ndarray
        RGB pixel color.
    target_color : ndarray
        Target RGB color to match.
    shading_aware : bool, optional
        Whether to use shading-aware matching.

    Returns
    -------
    float
        Color distance metric.
    """
    if not shading_aware:
        # Simple Euclidean distance
        return np.linalg.norm(rgb_pixel - target_color)

    # Convert to HSV for shading-aware matching
    hsv_pixel = _rgb_to_hsv(rgb_pixel.reshape(1, 1, 3))[0, 0]
    hsv_target = _rgb_to_hsv(target_color.reshape(1, 1, 3))[0, 0]

    h_pixel, s_pixel, v_pixel = hsv_pixel
    h_target, s_target, v_target = hsv_target

    # For grayscale colors (low saturation), use brightness matching
    if s_target < 0.1 or s_pixel < 0.1:
        # Both are grayscale - compare brightness
        return abs(v_pixel - v_target) * 255

    # For colored pixels, hue is most important
    # Hue distance (circular - wraps at 360)
    h_diff = min(abs(h_pixel - h_target), 360 - abs(h_pixel - h_target))

    # Normalize hue difference to 0-1 range
    h_distance = h_diff / 180.0

    # Saturation difference (less important with shading)
    s_distance = abs(s_pixel - s_target)

    # Value/brightness difference (least important with shading)
    v_distance = abs(v_pixel - v_target) * 0.3  # Reduced weight for brightness

    # Combined distance
    combined = np.sqrt(h_distance**2 * 100 + s_distance**2 * 30 + v_distance**2 * 10)

    return combined


def _create_color_mask(rgb, target_color, tolerance):
    """Create binary mask for pixels matching target color.

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    target_color : ndarray
        Target color to match.
    tolerance : float
        Color matching tolerance.

    Returns
    -------
    ndarray
        Boolean mask of matching pixels.
    """
    diff = np.linalg.norm(rgb - target_color, axis=-1)
    return diff < tolerance


def _create_shading_aware_mask(
    rgb, target_color, tolerance, shading_tolerance_factor=2.0
):
    """Create mask using shading-aware color matching.

    This uses HSV color space and allows more tolerance for brightness variations
    while being strict about hue (actual color).

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    target_color : ndarray
        Target color to match.
    tolerance : float
        Color matching tolerance.
    shading_tolerance_factor : float, optional
        Factor for shading tolerance.

    Returns
    -------
    ndarray
        Boolean mask of matching pixels.
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    # Convert both to HSV
    hsv_image = _rgb_to_hsv(rgb)
    hsv_target = _rgb_to_hsv(target_color.reshape(1, 1, 3))[0, 0]

    h_target, s_target, v_target = hsv_target

    # For grayscale targets (low saturation)
    if s_target < 0.1:
        # Match based on brightness with wider tolerance
        v_image = hsv_image[..., 2]
        v_diff = np.abs(v_image - v_target)
        mask = v_diff < (tolerance / 255.0 * shading_tolerance_factor)
    else:
        # For colored targets, focus on hue
        h_image = hsv_image[..., 0]
        s_image = hsv_image[..., 1]
        v_image = hsv_image[..., 2]

        # Hue distance (circular)
        h_diff = np.minimum(
            np.abs(h_image - h_target), 360 - np.abs(h_image - h_target)
        )

        # Hue tolerance (in degrees) - strict on actual color
        hue_match = h_diff < (tolerance / 2.0)

        # Saturation should be similar (but allow some variation)
        s_diff = np.abs(s_image - s_target)
        sat_match = s_diff < 0.4  # Allow significant saturation variation

        # Pixel must match hue and have reasonable saturation
        # Value is least important with shading
        mask = hue_match & (
            sat_match | (s_image > 0.2)
        )  # Allow if reasonably saturated

    return mask


def _find_colors(rgb, target_colors, mask, tolerance):
    """Find which target colors are present using shading-aware detection.

    This enhanced version uses HSV color space and dominant color extraction
    to handle Phong shading and lighting effects.

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    target_colors : ndarray
        Target colors to find.
    mask : ndarray
        Boolean mask of pixels to search.
    tolerance : float
        Color matching tolerance.

    Returns
    -------
    tuple
        Lists of booleans and coverage percentages.
    """
    found = []
    coverage = []

    total_pixels = np.sum(mask)

    for color in target_colors:
        # Try shading-aware matching first
        color_mask = _create_shading_aware_mask(rgb, color, tolerance) & mask
        is_found = np.any(color_mask)

        # If not found with shading-aware method, fall back to dominant color matching
        if not is_found:
            # Extract dominant colors from the masked region
            dominant = _extract_dominant_colors(rgb, mask, n_colors=5)

            # Check if any dominant color is close to target
            for dom_color in dominant:
                dom_color_arr = np.array(dom_color)
                distance = _color_family_match(dom_color_arr, color, shading_aware=True)
                if (
                    distance < tolerance * 1.5
                ):  # Slightly more tolerant for dominant colors
                    is_found = True
                    break

        found.append(bool(is_found))

        if total_pixels > 0:
            coverage.append(float(np.sum(color_mask)) / total_pixels * 100)
        else:
            coverage.append(0.0)

    return found, coverage


def _detect_unique_colors(rgb, mask, max_colors=50):
    """Detect unique colors in the image.

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    mask : ndarray
        Boolean mask of pixels to analyze.
    max_colors : int, optional
        Maximum number of unique colors to return.

    Returns
    -------
    list
        List of unique colors.
    """
    masked_pixels = rgb[mask]

    if len(masked_pixels) == 0:
        return []

    unique_colors, inverse_indices, counts = np.unique(
        masked_pixels.reshape(-1, 3), axis=0, return_inverse=True, return_counts=True
    )

    if len(unique_colors) > max_colors:
        top_indices = np.argsort(counts)[-max_colors:]
        unique_colors = unique_colors[top_indices]

    return [tuple(c) for c in unique_colors]


def _count_objects(mask, min_size=10):
    """Count distinct connected components in the mask.

    Parameters
    ----------
    mask : ndarray
        Boolean mask to analyze.
    min_size : int, optional
        Minimum object size in pixels.

    Returns
    -------
    tuple
        Number of objects and component statistics.
    """
    if not np.any(mask):
        return 0, []

    labeled, num_objects = ndimage.label(mask)

    stats = []
    valid_objects = 0

    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled == obj_id
        size = np.sum(obj_mask)

        if size >= min_size:
            valid_objects += 1

            coords = np.column_stack(np.where(obj_mask))
            bbox_min = coords.min(axis=0)
            bbox_max = coords.max(axis=0)

            stats.append(
                {
                    "id": obj_id,
                    "size": int(size),
                    "bbox": (tuple(bbox_min), tuple(bbox_max)),
                    "centroid": tuple(coords.mean(axis=0)),
                }
            )

    return valid_objects, stats


def _analyze_brightness(rgb, mask):
    """Compute brightness statistics.

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    mask : ndarray
        Boolean mask of pixels to analyze.

    Returns
    -------
    tuple
        Mean and standard deviation of brightness.
    """
    if not np.any(mask):
        return 0.0, 0.0

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    masked_gray = gray[mask]

    mean_brightness = float(np.mean(masked_gray))
    std_brightness = float(np.std(masked_gray))

    return mean_brightness, std_brightness


def _compute_contrast(rgb, mask):
    """Compute image contrast (standard deviation of pixel intensities).

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    mask : ndarray
        Boolean mask of pixels to analyze.

    Returns
    -------
    float
        Contrast value.
    """
    if not np.any(mask):
        return 0.0

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    masked_gray = gray[mask]

    return float(np.std(masked_gray))


def _analyze_shading(rgb, mask):
    """Analyze shading gradients using advanced scipy techniques.

    This uses multiple gradient operators and frequency analysis to detect
    realistic 3D shading patterns from Phong lighting.

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    mask : ndarray
        Boolean mask of pixels to analyze.

    Returns
    -------
    tuple
        Shading presence flag, quality score, and gradient magnitude.
    """
    if not np.any(mask):
        return False, 0.0, 0.0

    # Convert to grayscale
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # Multi-scale gradient analysis for better shading detection
    # Standard gradients for sharp features
    grad_x = ndimage.sobel(gray, axis=0)
    grad_y = ndimage.sobel(gray, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Smooth gradients to detect gradual shading
    smooth_gray = gaussian_filter(gray, sigma=2.0)
    smooth_gx = ndimage.sobel(smooth_gray, axis=0)
    smooth_gy = ndimage.sobel(smooth_gray, axis=1)
    smooth_gradient = np.sqrt(smooth_gx**2 + smooth_gy**2)

    masked_gradient = gradient_magnitude[mask]
    masked_smooth = smooth_gradient[mask]

    if len(masked_gradient) == 0:
        return False, 0.0, 0.0

    avg_gradient = float(np.mean(masked_gradient))
    smooth_avg = float(np.mean(masked_smooth))

    # Improved shading detection:
    # - Smooth gradients indicate realistic lighting
    # - Not just sharp edges
    has_shading = avg_gradient > 3.0 or smooth_avg > 2.0

    if avg_gradient > 0:
        # Quality based on gradient smoothness and magnitude
        smoothness = min(smooth_avg / max(avg_gradient, 0.1), 1.0)
        magnitude_score = min(avg_gradient / 30.0, 1.0)
        quality = smoothness * 0.6 + magnitude_score * 0.4
    else:
        quality = 0.0

    return has_shading, float(quality), avg_gradient


def _analyze_opacity(alpha_channel, mask):
    """Analyze alpha channel for transparency effects.

    Parameters
    ----------
    alpha_channel : ndarray
        Alpha channel array.
    mask : ndarray
        Boolean mask of pixels to analyze.

    Returns
    -------
    tuple
        Opacity presence flag and list of opacity levels.
    """
    if not np.any(mask):
        return False, []

    masked_alpha = alpha_channel[mask]

    unique_alphas = np.unique(masked_alpha)
    semi_transparent = np.any((unique_alphas > 0) & (unique_alphas < 255))

    opacity_levels = [float(a) / 255.0 for a in unique_alphas if 0 < a < 255]

    return bool(semi_transparent), opacity_levels


def _compute_edge_density(rgb, mask):
    """Compute density of edges in the image.

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    mask : ndarray
        Boolean mask of pixels to analyze.

    Returns
    -------
    float
        Edge density value.
    """
    if not np.any(mask):
        return 0.0

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    edges = ndimage.sobel(gray)
    edge_mask = edges > 20

    fg_edges = edge_mask & mask
    density = float(np.sum(fg_edges)) / np.sum(mask) if np.sum(mask) > 0 else 0.0

    return density


def _compute_histogram(rgb, mask, bins=64):
    """Compute color histogram for each channel.

    Parameters
    ----------
    rgb : ndarray
        RGB image array.
    mask : ndarray
        Boolean mask of pixels to analyze.
    bins : int, optional
        Number of histogram bins.

    Returns
    -------
    dict
        Dictionary with R, G, B histogram arrays.
    """
    histogram = {}

    if not np.any(mask):
        return histogram

    for i, channel in enumerate(["r", "g", "b"]):
        masked_channel = rgb[..., i][mask]
        hist, bin_edges = np.histogram(masked_channel, bins=bins, range=(0, 256))
        histogram[channel] = {"counts": hist.tolist(), "bin_edges": bin_edges.tolist()}

    return histogram


def assert_colors_present(
    snapshot: Union[np.ndarray, str],
    colors: Union[List, np.ndarray],
    tolerance: float = 20.0,
) -> SnapshotReport:
    """Assert that specified colors are present in the snapshot.

    Parameters
    ----------
    snapshot : ndarray or str
        Image array or file path.
    colors : list or ndarray
        Colors to check for.
    tolerance : float, optional
        Color matching tolerance.

    Returns
    -------
    SnapshotReport
        Analysis report.

    Raises
    ------
    AssertionError
        If any color is not found
    """
    report = analyze_snapshot(snapshot, colors=colors, color_tolerance=tolerance)

    for i, found in enumerate(report.colors_found):
        if not found:
            raise AssertionError(
                f"Color {colors[i]} not found in snapshot (tolerance={tolerance})"
            )

    return report


def assert_shading_present(
    snapshot: Union[np.ndarray, str], min_quality: float = 0.3
) -> SnapshotReport:
    """Assert that shading gradients are present.

    Parameters
    ----------
    snapshot : ndarray or str
        Image array or file path.
    min_quality : float, optional
        Minimum shading quality score (0-1).

    Returns
    -------
    SnapshotReport
        Analysis report.

    Raises
    ------
    AssertionError
        If shading quality is below threshold
    """
    report = analyze_snapshot(snapshot, analyze_shading=True)

    if not report.has_shading:
        raise AssertionError("No shading detected in snapshot")

    if report.shading_quality < min_quality:
        raise AssertionError(
            f"Shading quality {report.shading_quality:.2f} below "
            f"threshold {min_quality:.2f}"
        )

    return report


def assert_opacity_correct(
    snapshot: Union[np.ndarray, str],
    expected_levels: Optional[List[float]] = None,
    tolerance: float = 0.05,
) -> SnapshotReport:
    """Assert that opacity/transparency is correctly rendered.

    Parameters
    ----------
    snapshot : ndarray or str
        Image array or file path (must have alpha channel).
    expected_levels : list of float, optional
        Expected opacity levels (0-1).
    tolerance : float, optional
        Tolerance for opacity matching.

    Returns
    -------
    SnapshotReport
        Analysis report.

    Raises
    ------
    AssertionError
        If opacity is incorrect
    """
    report = analyze_snapshot(snapshot, analyze_opacity=True)

    if expected_levels is not None:
        if not report.opacity_detected:
            raise AssertionError("No semi-transparent pixels detected")

        for expected in expected_levels:
            found = any(
                abs(level - expected) < tolerance for level in report.opacity_levels
            )
            if not found:
                raise AssertionError(
                    f"Expected opacity level {expected:.2f} not found. "
                    f"Detected levels: {report.opacity_levels}"
                )

    return report


def assert_object_count(
    snapshot: Union[np.ndarray, str], expected_count: int, min_object_size: int = 10
) -> SnapshotReport:
    """Assert that the expected number of objects are present.

    Parameters
    ----------
    snapshot : ndarray or str
        Image array or file path.
    expected_count : int
        Expected number of objects.
    min_object_size : int, optional
        Minimum object size in pixels.

    Returns
    -------
    SnapshotReport
        Analysis report.

    Raises
    ------
    AssertionError
        If object count doesn't match
    """
    report = analyze_snapshot(
        snapshot, find_objects=True, min_object_size=min_object_size
    )

    if report.objects != expected_count:
        raise AssertionError(
            f"Expected {expected_count} objects, found {report.objects}"
        )

    return report
