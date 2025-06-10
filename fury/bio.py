"""Medical actors for FURY."""

import numpy as np

from fury.actor import slicer
from fury.utils import apply_affine_to_group, show_slices


def volume_slicer(
    data,
    *,
    affine=None,
    value_range=None,
    opacity=1.0,
    interpolation="linear",
    visibility=(True, True, True),
    initial_slices=None,
):
    """Visualize a 3D volume data as a slice.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z) or (X, Y, Z, 3)
        The 3D volume data to be sliced.
    affine : ndarray, shape (4, 4), optional
        The affine transformation matrix to apply to the data.
    value_range : tuple, optional
        The minimum and maximum values for the color mapping.
        If None, the range is determined from the data.
    opacity : float, optional
        The opacity of the slice. Takes values from 0 (fully transparent) to 1 (opaque).
    interpolation : str, optional
        The interpolation method for the slice. Options are 'linear' and 'nearest'.
    visibility : tuple, optional
        A tuple of three boolean values indicating the visibility of the slices
        in the x, y, and z dimensions, respectively.
    initial_slices : tuple, optional
        A tuple of three initial slice positions in the x, y, and z dimensions,
        respectively. If None, the slices are initialized to the middle of the volume.

    Returns
    -------
    Group
        An actor containing the generated slice with the specified properties.
    """

    obj = slicer(
        data,
        value_range=value_range,
        opacity=opacity,
        interpolation=interpolation,
        visibility=visibility,
        initial_slices=initial_slices,
    )

    if affine is not None:
        apply_affine_to_group(obj, affine)
        bounds = obj.get_bounding_box()
        show_slices(obj, np.asarray(bounds).mean(axis=0))

    return obj
