"""Medical actors for FURY."""

import numpy as np

from fury.actor import (
    Group,
    apply_affine_to_group,
    contour_from_volume,
    data_slicer,
    ellipsoid_slicer,
    show_slices,
    vector_field_slicer,
)
from fury.colormap import normalize_colors
from fury.utils import get_transformed_cube_bounds


def volume_slicer(
    data,
    *,
    affine=None,
    value_range=None,
    opacity=1.0,
    interpolation="linear",
    visibility=(True, True, True),
    initial_slices=None,
    alpha_mode="auto",
    depth_write=False,
):
    """
    Visualize a 3D volume data as a slice.

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
    alpha_mode : str, optional
        The alpha mode for the material. Please see the below link for details:
        https://docs.pygfx.org/stable/_autosummary/materials/pygfx.materials.Material.html#pygfx.materials.Material.alpha_mode.
    depth_write : bool, optional
        Whether to write depth information for the material.

    Returns
    -------
    Group
        An actor containing the generated slice with the specified properties.
    """
    obj = data_slicer(
        data,
        value_range=value_range,
        opacity=opacity,
        interpolation=interpolation,
        visibility=visibility,
        initial_slices=initial_slices,
        alpha_mode=alpha_mode,
        depth_write=depth_write,
    )

    if affine is not None:
        apply_affine_to_group(obj, affine)
        bounds = obj.get_bounding_box()
        show_slices(obj, np.asarray(bounds).mean(axis=0))

    return obj


def peaks_slicer(
    peak_dirs,
    *,
    affine=None,
    peak_values=1.0,
    actor_type="thin_line",
    cross_section=None,
    colors=None,
    opacity=1.0,
    thickness=1.0,
    visibility=(True, True, True),
):
    """
    Visualize peaks as lines in 3D space.

    Parameters
    ----------
    peak_dirs : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
        The directions of the peaks.
    affine : ndarray, shape (4, 4), optional
        The affine transformation matrix to apply to the peak directions.
    peak_values : float or ndarray, optional
        The values associated with each peak direction. If a single float is provided,
        it is applied uniformly to all peaks.
    actor_type : str, optional
        The type of actor to create for the peaks. Options are 'thin_line' and
        'line'.
    cross_section : float, optional
        The cross-section size for the peaks. If None, it defaults to a small value.
    colors : ndarray, shape (N, 3) or None, optional
        The colors for each peak direction. If None, a default color is used.
    opacity : float, optional
        The opacity of the peaks. Takes values from 0 (fully transparent) to 1 (opaque).
    thickness : float, optional
        The thickness of the peaks if `actor_type` is 'thick_line'.
    visibility : tuple, optional
        A tuple of three boolean values indicating the visibility of the peaks in the x,
        y, and z dimensions, respectively.

    Returns
    -------
    VectorField
        An actor containing the generated peaks with the specified properties.
    """
    obj = vector_field_slicer(
        peak_dirs,
        scales=peak_values,
        actor_type=actor_type,
        cross_section=cross_section,
        colors=colors,
        opacity=opacity,
        thickness=thickness,
        visibility=visibility,
    )

    if affine is not None:
        bounds = obj.get_bounding_box()
        for child in obj.children:
            child.local.matrix = affine @ child.local.matrix
        obj.bounds = get_transformed_cube_bounds(
            affine,
            bounds[0],
            bounds[1],
        )
        for child in obj.children:
            child.bounds = obj.bounds
        show_slices(obj, np.asarray(obj.bounds).mean(axis=0))

    return obj


def tensor_slicer(
    evals,
    evecs,
    *,
    affine=None,
    mask=None,
    colors=None,
    scale=1.0,
    norm=True,
    cross_section=None,
    opacity=1.0,
    visibility=(True, True, True),
    enable_picking=True,
):
    """
    Render tensor eigenpairs as a sliced ellipsoid field.

    Parameters
    ----------
    evals : ndarray, shape (X, Y, Z, 3)
        Eigenvalues paired with the corresponding eigenvector columns.
        Negative values are clipped to zero; zero axes have no renderable volume.
    evecs : ndarray, shape (X, Y, Z, 3, 3)
        Orthonormal eigenvector columns. Eigenpairs need not be sorted, and
        either handedness is supported.
    affine : ndarray, shape (4, 4), optional
        Finite affine transforming voxel placement and glyph shape.
    mask : ndarray, shape (X, Y, Z), optional
        Converted to boolean: zero is inactive and nonzero is active. Must match
        the exact spatial shape. Masked eigenpairs may contain invalid values.
    colors : str, tuple, list or ndarray, optional
        Uniform RGB(A) or hex color, or an (X, Y, Z, 3/4) color field.
        Numeric colors accept [0, 1] and [0, 255]. None uses shader ``orient2rgb``
        of the principal eigenvector, weighted by fractional anisotropy in the
        input data frame; ties select the first maximum nonnegative eigenvalue.
    scale : float, optional
        Finite, nonnegative semi-axis multiplier, validated and applied by
        ``ellipsoid_slicer``.
    norm : bool, optional
        Normalize each voxel's nonnegative eigenvalues by their maximum before
        scaling. If False, preserve eigenvalue magnitudes.
    cross_section : list, tuple or ndarray, optional
        Three world-space plane positions, clamped to transformed voxel-center
        bounds and converted to integers. None selects the center-bounds
        midpoint after an affine, or the voxel-shape midpoint without one.
    opacity : float, optional
        Material opacity in [0, 1].
    visibility : tuple, optional
        Visibility of the X, Y and Z slice planes.
    enable_picking : bool, optional
        Enable surface picking with the original full-field voxel index.

    Returns
    -------
    Group
        A flat group of material-bearing resident ellipsoid tiles.

    Notes
    -----
    Inputs are not modified. Unmasked eigenvalues must be finite.

    FA is computed on eigenvalues already clipped to be nonnegative. Noisy
    DTI voxels with a negative eigenvalue get a different FA than
    ``dipy.reconst.dti.fractional_anisotropy(evals)``, so the default colors
    do not match DIPY's. For matching FA, compute it from the raw finite
    eigenvalues (masked entries zeroed), then clip only the glyph lengths.
    Any voxel with a nonpositive eigenvalue is omitted after clipping: a zero
    semi-axis makes the entire glyph inactive, rather than rendering a flat disc.
    """
    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    if evals.ndim != 4 or evals.shape[-1] != 3 or evecs.shape != (*evals.shape, 3):
        raise ValueError("Eigenvalues and eigenvectors have incompatible shapes.")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != evals.shape[:3]:
            raise ValueError("Mask must have shape (X, Y, Z).")
    if affine is not None:
        affine = np.asarray(affine)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("Affine must be a finite (4, 4) matrix.")

    lengths = np.array(evals, dtype=np.result_type(evals.dtype, np.float32), copy=True)
    if mask is not None:
        np.copyto(lengths, 0, where=~mask[..., None])
    if not np.isfinite(lengths).all():
        raise ValueError("Unmasked eigenvalues must be finite.")
    np.maximum(lengths, 0, out=lengths)
    orientation_colors = colors is None
    if norm or orientation_colors:
        maxima = lengths.max(axis=-1)
        normalized = lengths if norm else lengths.copy()
        np.divide(
            normalized,
            maxima[..., None],
            out=normalized,
            where=maxima[..., None] > 0,
        )
    if orientation_colors:
        ev1, ev2, ev3 = np.moveaxis(normalized, -1, 0)
        denominator = ev1**2 + ev2**2 + ev3**2
        all_zero = denominator == 0
        fa = np.sqrt(
            0.5
            * ((ev1 - ev2) ** 2 + (ev2 - ev3) ** 2 + (ev3 - ev1) ** 2)
            / (denominator + all_zero)
        )
        np.clip(fa, 0, 1, out=fa)
        colors = np.broadcast_to(fa[..., None], lengths.shape)

    obj = ellipsoid_slicer(
        lengths,
        orientation_matrices=evecs,
        mask=mask,
        colors=colors,
        scale=scale,
        cross_section=None if affine is not None else cross_section,
        opacity=opacity,
        visibility=visibility,
        enable_picking=enable_picking,
    )
    if orientation_colors:
        for child in obj.children:
            child.material.orientation_colors = True
    if affine is not None:
        bounds = get_transformed_cube_bounds(affine, obj.bounds[0], obj.bounds[1])
        obj.bounds = bounds
        for child in obj.children:
            child.local.matrix = affine @ child.local.matrix
            child.bounds = bounds
        show_slices(
            obj,
            np.asarray(bounds).mean(axis=0) if cross_section is None else cross_section,
        )
    return obj


def contour_from_roi(
    data, *, affine=None, color=(1, 0, 0), opacity=0.5, material="phong"
):
    """
    Generate surface actor from a binary ROI.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z)
        An ROI file that will be binarized and displayed.
    affine : ndarray, optional
        The affine transformation matrix to apply to the contour.
    color : str, tuple, list or ndarray, optional
        The output color of the contour. Accepts a hex string, RGB(A) in
        [0, 1], or RGB(A) in [0, 255].
    opacity : float, optional
        The opacity of the contour.
        Takes values from 0 (fully transparent) to 1 (opaque).
    material : str, optional
        The material type for the contour mesh. Options are 'phong' and 'basic'.

    Returns
    -------
    Group
        A group of actors containing the generated contours of ROI from the volume data.
    """
    contours = contour_from_volume(
        data, color=color, opacity=opacity, material=material
    )

    if affine is not None:
        apply_affine_to_group(contours, affine)

    return contours


def contour_from_label(data, *, affine=None, colors=None, opacities=None):
    """
    Generate surface actor from a labeled volume.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z)
        A labeled volume where each integer label represents a different region.
    affine : ndarray, optional
        The affine transformation matrix to apply to the contour.
    colors : list or ndarray, shape (N, 3) or (N, 4), optional
        A color for each unique label in the volume, where N is the number
        of unique labels (excluding background label 0). Each entry accepts a
        hex string, RGB(A) in [0, 1], or RGB(A) in [0, 255]. If None, random
        colors will be assigned.
    opacities : ndarray, shape (N,) or None, optional
        The opacities of the contours.
        Takes values from 0 (fully transparent) to 1 (opaque). It will be overridden
        if RGBA colors are provided.

    Returns
    -------
    Group
        A group of actors containing the generated contours for each label in the volume
        data.
    """
    unique_roi_id = np.delete(np.unique(data), 0)

    nb_surfaces = len(unique_roi_id)

    unique_roi_surfaces = Group()

    if colors is None:
        colors = np.random.rand(nb_surfaces, 3)
    else:
        colors = normalize_colors(colors)
        if colors.shape != (nb_surfaces, 3) and colors.shape != (nb_surfaces, 4):
            raise ValueError("Incorrect color array shape")

    if colors.shape == (nb_surfaces, 4):
        opacities = colors[:, -1]
        colors = colors[:, :-1]
    else:
        if opacities is None:
            opacities = np.ones(nb_surfaces)
        elif opacities.shape != (nb_surfaces,):
            raise ValueError("Incorrect opacity array shape")

    for i, roi_id in enumerate(unique_roi_id):
        roi_data = np.isin(data, roi_id).astype(int)
        roi_surface = contour_from_roi(
            roi_data, affine=affine, color=colors[i], opacity=opacities[i]
        )
        unique_roi_surfaces.add(roi_surface)

    return unique_roi_surfaces
