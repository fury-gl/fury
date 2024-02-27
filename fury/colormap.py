import json
from os.path import join as pjoin
from warnings import warn

import numpy as np
from scipy import linalg

from fury.data import DATA_DIR
from fury.lib import LookupTable

# Allow import, but disable doctests if we don't have matplotlib
from fury.optpkg import optional_package

cm, have_matplotlib, _ = optional_package('matplotlib.cm')


def colormap_lookup_table(
    scale_range=(0, 1),
    hue_range=(0.8, 0),
    saturation_range=(1, 1),
    value_range=(0.8, 0.8),
):
    """Lookup table for the colormap.

    Parameters
    ----------
    scale_range : tuple
        It can be anything e.g. (0, 1) or (0, 255). Usually it is the minimum
        and maximum value of your data. Default is (0, 1).
    hue_range : tuple of floats
        HSV values (min 0 and max 1). Default is (0.8, 0).
    saturation_range : tuple of floats
        HSV values (min 0 and max 1). Default is (1, 1).
    value_range : tuple of floats
        HSV value (min 0 and max 1). Default is (0.8, 0.8).

    Returns
    -------
    lookup_table : LookupTable

    """
    lookup_table = LookupTable()
    lookup_table.SetRange(scale_range)
    lookup_table.SetTableRange(scale_range)

    lookup_table.SetHueRange(hue_range)
    lookup_table.SetSaturationRange(saturation_range)
    lookup_table.SetValueRange(value_range)

    lookup_table.Build()
    return lookup_table


def cc(na, nd):
    return na * np.cos(nd * np.pi / 180.0)


def ss(na, nd):
    return na * np.sin(nd * np.pi / 180.0)


def boys2rgb(v):
    """Boys 2 rgb cool colormap

    Maps a given field of undirected lines (line field) to rgb
    colors using Boy's Surface immersion of the real projective
    plane.
    Boy's Surface is one of the three possible surfaces
    obtained by gluing a Mobius strip to the edge of a disk.
    The other two are the crosscap and Roman surface,
    Steiner surfaces that are homeomorphic to the real
    projective plane (Pinkall 1986). The Boy's surface
    is the only 3D immersion of the projective plane without
    singularities.
    Visit http://www.cs.brown.edu/~cad/rp2coloring for further details.
    Cagatay Demiralp, 9/7/2008.

    Code was initially in matlab and was rewritten in Python for fury by
    the FURY Team. Thank you Cagatay for putting this online.

    Parameters
    ----------
    v : array, shape (N, 3) of unit vectors (e.g., principal eigenvectors of
       tensor data) representing one of the two directions of the
       undirected lines in a line field.

    Returns
    -------
    c : array, shape (N, 3) matrix of rgb colors corresponding to the vectors
           given in V.

    Examples
    --------
    >>> from fury import colormap
    >>> v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> c = colormap.boys2rgb(v)

    """
    if v.ndim == 1:
        x = v[0]
        y = v[1]
        z = v[2]

    if v.ndim == 2:
        x = v[:, 0]
        y = v[:, 1]
        z = v[:, 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2

    x3 = x * x2
    y3 = y * y2
    z3 = z * z2

    z4 = z * z2

    xy = x * y
    xz = x * z
    yz = y * z

    hh1 = 0.5 * (3 * z2 - 1) / 1.58
    hh2 = 3 * xz / 2.745
    hh3 = 3 * yz / 2.745
    hh4 = 1.5 * (x2 - y2) / 2.745
    hh5 = 6 * xy / 5.5
    hh6 = (1 / 1.176) * 0.125 * (35 * z4 - 30 * z2 + 3)
    hh7 = 2.5 * x * (7 * z3 - 3 * z) / 3.737
    hh8 = 2.5 * y * (7 * z3 - 3 * z) / 3.737
    hh9 = ((x2 - y2) * 7.5 * (7 * z2 - 1)) / 15.85
    hh10 = ((2 * xy) * (7.5 * (7 * z2 - 1))) / 15.85
    hh11 = 105 * (4 * x3 * z - 3 * xz * (1 - z2)) / 59.32
    hh12 = 105 * (-4 * y3 * z + 3 * yz * (1 - z2)) / 59.32

    s0 = -23.0
    s1 = 227.9
    s2 = 251.0
    s3 = 125.0

    ss23 = ss(2.71, s0)
    cc23 = cc(2.71, s0)
    ss45 = ss(2.12, s1)
    cc45 = cc(2.12, s1)
    ss67 = ss(0.972, s2)
    cc67 = cc(0.972, s2)
    ss89 = ss(0.868, s3)
    cc89 = cc(0.868, s3)

    X = 0.0

    X = X + hh2 * cc23
    X = X + hh3 * ss23

    X = X + hh5 * cc45
    X = X + hh4 * ss45

    X = X + hh7 * cc67
    X = X + hh8 * ss67

    X = X + hh10 * cc89
    X = X + hh9 * ss89

    Y = 0.0

    Y = Y + hh2 * -ss23
    Y = Y + hh3 * cc23

    Y = Y + hh5 * -ss45
    Y = Y + hh4 * cc45

    Y = Y + hh7 * -ss67
    Y = Y + hh8 * cc67

    Y = Y + hh10 * -ss89
    Y = Y + hh9 * cc89

    Z = 0.0

    Z = Z + hh1 * -2.8
    Z = Z + hh6 * -0.5
    Z = Z + hh11 * 0.3
    Z = Z + hh12 * -2.5

    # scale and normalize to fit
    # in the rgb space

    w_x = 4.1925
    trl_x = -2.0425
    w_y = 4.0217
    trl_y = -1.8541
    w_z = 4.0694
    trl_z = -2.1899

    if v.ndim == 2:

        N = len(x)
        C = np.zeros((N, 3))

        C[:, 0] = 0.9 * np.abs(((X - trl_x) / w_x)) + 0.05
        C[:, 1] = 0.9 * np.abs(((Y - trl_y) / w_y)) + 0.05
        C[:, 2] = 0.9 * np.abs(((Z - trl_z) / w_z)) + 0.05

    if v.ndim == 1:

        C = np.zeros((3,))
        C[0] = 0.9 * np.abs(((X - trl_x) / w_x)) + 0.05
        C[1] = 0.9 * np.abs(((Y - trl_y) / w_y)) + 0.05
        C[2] = 0.9 * np.abs(((Z - trl_z) / w_z)) + 0.05

    return C


def orient2rgb(v):
    """Get Standard orientation 2 rgb colormap.

    v : array, shape (N, 3) of vectors not necessarily normalized

    Returns
    -------
    c : array, shape (N, 3) matrix of rgb colors corresponding to the vectors
           given in V.

    Examples
    --------
    >>> from fury import colormap
    >>> v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> c = colormap.orient2rgb(v)

    """
    if v.ndim == 1:
        r = np.linalg.norm(v)
        orient = np.abs(np.divide(v, r, where=r != 0))

    elif v.ndim == 2:
        orientn = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
        orientn.shape = orientn.shape + (1,)
        orient = np.abs(np.divide(v, orientn, where=orientn != 0))
    else:
        raise IOError(
            'Wrong vector dimension, It should be an array' ' with a shape (N, 3)'
        )

    return orient


def line_colors(streamlines, cmap='rgb_standard'):
    """Create colors for streamlines to be used in actor.line.

    Parameters
    ----------
    streamlines : sequence of ndarrays
    cmap : ('rgb_standard', 'boys_standard')

    Returns
    -------
    colors : ndarray

    """
    if cmap == 'rgb_standard':
        col_list = [
            orient2rgb(streamline[-1] - streamline[0]) for streamline in streamlines
        ]

    if cmap == 'boys_standard':
        col_list = [
            boys2rgb(streamline[-1] - streamline[0]) for streamline in streamlines
        ]

    return np.vstack(col_list)


lowercase_cm_name = {'blues': 'Blues', 'accent': 'Accent'}
dipy_cmaps = None


def get_cmap(name):
    """Make a callable, similar to maptlotlib.pyplot.get_cmap."""
    if name.lower() == 'accent':
        warn(
            'The `Accent` colormap is deprecated as of version'
            + ' 0.2 of Fury and will be removed in a future '
            + 'version. Please use another colormap',
            PendingDeprecationWarning,
        )

    global dipy_cmaps
    if dipy_cmaps is None:
        filename = pjoin(DATA_DIR, 'dipy_colormaps.json')
        with open(filename) as f:
            dipy_cmaps = json.load(f)

    desc = dipy_cmaps.get(name)
    if desc is None:
        return None

    def simple_cmap(v):
        """Emulate matplotlib colormap callable."""
        rgba = np.ones((len(v), 4))
        for i, color in enumerate(('red', 'green', 'blue')):
            x, y0, _ = zip(*desc[color])
            # Matplotlib allows more complex colormaps, but for users who do
            # not have Matplotlib fury makes a few simple colormaps available.
            # These colormaps are simple because y0 == y1, and therefore we
            # ignore y1 here.
            rgba[:, i] = np.interp(v, x, y0)
        return rgba

    return simple_cmap


def create_colormap(v, name='plasma', auto=True):
    """Create colors from a specific colormap and return it
    as an array of shape (N,3) where every row gives the corresponding
    r,g,b value. The colormaps we use are similar with those of matplotlib.

    Parameters
    ----------
    v : (N,) array
        vector of values to be mapped in RGB colors according to colormap
    name : str.
        Name of the colormap. Currently implemented: 'jet', 'blues',
        'accent', 'bone' and matplotlib colormaps if you have matplotlib
        installed. For example, we suggest using 'plasma', 'viridis' or
        'inferno'. 'jet' is popular but can be often misleading and we will
        deprecate it the future.
    auto : bool,
        if auto is True then v is interpolated to [0, 1] from v.min()
        to v.max()

    Notes
    -----
    FURY supports a few colormaps for those who do not use Matplotlib, for
    more colormaps consider downloading Matplotlib (see matplotlib.org).

    """
    if not have_matplotlib:
        msg = 'You do not have Matplotlib installed. Some colormaps'
        msg += ' might not work for you. Consider downloading Matplotlib.'
        warn(msg)

    if name.lower() == 'jet':
        msg = 'Jet is a popular colormap but can often be misleading'
        msg += 'Use instead plasma, viridis, hot or inferno.'
        warn(msg, PendingDeprecationWarning)

    if v.ndim > 1:
        msg = 'This function works only with 1d arrays. Use ravel()'
        raise ValueError(msg)

    if auto:
        v = np.interp(v, [v.min(), v.max()], [0, 1])
    else:
        v = np.clip(v, 0, 1)

    # For backwards compatibility with lowercase names
    newname = lowercase_cm_name.get(name) or name

    colormap = getattr(cm, newname) if have_matplotlib else get_cmap(newname)
    if colormap is None:
        e_s = 'Colormap {} is not yet implemented '.format(name)
        raise ValueError(e_s)

    rgba = colormap(v)
    rgb = rgba[:, :3].copy()
    return rgb


def _lab_delta(x, y):
    dL = y[:, 0] - x[:, 0]  # L
    dA = y[:, 1] - x[:, 1]  # A
    dB = y[:, 2] - x[:, 2]  # B
    return np.sqrt(dL**2 + dA**2 + dB**2)


def _rgb_lab_delta(x, y):
    labX = _rgb2lab(x)
    labY = _rgb2lab(y)
    return _lab_delta(labX, labY)


def _rgb2xyz(rgb):
    var_R = rgb[:, 0] / 255  # R from 0 to 255
    var_G = rgb[:, 1] / 255  # G from 0 to 255
    var_B = rgb[:, 2] / 255  # B from 0 to 255

    idx = var_R > 0.04045
    var_R[idx] = ((var_R[idx] + 0.055) / 1.055) ** 2.4
    idx = np.logical_not(idx)
    var_R[idx] = var_R[idx] / 12.92

    idx = var_G > 0.04045
    var_G[idx] = ((var_G[idx] + 0.055) / 1.055) ** 2.4
    idx = np.logical_not(idx)
    var_G[idx] = var_G[idx] / 12.92

    idx = var_B > 0.04045
    var_B[idx] = ((var_B[idx] + 0.055) / 1.055) ** 2.4
    idx = np.logical_not(idx)
    var_B[idx] = var_B[idx] / 12.92

    var_R = var_R * 100
    var_G = var_G * 100
    var_B = var_B * 100

    # Observer. = Illuminant = D65
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    return np.c_[X, Y, Z]


def _xyz2lab(xyz):
    ref_X = 095.047
    ref_Y = 100.000
    ref_Z = 108.883
    var_X = xyz[:, 0] / ref_X
    var_Y = xyz[:, 1] / ref_Y
    var_Z = xyz[:, 2] / ref_Z

    idx = var_X > 0.008856
    var_X[idx] = var_X[idx] ** (1 / 3)
    idx = np.logical_not(idx)
    var_X[idx] = (7.787 * var_X[idx]) + (16.0 / 116.0)

    idx = var_Y > 0.008856
    var_Y[idx] = var_Y[idx] ** (1 / 3)
    idx = np.logical_not(idx)
    var_Y[idx] = (7.787 * var_Y[idx]) + (16.0 / 116.0)

    idx = var_Z > 0.008856
    var_Z[idx] = var_Z[idx] ** (1 / 3)
    idx = np.logical_not(idx)
    var_Z[idx] = (7.787 * var_Z[idx]) + (16.0 / 116.0)

    L = (116 * var_Y) - 16
    A = 500 * (var_X - var_Y)
    B = 200 * (var_Y - var_Z)

    return np.c_[L, A, B]


def _lab2xyz(lab):
    var_Y = (lab[:, 0] + 16) / 116.0
    var_X = lab[:, 1] / 500.0 + var_Y
    var_Z = var_Y - lab[:, 2] / 200.0

    if var_Y**3 > 0.008856:
        var_Y = var_Y**3
    else:
        var_Y = (var_Y - 16.0 / 116.0) / 7.787

    if var_X**3 > 0.008856:
        var_X = var_X**3
    else:
        var_X = (var_X - 16.0 / 116.0) / 7.787

    if var_Z**3 > 0.008856:
        var_Z = var_Z**3
    else:
        var_Z = (var_Z - 16.0 / 116.0) / 7.787

    ref_X = 095.047
    ref_Y = 100.000
    ref_Z = 108.883
    X = ref_X * var_X
    Y = ref_Y * var_Y
    Z = ref_Z * var_Z

    return np.c_[X, Y, Z]


def _xyz2rgb(xyz):
    var_X = xyz[:, 0] / 100  # X from 0 to  95.047
    var_Y = xyz[:, 1] / 100  # Y from 0 to 100.000
    var_Z = xyz[:, 2] / 100  # Z from 0 to 108.883

    var_R = var_X * 03.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y * 01.8758 + var_Z * 00.0415
    var_B = var_X * 00.0557 + var_Y * -0.2040 + var_Z * 01.0570

    if var_R > 0.0031308:
        var_R = 1.055 * (var_R ** (1 / 2.4)) - 0.055
    else:
        var_R = 12.92 * var_R

    if var_G > 0.0031308:
        var_G = 1.055 * (var_G ** (1 / 2.4)) - 0.055
    else:
        var_G = 12.92 * var_G

    if var_B > 0.0031308:
        var_B = 1.055 * (var_B ** (1 / 2.4)) - 0.055
    else:
        var_B = 12.92 * var_B

    R = var_R * 255
    G = var_G * 255
    B = var_B * 255

    return np.c_[R, G, B]


def _rgb2lab(rgb):
    tmp = _rgb2xyz(rgb)
    return _xyz2lab(tmp)


def _lab2rgb(lab):
    tmp = _lab2xyz(lab)
    return _xyz2rgb(tmp)


def distinguishable_colormap(bg=(0, 0, 0), exclude=[], nb_colors=None):
    """Generate colors that are maximally perceptually distinct.

    This function generates a set of colors which are distinguishable
    by reference to the "Lab" color space, which more closely matches
    human color perception than RGB. Given an initial large list of possible
    colors, it iteratively chooses the entry in the list that is farthest (in
    Lab space) from all previously-chosen entries. While this "greedy"
    algorithm does not yield a global maximum, it is simple and efficient.
    Moreover, the sequence of colors is consistent no matter how many you
    request, which facilitates the users' ability to learn the color order
    and avoids major changes in the appearance of plots when adding or
    removing lines.

    Parameters
    ----------
    bg : tuple (optional)
        Background RGB color, to make sure that your colors are also
        distinguishable from the background. Default: (0, 0, 0).
    exclude : list of tuples (optional)
        Additional RGB colors to be distinguishable from.
    nb_colors : int (optional)
        Number of colors desired. Default: generate as many colors as needed.

    Returns
    -------
    iterable of ndarray
        If `nb_colors` is provided, returns a list of RBG colors.
        Otherwise, yields the next RBG color maximally perceptually
        distinct from previous ones.

    Examples
    --------
    >>> from dipy.viz.colormap import distinguishable_colormap
    >>> # Generate 5 colors
    >>> [c for i, c in zip(range(5), distinguishable_colormap())]
    [array([ 0.,  1.,  0.]),
     array([ 1.,  0.,  1.]),
     array([ 1.        ,  0.75862069,  0.03448276]),
     array([ 0.        ,  1.        ,  0.89655172]),
     array([ 0.        ,  0.17241379,  1.        ])]


    Notes
    -----
    Code was initially in matlab and was rewritten in Python for dipy by
    the Dipy Team. Thank you Tim Holy for putting this online. Visit
    http://www.mathworks.com/matlabcentral/fileexchange/29702 for the
    original implementation (v1.2), 14 Dec 2010 (Updated 07 Feb 2011).

    """
    NB_DIVISIONS = 30  # This constant come from the original code.

    # Generate a sizable number of RGB triples. This represents our space of
    # possible choices. By starting in RGB space, we ensure that all of the
    # colors can be generated by the monitor.

    colors_to_exclude = np.array([bg] + exclude)
    # Divisions along each axis in RGB space.
    x = np.linspace(0, 1, NB_DIVISIONS)
    R, G, B = np.meshgrid(x, x, x)
    rgb = np.c_[R.flatten(), G.flatten(), B.flatten()]

    lab = _rgb2lab(rgb)
    bglab = _rgb2lab(colors_to_exclude)

    def _generate_next_color():
        lastlab = bglab[0]
        mindist2 = np.ones(len(rgb)) * np.inf
        for bglab_i in bglab[1:]:
            dist2 = np.sum((lab - bglab_i) ** 2, axis=1)
            # Dist2 to closest previously-chosen color.
            mindist2 = np.minimum(dist2, mindist2)

        while True:
            dX = lab - lastlab  # Displacement of last from all colors on list.
            dist2 = np.sum(dX**2, axis=1)  # Square distance.
            # Dist2 to closest previously-chosen color.
            mindist2 = np.minimum(dist2, mindist2)
            # Find the entry farthest from all previously-chosen colors.
            idx = np.argmax(mindist2)
            yield rgb[idx]

            lastlab = lab[idx]

    if nb_colors is not None:
        return [c for i, c in zip(range(nb_colors), _generate_next_color())]

    return _generate_next_color()


def hex_to_rgb(color):
    """Converts Hexadecimal color code to rgb()

    color : string containing hexcode of color (can also start with a hash)

    Returns
    -------
    c : array, shape(1, 3) matrix of rbg colors corresponding to the
        hexcode string given in color.

    Examples
    --------
    >>> from fury import colormap
    >>> color = "#FFFFFF"
    >>> c = colormap.hex_to_rgb(color)


    >>> from fury import colormap
    >>> color = "FFFFFF"
    >>> c = colormap.hex_to_rgb(color)

    """
    if color[0] == '#':
        color = color[1:]

    r = int('0x' + color[0:2], 0) / 255
    g = int('0x' + color[2:4], 0) / 255
    b = int('0x' + color[4:6], 0) / 255

    return np.array([r, g, b])


def rgb2hsv(rgb):
    """RGB to HSV color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in HSV format. Same dimensions as input.

    Notes
    -----
    Original Implementation from scikit-image package.
    it can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
    This implementation might have been modified.

    """
    input_is_one_pixel = rgb.ndim == 1
    if input_is_one_pixel:
        rgb = rgb[np.newaxis, ...]

    out = np.empty_like(rgb)

    # -- V channel
    out_v = rgb.max(-1)

    # -- S channel
    delta = rgb.ptp(-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.0] = 0.0

    # -- H channel
    # red is max
    idx = rgb[..., 0] == out_v
    out[idx, 0] = (rgb[idx, 1] - rgb[idx, 2]) / delta[idx]

    # green is max
    idx = rgb[..., 1] == out_v
    out[idx, 0] = 2.0 + (rgb[idx, 2] - rgb[idx, 0]) / delta[idx]

    # blue is max
    idx = rgb[..., 2] == out_v
    out[idx, 0] = 4.0 + (rgb[idx, 0] - rgb[idx, 1]) / delta[idx]
    out_h = (out[..., 0] / 6.0) % 1.0
    out_h[delta == 0.0] = 0.0

    np.seterr(**old_settings)

    # -- output
    out[..., 0] = out_h
    out[..., 1] = out_s
    out[..., 2] = out_v

    # # remove NaN
    out[np.isnan(out)] = 0

    if input_is_one_pixel:
        out = np.squeeze(out, axis=0)

    return out


def hsv2rgb(hsv):
    """HSV to RGB color space conversion.

    Parameters
    ----------
    hsv : (..., 3, ...) array_like
        The image in HSV format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Notes
    -----
    Original Implementation from scikit-image package.
    it can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
    This implementation might have been modified.

    """
    hi = np.floor(hsv[..., 0] * 6)
    f = hsv[..., 0] * 6 - hi
    p = hsv[..., 2] * (1 - hsv[..., 1])
    q = hsv[..., 2] * (1 - f * hsv[..., 1])
    t = hsv[..., 2] * (1 - (1 - f) * hsv[..., 1])
    v = hsv[..., 2]

    hi = np.stack([hi, hi, hi], axis=-1).astype(np.uint8) % 6
    out = np.choose(
        hi,
        np.stack(
            [
                np.stack((v, t, p), axis=-1),
                np.stack((q, v, p), axis=-1),
                np.stack((p, v, t), axis=-1),
                np.stack((p, q, v), axis=-1),
                np.stack((t, p, v), axis=-1),
                np.stack((v, p, q), axis=-1),
            ]
        ),
    )
    return out


# From sRGB specification
xyz_from_rgb = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)

rgb_from_xyz = linalg.inv(xyz_from_rgb)


def xyz2rgb(xyz):
    """XYZ to RGB color space conversion.

    Parameters
    ----------
    xyz : (..., 3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Notes
    -----
    Original Implementation from scikit-image package.
    it can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
    This implementation might have been modified.

    """
    arr = xyz @ rgb_from_xyz.T.astype(xyz.dtype)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    np.clip(arr, 0, 1, out=arr)
    return arr


def rgb2xyz(rgb):
    """RGB to XYZ color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in XYZ format. Same dimensions as input.

    Notes
    -----
    Original Implementation from scikit-image package.
    it can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
    This implementation might have been modified.

    """
    rgb = rgb.astype(float)
    mask = rgb > 0.04045
    rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] /= 12.92
    return rgb @ xyz_from_rgb.T.astype(rgb.dtype)


# XYZ coordinates of the illuminants, scaled to [0, 1]. For each illuminant I.
# Original Implementation of this object is from scikit-image package.
# it can be found at:
# https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
illuminants = {
    'A': {
        '2': (1.098466069456375, 1, 0.3558228003436005),
        '10': (1.111420406956693, 1, 0.3519978321919493),
        'R': (1.098466069456375, 1, 0.3558228003436005),
    },
    'B': {
        '2': (0.9909274480248003, 1, 0.8531327322886154),
        '10': (0.9917777147717607, 1, 0.8434930535866175),
        'R': (0.9909274480248003, 1, 0.8531327322886154),
    },
    'C': {
        '2': (0.980705971659919, 1, 1.1822494939271255),
        '10': (0.9728569189782166, 1, 1.1614480488951577),
        'R': (0.980705971659919, 1, 1.1822494939271255),
    },
    'D50': {
        '2': (0.9642119944211994, 1, 0.8251882845188288),
        '10': (0.9672062750333777, 1, 0.8142801513128616),
        'R': (0.9639501491621826, 1, 0.8241280285499208),
    },
    'D55': {
        '2': (0.956797052643698, 1, 0.9214805860173273),
        '10': (0.9579665682254781, 1, 0.9092525159847462),
        'R': (0.9565317453467969, 1, 0.9202554587037198),
    },
    'D65': {
        '2': (0.95047, 1.0, 1.08883),
        '10': (0.94809667673716, 1, 1.0730513595166162),
        'R': (0.9532057125493769, 1, 1.0853843816469158),
    },
    'D75': {
        '2': (0.9497220898840717, 1, 1.226393520724154),
        '10': (0.9441713925645873, 1, 1.2064272211720228),
        'R': (0.9497220898840717, 1, 1.226393520724154),
    },
    'E': {'2': (1.0, 1.0, 1.0), '10': (1.0, 1.0, 1.0), 'R': (1.0, 1.0, 1.0)},
}


def get_xyz_coords(illuminant, observer):
    """Get the XYZ coordinates of the given illuminant and observer [1]_.

    Parameters
    ----------
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function grDevices::convertColor.

    Returns
    -------
    out : array
        Array with 3 elements containing the XYZ coordinates of the given
        illuminant.

    Notes
    -----
    Original Implementation from scikit-image package.
    it can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
    This implementation might have been modified.

    """
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return np.asarray(illuminants[illuminant][observer], dtype=float)
    except KeyError:
        raise ValueError(
            f'Unknown illuminant/observer combination '
            f'(`{illuminant}`, `{observer}`)'
        )


def xyz2lab(xyz, illuminant='D65', observer='2'):
    """XYZ to CIE-LAB color space conversion.

    Parameters
    ----------
    xyz : (..., 3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function grDevices::convertColor.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in CIE-LAB format. Same dimensions as input.

    Notes
    -----
    Original Implementation from scikit-image package.
    it can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
    This implementation might have been modified.

    """
    xyz_ref_white = get_xyz_coords(illuminant, observer)

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = xyz / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = np.cbrt(arr[mask])
    arr[~mask] = 7.787 * arr[~mask] + 16.0 / 116.0

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # Vector scaling
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return np.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)


def lab2xyz(lab, illuminant='D65', observer='2'):
    """CIE-LAB to XYZcolor space conversion.

    Parameters
    ----------
    lab : (..., 3, ...) array_like
        The image in Lab format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case-sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in XYZ format. Same dimensions as input.

    Notes
    -----
    Original Implementation from scikit-image package.
    it can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
    This implementation might have been modified.

    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    if np.any(z < 0):
        invalid = np.nonzero(z < 0)
        warn(
            'Color data out of range: Z < 0 in %s pixels' % invalid[0].size,
            stacklevel=2,
        )
        z[invalid] = 0

    out = np.stack([x, y, z], axis=-1)

    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.0)
    out[~mask] = (out[~mask] - 16.0 / 116.0) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    out *= xyz_ref_white
    return out


def rgb2lab(rgb, illuminant='D65', observer='2'):
    """Conversion from the sRGB color space (IEC 61966-2-1:1999)
    to the CIE Lab colorspace under the given illuminant and observer.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in Lab format. Same dimensions as input.

    Notes
    -----
    Original Implementation from scikit-image package.
    it can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
    This implementation might have been modified.

    """
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)


def lab2rgb(lab, illuminant='D65', observer='2'):
    """Lab to RGB color space conversion.

    Parameters
    ----------
    lab : (..., 3, ...) array_like
        The image in Lab format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Notes
    -----
    Original Implementation from scikit-image package.
    it can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
    This implementation might have been modified.

    """
    return xyz2rgb(lab2xyz(lab, illuminant, observer))
