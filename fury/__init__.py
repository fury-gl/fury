"""Init file for visualization package."""

import warnings

import lazy_loader as lazy

from fury.pkg_info import __version__, pkg_commit_hash

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

__all__ += [
    "__version__",
    "disable_warnings",
    "enable_warnings",
    "get_info",
]


def get_info(verbose=False):
    """Return dict describing the context of this package.

    Parameters
    ----------
    pkg_path : str
       path containing __init__.py for package

    Returns
    -------
    context : dict
       with named parameters of interest

    """
    from os.path import dirname
    import sys

    import numpy
    import scipy
    import vtkmodules.vtkCommonCore as ccvtk

    from fury.optpkg import optional_package

    mpl, have_mpl, _ = optional_package("matplotlib")
    dipy, have_dipy, _ = optional_package("dipy")

    install_type, commit_hash = pkg_commit_hash(dirname(__file__))

    info = {
        "fury_version": __version__,
        "pkg_path": dirname(__file__),
        "commit_hash": commit_hash,
        "sys_version": sys.version,
        "sys_executable": sys.executable,
        "sys_platform": sys.platform,
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "vtk_version": ccvtk.vtkVersion.GetVTKVersion(),
    }

    d_mpl = {"matplotlib_version": mpl.__version__} if have_mpl else {}
    d_dipy = {"dipy_version": dipy.__version__} if have_dipy else {}

    info.update(d_mpl)
    info.update(d_dipy)

    if verbose:
        print("\n".join(["{0}: {1}".format(k, v) for k, v in info.items()]))

    return info


def enable_warnings(warnings_origin=None):
    """Enable global warnings.

    Parameters
    ----------
    warnings_origin : list
        list origin ['all', 'fury', 'vtk', 'matplotlib', ...]

    """
    warnings_origin = warnings_origin or ("all",)

    if "all" in warnings_origin or "vtk" in warnings_origin:
        import vtkmodules.vtkCommonCore as ccvtk

        ccvtk.vtkObject.GlobalWarningDisplayOn()


def disable_warnings(warnings_origin=None):
    """Disable global warnings.

    Parameters
    ----------
    warnings_origin : list
        list origin ['all', 'fury', 'vtk', 'matplotlib', ...]

    """
    warnings_origin = warnings_origin or ("all",)

    if "all" in warnings_origin or "vtk" in warnings_origin:
        import vtkmodules.vtkCommonCore as ccvtk

        ccvtk.vtkObject.GlobalWarningDisplayOff()


# We switch off the warning display during the release
if "post" not in __version__ and "dev" not in __version__:
    disable_warnings()

# Ignore this specific warning below from vtk < 8.2.
# FutureWarning: Conversion of the second argument of issubdtype from
# `complex` to `np.complexfloating` is deprecated. In future, it will be
# treated as `np.complex128 == np.dtype(complex).type`.
# assert not numpy.issubdtype(z.dtype, complex), \
warnings.filterwarnings(
    "ignore",
    message="Conversion of the second argument of"
    " issubdtype from `complex` to"
    " `np.complexfloating` is deprecated.*",
    category=FutureWarning,
)
