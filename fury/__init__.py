"""Init file for visualization package."""

import warnings
import sys
from os.path import dirname

import lazy_loader as lazy

from fury.pkg_info import __version__, pkg_commit_hash
from fury.optpkg import optional_package

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
    verbose : bool, optional
        If `True`, print the information to stdout.

    Returns
    -------
    dict
        With named parameters of interest.
    """
    import numpy
    import scipy
    

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
        # TODO: Add pygfx version if applicable
    }

    if have_mpl:
        info["matplotlib_version"] = mpl.__version__
    if have_dipy:
        info["dipy_version"] = dipy.__version__

    if verbose:
        print("\n".join([f"{k}: {v}" for k, v in info.items()]))

    return info


def enable_warnings(warnings_origin=None):
    """Enable global warnings.

    Parameters
    ----------
    warnings_origin : list
        List origin ['all', 'fury', 'matplotlib', ...].
    """
    warnings_origin = warnings_origin or ("all",)

    if "all" in warnings_origin:
        warnings.filterwarnings("default")


def disable_warnings(warnings_origin=None):
    """Disable global warnings.

    Parameters
    ----------
    warnings_origin : list
        List origin ['all', 'fury', 'matplotlib', ...].
    """
    warnings_origin = warnings_origin or ("all",)

    if "all" in warnings_origin:
        warnings.filterwarnings("ignore")


# Disable warnings in release mode
if "post" not in __version__ and "dev" not in __version__:
    disable_warnings()

# Suppress known NumPy future warning
warnings.filterwarnings(
    "ignore",
    message=(
        "Conversion of the second argument of issubdtype from `complex` to"
        " `np.complexfloating` is deprecated.*"
    ),
    category=FutureWarning,
)
