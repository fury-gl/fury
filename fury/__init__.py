"""Init file for visualization package."""

from __future__ import division, print_function, absolute_import
import warnings
from fury._version import get_versions

__version__ = get_versions()['version']
del get_versions

# Ignore this specific warning below from vtk < 8.2.
# FutureWarning: Conversion of the second argument of issubdtype from
# `complex` to `np.complexfloating` is deprecated. In future, it will be
# treated as `np.complex128 == np.dtype(complex).type`.
# assert not numpy.issubdtype(z.dtype, complex), \
warnings.filterwarnings("ignore",
                        message="Conversion of the second argument of"
                                " issubdtype from `complex` to"
                                " `np.complexfloating` is deprecated.*",
                        category=FutureWarning)
