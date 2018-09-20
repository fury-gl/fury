# Init file for visualization package
from __future__ import division, print_function, absolute_import
from fury._version import get_versions

__version__ = get_versions()['version']
del get_versions

# We make the visualization requirements optional imports:

try:
    import matplotlib
    has_mpl = True
except ImportError:
    e_s = "You do not have Matplotlib installed. Some visualization functions"
    e_s += " might not work for you."
    print(e_s)
    has_mpl = False

if has_mpl:
    from . import projections
