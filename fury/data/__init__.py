"""Read or fetch test or example data."""

from os.path import dirname, join as pjoin

import lazy_loader as lazy

DATA_DIR = pjoin(dirname(__file__), "files")
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

__all__.append("DATA_DIR")
