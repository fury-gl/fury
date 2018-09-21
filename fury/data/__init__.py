"""Read test or example data."""
from __future__ import division, print_function, absolute_import

from os.path import join as pjoin, dirname

from fury.data.fetcher import fetch_viz_icons, read_viz_icons

DATA_DIR = pjoin(dirname(__file__), 'files')

__all__ = ['fetch_viz_icons', 'read_viz_icons', 'DATA_DIR']
