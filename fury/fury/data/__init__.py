"""Read test or example data."""

from os.path import join as pjoin, dirname

from fury.data.fetcher import (fetch_viz_icons, read_viz_icons,
                               fetch_viz_wiki_nw)

DATA_DIR = pjoin(dirname(__file__), 'files')

__all__ = ['fetch_viz_icons', 'read_viz_icons', 'DATA_DIR']
