"""Read or fetch test or example data."""

from os.path import join as pjoin, dirname

from fury.data.fetcher import (fetch_viz_icons, read_viz_icons,
                               fetch_viz_wiki_nw, fetch_viz_textures,
                               read_viz_textures, fetch_viz_models,
                               read_viz_models)

DATA_DIR = pjoin(dirname(__file__), 'files')

__all__ = ['fetch_viz_icons', 'read_viz_icons', 'DATA_DIR',
           'fetch_viz_textures', 'read_viz_textures',
           'fetch_viz_wiki_nw', 'fetch_viz_models',
           'read_viz_models']
