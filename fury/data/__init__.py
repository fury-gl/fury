"""Read or fetch test or example data."""

from os.path import dirname, join as pjoin

from fury.data.fetcher import (
    fetch_gltf,
    fetch_viz_cubemaps,
    fetch_viz_dmri,
    fetch_viz_icons,
    fetch_viz_models,
    fetch_viz_new_icons,
    fetch_viz_textures,
    fetch_viz_wiki_nw,
    list_gltf_sample_models,
    read_viz_cubemap,
    read_viz_dmri,
    read_viz_gltf,
    read_viz_icons,
    read_viz_models,
    read_viz_textures,
)

DATA_DIR = pjoin(dirname(__file__), 'files')

__all__ = ['DATA_DIR', 'fetch_viz_cubemaps', 'read_viz_cubemap',
           'fetch_viz_icons', 'fetch_viz_new_icons',
           'read_viz_icons', 'fetch_viz_textures',
           'read_viz_textures', 'fetch_viz_wiki_nw',
           'fetch_viz_models', 'read_viz_models',
           'fetch_viz_dmri', 'read_viz_dmri',
           'fetch_gltf', 'read_viz_gltf',
           'list_gltf_sample_models']
