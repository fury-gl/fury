"""Read shader files."""

from os.path import join as pjoin, dirname
from fury.shaders.base import (add_shader_to_actor, add_shader_callback,
                               delete_shader_callback)

SHADERS_DIR = pjoin(dirname(__file__))


def load(filename):
    with open(pjoin(SHADERS_DIR, filename)) as shader_file:
        return shader_file.read()


__all__ = ['SHADERS_DIR', 'load', 'add_shader_to_actor', 'add_shader_callback',
           'delete_shader_callback']
