"""Read shader files."""

from os.path import join as pjoin, dirname

SHADERS_DIR = pjoin(dirname(__file__))


def load(filename):
    with open(pjoin(SHADERS_DIR, filename)) as shader_file:
        return shader_file.read()


__all__ = ['SHADERS_DIR', 'load']
