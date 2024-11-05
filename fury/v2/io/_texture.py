import numpy as np
from pygfx import Texture

from fury.decorators import warn_on_args_to_kwargs
from fury.io import load_image


@warn_on_args_to_kwargs
def load_cube_map_texture(fnames, *, size=None, generate_mipmaps=True):
    images = []

    for fname in fnames:
        images.append(load_image(fname))

    data = np.stack(*images, axis=0)

    return Texture(data, dim=2, size=size, generate_mipmaps=generate_mipmaps)
