from numpy import stack as np_stack
from pygfx import Texture

from fury.decorators import warn_on_args_to_kwargs
from fury.io import load_image


@warn_on_args_to_kwargs()
def load_cube_map_texture(fnames, *, size=None, generate_mipmaps=True):
    images = []

    for fname in fnames:
        images.append(load_image(fname))

    if size is None:
        min_side = min(*images[0].shape[:2])
        for image in images:
            min_side = min(*image.shape[:2])
        size = (min_side, min_side, 6)

    data = np_stack(images, axis=0)


    return Texture(data, dim=2, size=size, generate_mipmaps=generate_mipmaps)
