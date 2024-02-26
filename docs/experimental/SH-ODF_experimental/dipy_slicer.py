"""
This script adds an argument parser to the ray_traced_6.0.py script.
"""

import argparse
import os

import numpy as np
from dipy.data import get_sphere
from dipy.io.image import load_nifti
from dipy.reconst.shm import sh_to_sf

from fury import actor, window


def uv_calculations(n):
    uvs = []
    for i in range(0, n):
        a = (n - (i + 1)) / n
        b = (n - i) / n
        # glyph_coord [0, a], [0, b], [1, b], [1, a]
        uvs.extend(
            [
                [0.001, a + 0.001],
                [0.001, b - 0.001],
                [0.999, b - 0.001],
                [0.999, a + 0.001],
                [0.001, a + 0.001],
                [0.001, b - 0.001],
                [0.999, b - 0.001],
                [0.999, a + 0.001],
            ]
        )
    return uvs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", help="Path to spherical harmonic coefficients file."
    )
    args = parser.parse_args()

    show_man = window.ShowManager(size=(1280, 720))

    coeffs, affine = load_nifti(args.file)

    sphere = get_sphere("repulsion724")

    sh_basis = "tournier07"
    sh_order = 8

    tensor_sf = sh_to_sf(
        coeffs,
        sh_order=sh_order,
        basis_type=sh_basis,
        sphere=sphere,
        legacy=False,
    )

    odf_slicer_actor = actor.odf_slicer(tensor_sf, sphere=sphere, scale=0.5)

    show_man.scene.add(odf_slicer_actor)

    show_man.start()
