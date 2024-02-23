import os

import numpy as np
from dipy.data.fetcher import dipy_home
from dipy.io.image import load_nifti

from fury import actor, window

if __name__ == "__main__":
    show_man = window.ShowManager(size=(1280, 720))

    dataset_dir = os.path.join(dipy_home, "stanford_hardi")

    coeffs, affine = load_nifti("docs\experimental\SH-ODF experimental\coefs_odf.nii")

    valid_mask = np.abs(coeffs).max(axis=(-1)) > 0
    indices = np.nonzero(valid_mask)

    centers = np.asarray(indices).T

    x, y, z, s = coeffs.shape
    coeffs = coeffs[:, :, :].reshape((x * y * z, s))
    n_glyphs = coeffs.shape[0]

    odf_actor = actor.odf(centers=centers, coeffs=coeffs, scales=1.0)
    show_man.scene.add(odf_actor)
    show_man.start()