from dipy.data import get_sphere
from dipy.direction import peak_directions
from dipy.reconst.shm import sh_to_sf_matrix
from fury import actor, ui, window
from fury.data import fetch_viz_dmri, read_viz_dmri
from fury.utils import fix_winding_order


import nibabel as nib
import numpy as np


if __name__ == '__main__':
    fetch_viz_dmri()

    fodf_img = nib.load(read_viz_dmri('fodf.nii.gz'))

    sh = fodf_img.get_fdata()
    affine = fodf_img.affine

    grid_shape = sh.shape[:-1]

    sphere_low = get_sphere('repulsion100')
    B_low = sh_to_sf_matrix(sphere_low, 8, return_inv=False)

    valid_odf_mask = np.abs(sh).max(axis=-1) > 0.
    # TODO: Careful when not None mask
    """
    if mask is not None:
        valid_odf_mask = np.logical_and(valid_odf_mask, mask)
    """
    indices = np.nonzero(valid_odf_mask)

    sf = sh[indices].dot(B_low)

    relative_peak_threshold = .5
    min_separation_angle = 25

    dirs, vals, ind = peak_directions(
        sh, sphere_low, relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle)

    peak_slicer_z = actor.peak_slicer()