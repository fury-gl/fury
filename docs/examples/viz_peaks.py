from dipy.data import get_sphere
from dipy.direction import peak_directions
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.viz.app import horizon
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

    #horizon(images=[(sh, affine)])

    shape = sh.shape[:-1]

    sphere = get_sphere('repulsion100')
    B = sh_to_sf_matrix(sphere, sh_order=8, return_inv=False)

    valid_odf_mask = np.abs(sh).max(axis=-1) > 0.
    # TODO: Careful when not None mask
    """
    if mask is not None:
        valid_odf_mask = np.logical_and(valid_odf_mask, mask)
    """
    indices = np.nonzero(valid_odf_mask)

    sf = sh[indices].dot(B)

    npeaks = 5

    peak_dirs = np.zeros((shape + (npeaks, 3)))
    peak_values = np.zeros((shape + (npeaks,)))
    peak_indices = np.zeros((shape + (npeaks,)), dtype='int')
    peak_indices.fill(-1)

    relative_peak_threshold = .5
    min_separation_angle = 25

    for idx in range(sf.shape[0]):
        x = indices[0][idx]
        y = indices[1][idx]
        z = indices[2][idx]
        # TODO: Keep mask into account for final implementation

        dirs, vals, ind = peak_directions(
            sf[idx], sphere, relative_peak_threshold=relative_peak_threshold,
            min_separation_angle=min_separation_angle)

        # Calculate peak metrics
        if vals.shape[0] != 0:
            n = min(npeaks, vals.shape[0])

            peak_dirs[x, y, z][:n] = dirs[:n]
            peak_indices[x, y, z][:n] = ind[:n]
            peak_values[x, y, z][:n] = vals[:n]

    peak_slicer_z = actor.peak_slicer(peak_dirs, peaks_values=peak_values)

    scene = window.Scene()
    scene.add(peak_slicer_z)

    window.show(scene)
