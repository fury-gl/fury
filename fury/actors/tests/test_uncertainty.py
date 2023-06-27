"""
This spript includes the implementation of cone of uncertainty using matrix
perturbation analysis
"""
from fury import actor, window

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs

from dipy.data import get_fnames


def test_uncertainty():
    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

    data, affine = load_nifti(hardi_fname)

    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)

    from dipy.segment.mask import median_otsu

    maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                                 numpass=1, autocrop=True, dilate=2)

    uncertainty_cones = actor.dti_uncertainty(
        data=maskdata[13:43, 44:74, 28:29], bvals=bvals, bvecs=bvecs)

    scene = window.Scene()
    scene.background([255, 255, 255])

    scene.add(uncertainty_cones)

    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene, reset_camera=False)
