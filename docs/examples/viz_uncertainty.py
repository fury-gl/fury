"""
This script includes the implementation of dti_uncertainty actor for the
visualization of the cones of uncertainty along with the diffusion tensors for
comparison
"""
from dipy.core.gradients import gradient_table
from dipy.reconst import dti
from dipy.segment.mask import median_otsu

import dipy.denoise.noise_estimate as ne

from fury import actor, window

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs

from dipy.data import get_fnames, read_stanford_hardi

from fury.primitive import prim_sphere


class Sphere:

    vertices = None
    faces = None

def diffusion_tensors():
    # https://dipy.org/documentation/1.0.0./examples_built/reconst_dti/
    img, gtab = read_stanford_hardi()
    data = img.get_fdata()

    maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                                 numpass=1, autocrop=True, dilate=2)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    evals = tenfit.evals[13:43, 44:74, 28:29]
    evecs = tenfit.evecs[13:43, 44:74, 28:29]

    vertices, faces = prim_sphere('symmetric724', True)
    sphere = Sphere()
    sphere.vertices = vertices
    sphere.faces = faces

    #from dipy.data import get_sphere
    #sphere = get_sphere('symmetric724')

    return actor.tensor_slicer(evals, evecs, sphere=sphere, scale=0.3)

if __name__ == '__main__':
    hardi_fname, hardi_bval_fname, hardi_bvec_fname =\
        get_fnames('stanford_hardi')

    data, affine = load_nifti(hardi_fname)

    # load the b-values and b-vectors
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)

    maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                                 numpass=1, autocrop=True, dilate=2)

    gtab = gradient_table(bvals, bvecs)

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata[13:43, 44:74, 28:29])

    # Eigenvalues and eigenvectors
    fevals = tenfit.evals
    fevecs = tenfit.evecs

    tensor_vals = dti.lower_triangular(tenfit.quadratic_form)
    dti_params = dti.eig_from_lo_tri(tensor_vals)

    # Predicted signal given tensor parameters
    fsignal = dti.tensor_prediction(dti_params, gtab, 1.0)

    # Design matrix or B matrix
    b_matrix = dti.design_matrix(gtab)

    # Standard deviation of the noise
    sigma = ne.estimate_sigma(maskdata[13:43, 44:74, 28:29])

    uncertainty_cones = actor.uncertainty_cone(evecs=fevecs, evals=fevals,
                                               signal=fsignal, sigma=sigma,
                                               b_matrix=b_matrix)

    scene = window.Scene()
    scene.background([255, 255, 255])

    scene.add(diffusion_tensors())
    window.show(scene, reset_camera=False)
    scene.clear()

    scene.add(uncertainty_cones)
    window.show(scene, reset_camera=False)
