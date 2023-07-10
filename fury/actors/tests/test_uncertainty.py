"""
This spript includes the implementation of dti_uncertainty actor for the
visualization of the cones of uncertainty along with the diffusion tensors for
comparison
"""
from dipy.reconst import dti
from dipy.segment.mask import median_otsu

from fury import actor, window

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs

from dipy.data import get_fnames, read_stanford_hardi

from fury.primitive import prim_sphere


def test_uncertainty():
    hardi_fname, hardi_bval_fname, hardi_bvec_fname =\
        get_fnames('stanford_hardi')

    data, affine = load_nifti(hardi_fname)

    # load the b-values and b-vectors
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)

    from dipy.segment.mask import median_otsu

    maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                                 numpass=1, autocrop=True, dilate=2)

    uncertainty_cones = actor.dti_uncertainty(
        data=maskdata[13:43, 44:74, 28:29], bvals=bvals, bvecs=bvecs)

    scene = window.Scene()
    scene.background([255, 255, 255])

    scene.add(diffusion_tensors())
    window.show(scene, reset_camera=False)
    scene.clear()

    scene.add(uncertainty_cones)
    window.show(scene, reset_camera=False)


class Sphere:

    vertices = None
    faces = None

def diffusion_tensors():
    # https://dipy.org/documentation/1.0.0./examples_built/reconst_dti/
    img, gtab = read_stanford_hardi()
    data = img.get_data()

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
