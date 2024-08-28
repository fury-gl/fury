import os

import numpy as np
from dipy.data.fetcher import dipy_home
from dipy.io.image import load_nifti

from fury import actor, window

if __name__ == "__main__":
    show_man = window.ShowManager(size=(1280, 720))

    dataset_dir = os.path.join(dipy_home, "stanford_hardi")

    coeffs, affine = load_nifti("docs\experimental\SH-ODF_experimental\odf_debug_sh_coeffs_9x11x45(8).nii")

    valid_mask = np.abs(coeffs).max(axis=(-1)) > 0
    indices = np.nonzero(valid_mask)

    centers = np.asarray(indices).T

    x, y, z, s = coeffs.shape
    coeffs = coeffs[:, :, :].reshape((x * y * z, s))
    
    #'''
    coeffs = np.array([
        [
            -0.2739740312099, 0.2526670396328, 1.8922271728516, 0.2878578901291,
            -0.5339795947075, -0.2620058953762, 0.1580424904823, 0.0329004973173,
            -0.1322413831949, -0.1332057565451, 1.0894461870193, -0.6319401264191,
            -0.0416776277125, -1.0772529840469,  0.1423762738705, 0.7941166162491,
            0.7490307092667, -0.3428381681442, 0.1024847552180, -0.0219132602215,
            0.0499043911695, 0.2162453681231, 0.0921059995890, -0.2611238956451,
            0.2549301385880, -0.4534865319729, 0.1922748684883, -0.6200597286224,
            -0.0532187558711, -0.3569841980934, 0.0293972902000, -0.1977960765362,
            -0.1058669015765, 0.2372217923403, -0.1856198310852, -0.3373193442822,
            -0.0750469490886, 0.2146576642990, -0.0490148440003, 0.1288588196039,
            0.3173974752426, 0.1990085393190, -0.1736343950033, -0.0482443645597,
            0.1749017387629
        ]
    ])
    centers= np.array([0, 0, 0])
    #'''

    odf_actor = actor.odf(centers=centers, coeffs=coeffs, scales=1.0, 
                          sh_basis='descoteaux')
    show_man.scene.add(odf_actor)
    show_man.start()