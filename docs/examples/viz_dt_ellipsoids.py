"""
===============================================================================
Display Tensor Ellipsoids for DTI using tensor_slicer vs ellipsoid actor
===============================================================================
This tutorial is intended to show two ways of displaying diffusion tensor
ellipsoids for DTI visualization. The first is using the basic tensor_slicer
that allows us to slice many tensors as ellipsoids. The second is the generic
ellipsoid actor that can be used to display different amount of tensors.

We start by importing the necessary modules:
"""

import numpy as np

from dipy.io.image import load_nifti

from fury.actor import _fa, _color_fa
from fury.data import fetch_viz_dmri, read_viz_dmri


