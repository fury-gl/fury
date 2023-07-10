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

from fury import window, actor
#from fury.actor import _fa, _color_fa
from fury.data import fetch_viz_dmri, read_viz_dmri
from fury.primitive import prim_sphere

# Now, we fetch and load the data needed to display the Diffusion Tensor
# Images.

fetch_viz_dmri()

# The tensor ellipsoids are expressed as eigen values and eigen vectors
# which are the decomposition of the diffusion tensor that describes the water
# diffusion within a voxel.

evecs, _ = load_nifti(read_viz_dmri('slice_evecs.nii.gz'))
evals, _ = load_nifti(read_viz_dmri('slice_evals.nii.gz'))
roi_evecs, _ = load_nifti(read_viz_dmri('roi_evecs.nii.gz'))
roi_evals, _ = load_nifti(read_viz_dmri('roi_evals.nii.gz'))
#whole_brain_evecs, _ = load_nifti(read_viz_dmri('whole_brain_evecs.nii.gz'))
#whole_brain_evals, _ = load_nifti(read_viz_dmri('whole_brain_evals.nii.gz'))

interactive = True

###############################################################################
# Using tensor_slicer actor
# =========================
#

affine = np.eye(4)

data_shape = evals.shape[:3]
mask = np.ones(data_shape).astype(bool)
vertices, faces = prim_sphere('repulsion100', True)
# vertices, faces = prim_sphere('repulsion200', True)
# vertices, faces = prim_sphere('repulsion724', True)
class Sphere:
    vertices = None
    faces = None

sphere = Sphere()
sphere.vertices = vertices
sphere.faces = faces

tensor_slicer_actor = actor.tensor_slicer(
    evals, evecs, affine=affine, mask=mask, sphere=sphere, scale=.3)
tensor_slicer_actor.display_extent(
    0, data_shape[0], 0, data_shape[1], 0, data_shape[2])

scene = window.Scene()
scene.add(tensor_slicer_actor)

scene.reset_camera()
scene.reset_clipping_range()

if interactive:
    window.show(scene)

scene.clear()

###############################################################################
# Using ellipsoid actor
# =====================
#

'''
valid_mask = np.abs(evecs).max(axis=(-2, -1)) > 0
indices = np.nonzero(valid_mask)

centers = np.asarray(indices).T

num_centers = centers.shape[0]

dofs_vecs = evecs[indices]
dofs_vals = evals[indices]

colors = [_color_fa(_fa(dofs_vals[i, :]), dofs_vecs[i, ...])
          for i in range(num_centers)]
colors = np.asarray(colors)

tensors = actor.ellipsoid(
    centers, colors=colors, axes=dofs_vecs, lengths=dofs_vals, scales=.6)
scene.add(tensors)

if interactive:
    window.show(scene)

scene.clear()
'''

###############################################################################
# Visualize a larger amount of data
# =================================
# With tensor_slicer is possible to visualize more than one slice using
# display_extent(). Here we can see an example of a region of interest (ROI)
# using a sphere of 100 vertices.

data_shape = roi_evals.shape[:3]
mask = np.ones(data_shape).astype(bool)

tensor_roi = actor.tensor_slicer(
    roi_evals, roi_evecs, affine=affine, mask=mask, sphere=sphere, scale=.3)
tensor_roi.display_extent(
    0, data_shape[0], 0, data_shape[1], 0, data_shape[2])

scene = window.Scene()
scene.add(tensor_roi)

if interactive:
    window.show(scene)

scene.clear()

# We can do it also with a sphere of 200 vertices, but if we try to do it with
# one of 724 the visualization can no longer be rendered. In contrast, we can
# visualize the ROI with the ellipsoid actor without compromising the quality
# of the visualization.

'''
valid_mask = np.abs(roi_evecs).max(axis=(-2, -1)) > 0
indices = np.nonzero(valid_mask)

centers = np.asarray(indices).T

num_centers = centers.shape[0]

dofs_vecs = roi_evecs[indices]
dofs_vals = roi_evals[indices]

colors = [_color_fa(_fa(dofs_vals[i, :]), dofs_vecs[i, ...])
          for i in range(num_centers)]
colors = np.asarray(colors)

tensors = actor.ellipsoid(
    centers, colors=colors, axes=dofs_vecs, lengths=dofs_vals, scales=.6)
scene.add(tensors)

if interactive:
    window.show(scene)

scene.clear()
'''

# In fact, although with a low performance, this actor allows us to visualize
# the whole brain, which contains a much larger amount of data.

'''
valid_mask = np.abs(whole_brain_evecs).max(axis=(-2, -1)) > 0
indices = np.nonzero(valid_mask)

centers = np.asarray(indices).T

num_centers = centers.shape[0]

dofs_vecs = whole_brain_evecs[indices]
dofs_vals = whole_brain_evals[indices]

colors = [_color_fa(_fa(dofs_vals[i, :]), dofs_vecs[i, ...])
          for i in range(num_centers)]
colors = np.asarray(colors)

tensors = actor.ellipsoid(
    centers, colors=colors, axes=dofs_vecs, lengths=dofs_vals, scales=.6)
scene.add(tensors)

if interactive:
    window.show(scene)

scene.clear()
'''
