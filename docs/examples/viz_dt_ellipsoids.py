"""
===============================================================================
Display Tensor Ellipsoids for DTI using tensor_slicer vs ellipsoid actor
===============================================================================
This tutorial is intended to show two ways of displaying diffusion tensor
ellipsoids for DTI visualization. The first is using the usual tensor_slicer
that allows us to slice many tensors as ellipsoids. The second is the generic
ellipsoid actor that can be used to display different amount of ellipsoids.

We start by importing the necessary modules:
"""

import numpy as np

from dipy.io.image import load_nifti

from fury import window, actor
from fury.actor import _fa, _color_fa
from fury.data import fetch_viz_dmri, read_viz_dmri
from fury.primitive import prim_sphere

###############################################################################
# Now, we fetch and load the data needed to display the Diffusion Tensor
# Images.

fetch_viz_dmri()

###############################################################################
# The tensor ellipsoids are expressed as eigenvalues and eigenvectors which are
# the decomposition of the diffusion tensor that describes the water diffusion
# within a voxel.

slice_evecs, _ = load_nifti(read_viz_dmri('slice_evecs.nii.gz'))
slice_evals, _ = load_nifti(read_viz_dmri('slice_evals.nii.gz'))
roi_evecs, _ = load_nifti(read_viz_dmri('roi_evecs.nii.gz'))
roi_evals, _ = load_nifti(read_viz_dmri('roi_evals.nii.gz'))
whole_brain_evecs, _ = load_nifti(read_viz_dmri('whole_brain_evecs.nii.gz'))
whole_brain_evals, _ = load_nifti(read_viz_dmri('whole_brain_evals.nii.gz'))

###############################################################################
# Using tensor_slicer actor
# =========================
# First we must define the 3 parameters needed to use the ``tensor_slicer``
# actor, which correspond to the eigenvalues, the eigenvectors, and the sphere.
# For the sphere we use ``prim_sphere`` which provide vertices and triangles of
# the spheres. These are labeled as 'repulsionN' with N been the number of
# vertices that made up the sphere, which have a standard number of 100, 200,
# and 724 vertices.

vertices, faces = prim_sphere('repulsion100', True)

###############################################################################
# As we need to provide a sphere object we create a class Sphere to which we
# assign the values obtained from vertices and faces.

class Sphere:
    vertices = None
    faces = None


sphere = Sphere()
sphere.vertices = vertices
sphere.faces = faces

###############################################################################
# Now we are ready to create the ``tensor_slicer`` actor with the values of a
# brain slice. We also define the scale so that the tensors are not so large
# and overlap each other.

tensor_slice = actor.tensor_slicer(evals=slice_evals, evecs=slice_evecs,
                                   sphere=sphere, scale=.3)

###############################################################################
# Next, we set up a new scene to add and visualize the tensor ellipsoids
# created.

scene = window.Scene()
scene.add(tensor_slice)

# Enables/disables interactive visualization
interactive = True

if interactive:
    window.show(scene)

window.record(scene, size=(600, 600), out_path='tensor_slice_100.png')
window.record(scene)

###############################################################################
# To render the same tensor slice using a different sphere we redefine the
# vertices and faces of the sphere using prim_sphere with other sphere
# specification, let's say 'repulsion200' and 'repulsion724'.
#
# If we zoom in at the scene to see with detail the tensor ellipsoids displayed
# with the different spheres, we get the following results.
#

###############################################################################
# We clear the scene for the next visualization.

scene.clear()

###############################################################################
# Using ellipsoid actor
# =====================
# In order to use the ellipsoid actor to display the same tensor slice we need
# to set additional parameters. First, we define the centers which corresponds
# to ellipsoids positions.

valid_mask = np.abs(slice_evecs).max(axis=(-2, -1)) > 0
indices = np.nonzero(valid_mask)
centers = np.asarray(indices).T

###############################################################################
# Second, we need to pass the data of the axes and lengths of the ellipsoid as
# a ndarray, so it is necessary to rearrange the data of the eigenvectors and
# eigenvalues.

evecs = slice_evecs[indices]
evals = slice_evals[indices]

###############################################################################
# Third, we need to define the colors of the ellipsoids following the default
# coloring in ``tensor_slicer`` that is using ``_color_fa`` that is a way to
# map colors to each tensor based on the fractional anisotropy (FA) of each
# diffusion tensor.

colors = _color_fa(_fa(evals), evecs)

###############################################################################
# Now, we can use the ``ellipsoid`` actor to create the tensor ellipsoids as
# follows.

tensors = actor.ellipsoid(centers=centers, colors=colors, axes=evecs,
                          lengths=evals, scales=.6)
scene.add(tensors)

if interactive:
    window.show(scene)

window.record(scene, size=(600, 600), out_path='tensor_slice_sdf.png')
window.record(scene)

scene.clear()

###############################################################################
# Thus, one can see that the same result is obtained, however there is a
# difference in the visual quality and this is because the ellipsoid actor uses
# raymarching technique, so the objects that are generated are smoother since
# they are not made with polygons but defined by an SDF function. Next we can
# see in more detail the tensor ellipsoids generated.
#

###############################################################################
# Visualize a larger amount of data
# =================================
# With ``tensor_slicer`` is possible to visualize more than one slice using
# ``display_extent()``. Here we can see an example of a region of interest
# (ROI) using a sphere of 100 vertices.

tensor_roi = actor.tensor_slicer(evals=roi_evals, evecs=roi_evecs,
                                 sphere=sphere, scale=.3)

data_shape = roi_evals.shape[:3]
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
