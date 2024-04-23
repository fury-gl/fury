# -*- coding: utf-8 -*-
import numpy as np

from fury.colormap import create_colormap
from fury.lib import Actor, PolyData, PolyDataMapper
from fury.utils import (
    apply_affine,
    set_polydata_colors,
    set_polydata_triangles,
    set_polydata_vertices,
)


class OdfSlicerActor(Actor):
    """VTK actor for visualizing slices of ODF field.

    Parameters
    ----------
    odfs : ndarray
        SF or SH coefficients 2-dimensional array.
    vertices: ndarray
        The sphere vertices used for SH to SF projection.
    faces: ndarray
        Indices of sphere vertices forming triangles. Should be
        ordered clockwise (see fury.utils.fix_winding_order).
    indices: tuple
        Indices given in tuple(x_indices, y_indices, z_indices)
        format for mapping 2D ODF array to 3D voxel grid.
    scale : float
        Multiplicative factor to apply to ODF amplitudes.
    norm : bool
        Normalize SF amplitudes so that the maximum
        ODF amplitude per voxel along a direction is 1.
    radial_scale : bool
        Scale sphere points by ODF values.
    global_cm : bool
        If True the colormap will be applied in all ODFs. If False
        it will be applied individually at each voxel.
    colormap : None or str
        The name of the colormap to use. Matplotlib colormaps are supported
        (e.g., 'inferno'). If None then a RGB colormap is used.
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque).
    affine : array
        optional 4x4 transformation array from native
        coordinates to world coordinates.
    B : ndarray (n_coeffs, n_vertices)
        Optional SH to SF matrix for projecting `odfs` given in SH
        coefficients on the `sphere`. If None, then the input is assumed
        to be expressed in SF coefficients.

    """

    def __init__(
        self,
        odfs,
        vertices,
        faces,
        indices,
        scale,
        norm,
        radial_scale,
        shape,
        global_cm,
        colormap,
        opacity,
        affine=None,
        B=None,
    ):
        self.vertices = vertices
        self.faces = faces
        self.odfs = odfs
        self.indices = indices
        self.B = B
        self.radial_scale = radial_scale
        self.colormap = colormap
        self.grid_shape = shape
        self.global_cm = global_cm

        # declare a mask to be instantiated in slice_along_axis
        self.mask = None

        # If a B matrix is given, odfs are expected to
        # be in SH basis coefficients.
        if self.B is not None:
            # In that case, we need to save our normalisation and scale
            # to apply them after conversion from SH to SF.
            self.norm = norm
            self.scale = scale
        else:
            # If our input is in SF coefficients, we can normalise and
            # scale it only once, here.
            if norm:
                self.odfs /= np.abs(self.odfs).max(axis=-1, keepdims=True)
            self.odfs *= scale

        # Compute world coordinates of an affine is supplied
        self.affine = affine
        if self.affine is not None:
            self.w_verts = self.vertices.dot(affine[:3, :3])
            self.w_pos = apply_affine(affine, np.asarray(self.indices).T)

        # Initialize mapper and slice to the
        # middle of the volume along Z axis
        self.mapper = PolyDataMapper()
        self.SetMapper(self.mapper)
        self.slice_along_axis(self.grid_shape[-1] // 2)
        self.set_opacity(opacity)

    def set_opacity(self, opacity):
        """Set opacity value of ODFs to display.
        """
        self.GetProperty().SetOpacity(opacity)

    def display_extent(self, x1, x2, y1, y2, z1, z2):
        """Set visible volume from x1 (inclusive) to x2 (inclusive),
        y1 (inclusive) to y2 (inclusive), z1 (inclusive) to z2
        (inclusive).
        """
        mask = np.zeros(self.grid_shape, dtype=bool)
        mask[x1: x2 + 1, y1: y2 + 1, z1: z2 + 1] = True
        self.mask = mask

        self._update_mapper()

    def slice_along_axis(self, slice_index, axis='zaxis'):
        """Slice ODF field at given `slice_index` along axis
        in ['xaxis', 'yaxis', zaxis'].
        """
        if axis == 'xaxis':
            self.display_extent(
                slice_index,
                slice_index,
                0,
                self.grid_shape[1] - 1,
                0,
                self.grid_shape[2] - 1,
            )
        elif axis == 'yaxis':
            self.display_extent(
                0,
                self.grid_shape[0] - 1,
                slice_index,
                slice_index,
                0,
                self.grid_shape[2] - 1,
            )
        elif axis == 'zaxis':
            self.display_extent(
                0,
                self.grid_shape[0] - 1,
                0,
                self.grid_shape[1] - 1,
                slice_index,
                slice_index,
            )
        else:
            raise ValueError('Invalid axis name {0}.'.format(axis))

    def display(self, x=None, y=None, z=None):
        """Display a slice along x, y, or z axis.
        """
        if x is None and y is None and z is None:
            self.slice_along_axis(self.grid_shape[2] // 2)
        elif x is not None:
            self.slice_along_axis(x, 'xaxis')
        elif y is not None:
            self.slice_along_axis(y, 'yaxis')
        elif z is not None:
            self.slice_along_axis(z, 'zaxis')

    def update_sphere(self, vertices, faces, B):
        """Dynamically change the sphere used for SH to SF projection.
        """
        if self.B is None:
            raise ValueError(
                "Can't update sphere when using "
                'SF coefficients.'
                )
        self.vertices = vertices
        if self.affine is not None:
            self.w_verts = self.vertices.dot(self.affine[:3, :3])
        self.faces = faces
        self.B = B

        # draw ODFs with new sphere
        self._update_mapper()

    def _update_mapper(self):
        """Map vtkPolyData to the actor.
        """
        polydata = PolyData()

        offsets = self._get_odf_offsets(self.mask)
        if len(offsets) == 0:
            self.mapper.SetInputData(polydata)
            return None

        sph_dirs = self._get_sphere_directions()
        sf = self._get_sf(self.mask)

        all_vertices = self._get_all_vertices(offsets, sph_dirs, sf)
        all_faces = self._get_all_faces(len(offsets), len(sph_dirs))
        all_colors = self._generate_color_for_vertices(sf)

        # TODO: There is a lot of deep copy here.
        # Optimize (see viz_network.py example).
        set_polydata_triangles(polydata, all_faces)
        set_polydata_vertices(polydata, all_vertices)
        set_polydata_colors(polydata, all_colors)

        self.mapper.SetInputData(polydata)

    def _get_odf_offsets(self, mask):
        """Get the position of non-zero voxels inside `mask`.
        """
        if self.affine is not None:
            return self.w_pos[mask[self.indices]]
        return np.asarray(self.indices).T[mask[self.indices]]

    def _get_sphere_directions(self):
        """Get the sphere directions onto which is projected the signal.
        """
        if self.affine is not None:
            return self.w_verts
        return self.vertices

    def _get_sf(self, mask):
        """Get SF coefficients inside `mask`.
        """
        # when odfs are expressed in SH coefficients
        if self.B is not None:
            sf = self.odfs[mask[self.indices]].dot(self.B)
            # normalisation and scaling is done on SF coefficients
            if self.norm:
                sf /= np.abs(sf).max(axis=-1, keepdims=True)
            return sf * self.scale
        # when odfs are in SF coefficients, the normalisation and scaling
        # are done during initialisation. We simply return them:
        return self.odfs[mask[self.indices]]

    def _get_all_vertices(self, offsets, sph_dirs, sf):
        """Get array of all the vertices of the ODFs to display.
        """
        if self.radial_scale:
            # apply SF amplitudes to all sphere
            # directions and offset each voxel
            return (
                np.tile(sph_dirs, (len(offsets), 1)) * sf.reshape(-1, 1)
                + np.repeat(offsets, len(sph_dirs), axis=0)
            )
        # return scaled spheres offsetted by `offsets`
        return np.tile(sph_dirs, (len(offsets), 1)) * self.scale + np.repeat(
            offsets, len(sph_dirs), axis=0
        )

    def _get_all_faces(self, nb_odfs, nb_dirs):
        """Get array of all the faces of the ODFs to display.
        """
        return np.tile(self.faces, (nb_odfs, 1)) + np.repeat(
            np.arange(nb_odfs) * nb_dirs, len(self.faces)
        ).reshape(-1, 1)

    def _generate_color_for_vertices(self, sf):
        """Get array of all vertices colors.
        """
        if self.global_cm:
            if self.colormap is None:
                raise IOError('if global_cm=True, colormap must be defined.')
            else:
                all_colors = create_colormap(sf.ravel(), self.colormap) * 255
        elif self.colormap is not None:
            if isinstance(self.colormap, str):
                # Map ODFs values [min, max] to [0, 1] for each ODF
                range_sf = sf.max(axis=-1) - sf.min(axis=-1)
                rescaled = sf - sf.min(axis=-1, keepdims=True)
                rescaled[range_sf > 0] /= range_sf[range_sf > 0][..., None]
                all_colors = create_colormap(rescaled.ravel(), self.colormap) \
                    * 255
            else:
                all_colors = np.tile(
                    np.array(self.colormap).reshape(1, 3),
                    (sf.shape[0] * sf.shape[1], 1),
                )
        else:
            all_colors = np.tile(np.abs(self.vertices) * 255, (len(sf), 1))
        return all_colors.astype(np.uint8)
