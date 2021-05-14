from fury.colormap import boys2rgb, orient2rgb
from fury.shaders import attribute_to_actor, load, shader_to_actor
from fury.utils import apply_affine, numpy_to_vtk_colors, numpy_to_vtk_points
from vtk.util import numpy_support


import numpy as np
import vtk


class PeakActor(vtk.vtkActor):
    def __init__(self, directions, indices, values=None, affine=None,
                 colors=None, linewidth=1):
        if affine is not None:
            w_pos = apply_affine(affine, np.asarray(indices).T)

        valid_dirs = directions[indices]

        points_array = []
        centers_array = []
        diffs_array = []
        for idx, center in enumerate(zip(indices[0], indices[1], indices[2])):
            if affine is None:
                xyz = np.array(center)
            else:
                xyz = w_pos[idx, :]
            valid_peaks = np.nonzero(
                np.abs(valid_dirs[idx, :, :]).max(axis=-1) > 0.)[0]
            for pdir in valid_peaks:
                if values is not None:
                    pv = values[center][pdir]
                else:
                    pv = 1.
                point_i = directions[center][pdir] * pv + xyz
                point_e = -directions[center][pdir] * pv + xyz
                diff = point_e - point_i
                points_array.append(point_e)
                points_array.append(point_i)
                centers_array.append(center)
                centers_array.append(center)
                diffs_array.append(diff)
                diffs_array.append(diff)

        points_array = np.asarray(points_array)
        centers_array = np.asarray(centers_array)
        diffs_array = np.asarray(diffs_array)

        vtk_points = numpy_to_vtk_points(points_array)

        vtk_cells = PeakActor._points_to_vtk_cells(points_array)

        peaks_colors_object = PeakActor._peaks_colors_from_points(
            points_array, colors=colors)
        vtk_colors, colors_are_scalars, self.__go = peaks_colors_object

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetLines(vtk_cells)
        poly_data.GetPointData().SetScalars(vtk_colors)

        self.__mapper = vtk.vtkPolyDataMapper()
        self.__mapper.SetInputData(poly_data)
        self.__mapper.ScalarVisibilityOn()
        self.__mapper.SetScalarModeToUsePointFieldData()
        self.__mapper.SelectColorArray('colors')
        self.__mapper.Update()

        self.SetMapper(self.__mapper)

        attribute_to_actor(self, centers_array, 'center')
        attribute_to_actor(self, diffs_array, 'diff')

        vs_dec_code = load('peak_dec.vert')
        vs_impl_code = load('peak_impl.vert')
        fs_dec_code = load('peak_dec.frag')
        fs_impl_code = load('peak_impl.frag')

        shader_to_actor(self, 'vertex', decl_code=vs_dec_code,
                        impl_code=vs_impl_code)
        shader_to_actor(self, 'fragment', decl_code=fs_dec_code)
        shader_to_actor(self, 'fragment', impl_code=fs_impl_code,
                        block='light')

        self.__lw = linewidth
        self.GetProperty().SetLineWidth(self.__lw)

        if self.__go >= 0:
            self.GetProperty().SetOpacity(self.__go)

        self.__min_centers = np.min(indices, axis=1)
        self.__max_centers = np.max(indices, axis=1)

        self.__is_range = True
        self.__low_ranges = self.__min_centers
        self.__high_ranges = self.__max_centers
        self.__cross_section = self.__high_ranges // 2

        self.__mapper.AddObserver(vtk.vtkCommand.UpdateShaderEvent,
                                  self.__display_peaks_vtk_callback)

    @staticmethod
    def _orientation_colors(points, cmap='rgb_standard'):
        """

        Parameters
        ----------
        points : (N, 3) array or ndarray
            points coordinates array.
        cmap : string ('rgb_standard', 'boys_standard')
            colormap.

        Returns
        -------
        colors_list : ndarray
            list of  Kx3 colors. Where K is the number of lines.

        """
        if cmap == 'rgb_standard':
            col_list = [orient2rgb(points[i + 1] - points[i]) for i in range(
                0, len(points), 2)]
        if cmap == 'boys_standard':
            col_list = [boys2rgb(points[i + 1] - points[i]) for i in range(
                0, len(points), 2)]
        return np.asarray(col_list)

    @staticmethod
    def _peaks_colors_from_points(points, colors=None):
        """
        Returns a VTK scalar array containing colors information for each one
        of the peaks according to the policy defined by the parameter colors.

        Parameters
        ----------
        points : (N, 3) array or ndarray
            points coordinates array.
        colors : None or string ('rgb_standard') or tuple (3D or 4D) or
                array/ndarray (N, 3 or 4) or array/ndarray (K, 3 or 4) or
                array/ndarray(N, ) or array/ndarray (K, )
            If None a standard orientation colormap is used for every
            line.
            If one tuple of color is used. Then all streamlines will have the
            same color.
            If an array (N, 3 or 4) is given, where N is equal to the number of
            points. Then every point is colored with a different RGB(A) color.
            If an array (K, 3 or 4) is given, where K is equal to the number of
            lines. Then every line is colored with a different RGB(A) color.
            If an array (N, ) is given, where N is the number of points then
            these are considered as the values to be used by the colormap.
            If an array (K,) is given, where K is the number of lines then
            these are considered as the values to be used by the colormap.

        Returns
        -------
        color_array : vtkDataArray
            vtk scalar array with name 'colors'.
        colors_are_scalars : bool
            indicates whether or not the colors are scalars to be interpreted
            by a colormap.
        global_opacity : float
            returns 1 if the colors array doesn't contain opacity otherwise -1.

        """
        pnts_per_line = 2
        num_pnts = len(points)
        num_lines = num_pnts // pnts_per_line
        colors_are_scalars = False
        global_opacity = 1
        if colors is None or colors == 'rgb_standard':
            # Automatic RGB colors
            # TODO: Add boys support
            colors = np.asarray((0, 0, 0))
            color_array = numpy_to_vtk_colors(
                np.tile(255 * colors, (num_pnts, 1)))
        elif type(colors) is tuple:
            global_opacity = 1 if len(colors) == 3 else -1
            colors = np.asarray(colors)
            color_array = numpy_to_vtk_colors(
                np.tile(255 * colors, (num_pnts, 1)))
        else:
            colors = np.asarray(colors)
            if colors.dtype == object:  # colors is a list of colors
                color_array = numpy_to_vtk_colors(255 * np.vstack(colors))
            else:
                if len(colors) == num_lines:
                    pnts_colors = np.repeat(colors, pnts_per_line, axis=0)
                    if colors.ndim == 1:  # Scalar per line
                        color_array = numpy_support.numpy_to_vtk(pnts_colors,
                                                                 deep=True)
                        colors_are_scalars = True
                    elif colors.ndim == 2:  # RGB(A) color per line
                        global_opacity = 1 if colors.shape[1] == 3 else -1
                        color_array = numpy_to_vtk_colors(255 * pnts_colors)
                elif len(colors) == num_pnts:
                    if colors.ndim == 1:  # Scalar per point
                        color_array = numpy_support.numpy_to_vtk(colors,
                                                                 deep=True)
                        colors_are_scalars = True
                    elif colors.ndim == 2:  # RGB(A) color per point
                        global_opacity = 1 if colors.shape[1] == 3 else -1
                        color_array = numpy_to_vtk_colors(255 * colors)
        color_array.SetName('colors')
        return color_array, colors_are_scalars, global_opacity

    @staticmethod
    def _points_to_vtk_cells(points):
        """
        Returns the VTK cell array for the peaks given the set of points
        coordinates.

        Parameters
        ----------
        points : (N, 3) array or ndarray
            points coordinates array.

        Returns
        -------
        cell_array : vtkCellArray
            connectivity + offset information.

        """
        pnts_per_line = 2
        num_pnts = len(points)
        num_cells = num_pnts // pnts_per_line

        cell_array = vtk.vtkCellArray()

        if vtk.vtkVersion.GetVTKMajorVersion() >= 9:
            """
            Connectivity is an array that contains the indices of the points 
            that need to be connected in the visualization. The indices start 
            from 0.
            """
            connectivity = np.array(list(range(0, num_pnts)), dtype=int)
            """
            Offset is an array that contains the indices of the first point of 
            each line. The indices start from 0 and given the known geometry of 
            this actor the creation of this array requires a 2 points padding 
            between indices.
            """
            offset = np.array(list(range(0, num_pnts + 1, pnts_per_line)),
                              dtype=int)

            vtk_array_type = numpy_support.get_vtk_array_type(
                connectivity.dtype)
            cell_array.SetData(
                numpy_support.numpy_to_vtk(offset, deep=True,
                                           array_type=vtk_array_type),
                numpy_support.numpy_to_vtk(connectivity, deep=True,
                                           array_type=vtk_array_type))
        else:
            connectivity = []
            i_pos = 0

            while i_pos < num_pnts:
                end_pos = i_pos + pnts_per_line
                """
                In old versions of VTK (<9.0) the connectivity array should 
                include the length of each line and immediately after the 
                indices of the points in each line.
                """
                connectivity += [pnts_per_line]
                connectivity += list(range(i_pos, end_pos))
                i_pos = end_pos

            connectivity = np.array(connectivity)
            cell_array.GetData().DeepCopy(
                numpy_support.numpy_to_vtk(connectivity))

        cell_array.SetNumberOfCells(num_cells)
        return cell_array

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def __display_peaks_vtk_callback(self, caller, event, calldata=None):
        if calldata is not None:
            calldata.SetUniformi('isRange', self.__is_range)
            calldata.SetUniform3f('highRanges', self.__high_ranges)
            calldata.SetUniform3f('lowRanges', self.__low_ranges)
            calldata.SetUniform3f('crossSection', self.__cross_section)

    def display_cross_section(self, x, y, z):
        if self.__is_range:
            self.__is_range = False
        self.__cross_section = [x, y, z]

    def display_extent(self, x1, x2, y1, y2, z1, z2):
        if not self.__is_range:
            self.__is_range = True
        self.__low_ranges = [x1, y1, z1]
        self.__high_ranges = [x2, y2, z2]

    @property
    def cross_section(self):
        return self.__cross_section

    @property
    def global_opacity(self):
        return self.__go

    @global_opacity.setter
    def global_opacity(self, opacity):
        self.__go = opacity
        self.GetProperty().SetOpacity(self.__go)

    @property
    def high_ranges(self):
        return self.__high_ranges

    @property
    def is_range(self):
        return self.__is_range

    @is_range.setter
    def is_range(self, is_range):
        self.__is_range = is_range

    @property
    def low_ranges(self):
        return self.__low_ranges

    @property
    def linewidth(self):
        return self.__lw

    @linewidth.setter
    def linewidth(self, linewidth):
        self.__lw = linewidth
        self.GetProperty().SetLineWidth(self.__lw)

    @property
    def max_centers(self):
        return self.__max_centers

    @property
    def min_centers(self):
        return self.__min_centers
