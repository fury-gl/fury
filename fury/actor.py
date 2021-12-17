"""Module that provide actors to render."""

import warnings
import os.path as op
from functools import partial

import numpy as np

from fury.shaders import (load, shader_to_actor, attribute_to_actor,
                          add_shader_callback, replace_shader_in_actor)
from fury import layout
from fury.actors.odf_slicer import OdfSlicerActor
from fury.actors.peak import PeakActor
from fury.colormap import colormap_lookup_table
from fury.deprecator import deprecated_params
from fury.io import load_image
from fury.lib import (numpy_support, Transform, ImageData, PolyData, Matrix4x4,
                      ImageReslice, ImageActor, CellPicker, OutlineFilter,
                      Actor, PolyDataMapper, LookupTable, ImageMapToColors,
                      Points, CleanPolyData, LoopSubdivisionFilter, TubeFilter,
                      ButterflySubdivisionFilter, ContourFilter, SplineFilter,
                      PolyDataNormals, Assembly, LODActor, VTK_UNSIGNED_CHAR,
                      PolyDataMapper2D, ScalarBarActor, PolyVertex, CellArray,
                      UnstructuredGrid, DataSetMapper, ConeSource, ArrowSource,
                      SphereSource, CylinderSource, TexturedSphereSource,
                      Texture, FloatArray, VTK_TEXT_LEFT, VTK_TEXT_RIGHT,
                      VTK_TEXT_BOTTOM, VTK_TEXT_TOP, VTK_TEXT_CENTERED,
                      TexturedActor2D, TextureMapToPlane, TextActor3D,
                      Follower, VectorText)
import fury.primitive as fp
from fury.utils import (lines_to_vtk_polydata, set_input, apply_affine,
                        set_polydata_vertices, set_polydata_triangles,
                        shallow_copy, rgb_to_vtk, numpy_to_vtk_matrix,
                        repeat_sources, get_actor_from_primitive,
                        fix_winding_order, numpy_to_vtk_colors)


def slicer(data, affine=None, value_range=None, opacity=1.,
           lookup_colormap=None, interpolation='linear', picking_tol=0.025):
    """Cut 3D scalar or rgb volumes into 2D images.

    Parameters
    ----------
    data : array, shape (X, Y, Z) or (X, Y, Z, 3)
        A grayscale or rgb 4D volume as a numpy array. If rgb then values
        expected on the range [0, 255].
    affine : array, shape (4, 4)
        Grid to space (usually RAS 1mm) transformation matrix. Default is None.
        If None then the identity matrix is used.
    value_range : None or tuple (2,)
        If None then the values will be interpolated from (data.min(),
        data.max()) to (0, 255). Otherwise from (value_range[0],
        value_range[1]) to (0, 255).
    opacity : float, optional
        Opacity of 0 means completely transparent and 1 completely visible.
    lookup_colormap : vtkLookupTable, optional
        If None (default) then a grayscale map is created.
    interpolation : string, optional
        If 'linear' (default) then linear interpolation is used on the final
        texture mapping. If 'nearest' then nearest neighbor interpolation is
        used on the final texture mapping.
    picking_tol : float, optional
        The tolerance for the vtkCellPicker, specified as a fraction of
        rendering window size.

    Returns
    -------
    image_actor : ImageActor
        An object that is capable of displaying different parts of the volume
        as slices. The key method of this object is ``display_extent`` where
        one can input grid coordinates and display the slice in space (or grid)
        coordinates as calculated by the affine parameter.

    """
    if value_range is None:
        value_range = (data.min(), data.max())

    if data.ndim != 3:
        if data.ndim == 4:
            if data.shape[3] != 3:
                raise ValueError('Only RGB 3D arrays are currently supported.')
            else:
                nb_components = 3
        else:
            raise ValueError('Only 3D arrays are currently supported.')
    else:
        nb_components = 1

    vol = data

    im = ImageData()
    I, J, K = vol.shape[:3]
    im.SetDimensions(I, J, K)
    # for now setting up for 1x1x1 but transformation comes later.
    voxsz = (1., 1., 1.)
    # im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])

    vtk_type = numpy_support.get_vtk_array_type(vol.dtype)
    im.AllocateScalars(vtk_type, nb_components)
    # im.AllocateScalars(VTK_UNSIGNED_CHAR, nb_components)

    # copy data
    # what I do below is the same as what is
    # commented here but much faster
    # for index in ndindex(vol.shape):
    #     i, j, k = index
    #     im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])
    vol = np.swapaxes(vol, 0, 2)
    vol = np.ascontiguousarray(vol)

    if nb_components == 1:
        vol = vol.ravel()
    else:
        vol = np.reshape(vol, [np.prod(vol.shape[:3]), vol.shape[3]])

    uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
    im.GetPointData().SetScalars(uchar_array)

    if affine is None:
        affine = np.eye(4)

    # Set the transform (identity if none given)
    transform = Transform()
    transform_matrix = Matrix4x4()
    transform_matrix.DeepCopy((
        affine[0][0], affine[0][1], affine[0][2], affine[0][3],
        affine[1][0], affine[1][1], affine[1][2], affine[1][3],
        affine[2][0], affine[2][1], affine[2][2], affine[2][3],
        affine[3][0], affine[3][1], affine[3][2], affine[3][3]))
    transform.SetMatrix(transform_matrix)
    transform.Inverse()

    # Set the reslicing
    image_resliced = ImageReslice()
    set_input(image_resliced, im)
    image_resliced.SetResliceTransform(transform)
    image_resliced.AutoCropOutputOn()

    # Adding this will allow to support anisotropic voxels
    # and also gives the opportunity to slice per voxel coordinates
    RZS = affine[:3, :3]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    image_resliced.SetOutputSpacing(*zooms)

    image_resliced.SetInterpolationModeToLinear()
    image_resliced.Update()

    vtk_resliced_data = image_resliced.GetOutput()

    ex1, ex2, ey1, ey2, ez1, ez2 = vtk_resliced_data.GetExtent()

    resliced = numpy_support.vtk_to_numpy(
        vtk_resliced_data.GetPointData().GetScalars())

    # swap axes here
    if data.ndim == 4:
        if data.shape[-1] == 3:
            resliced = resliced.reshape(ez2 + 1, ey2 + 1, ex2 + 1, 3)
    if data.ndim == 3:
        resliced = resliced.reshape(ez2 + 1, ey2 + 1, ex2 + 1)

    class ImActor(ImageActor):
        def __init__(self):
            self.picker = CellPicker()
            self.output = None
            self.shape = None
            self.outline_actor = None

        def input_connection(self, output):

            # outline only
            outline = OutlineFilter()
            outline.SetInputData(vtk_resliced_data)
            outline_mapper = PolyDataMapper()
            outline_mapper.SetInputConnection(outline.GetOutputPort())
            self.outline_actor = Actor()
            self.outline_actor.SetMapper(outline_mapper)
            self.outline_actor.GetProperty().SetColor(1, 0.5, 0)
            self.outline_actor.GetProperty().SetLineWidth(5)
            self.outline_actor.GetProperty().SetRenderLinesAsTubes(True)
            # crucial
            self.GetMapper().SetInputConnection(output.GetOutputPort())
            self.output = output
            self.shape = (ex2 + 1, ey2 + 1, ez2 + 1)

        def display_extent(self, x1, x2, y1, y2, z1, z2):
            self.SetDisplayExtent(x1, x2, y1, y2, z1, z2)
            self.Update()
            # bounds = self.GetBounds()
            # xmin, xmax, ymin, ymax, zmin, zmax = bounds
            # line = np.array([[xmin, ymin, zmin]])
            # self.outline_actor = actor.line()

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(ex1, ex2, ey1, ey2, ez2//2, ez2//2)
            if x is not None:
                self.display_extent(x, x, ey1, ey2, ez1, ez2)
            if y is not None:
                self.display_extent(ex1, ex2, y, y, ez1, ez2)
            if z is not None:
                self.display_extent(ex1, ex2, ey1, ey2, z, z)

        def resliced_array(self):
            """Return resliced array as numpy array."""
            resliced = numpy_support.vtk_to_numpy(
                vtk_resliced_data.GetPointData().GetScalars())

            # swap axes here
            if data.ndim == 4:
                if data.shape[-1] == 3:
                    resliced = resliced.reshape(ez2 + 1, ey2 + 1, ex2 + 1, 3)
            if data.ndim == 3:
                resliced = resliced.reshape(ez2 + 1, ey2 + 1, ex2 + 1)
            resliced = np.swapaxes(resliced, 0, 2)
            resliced = np.ascontiguousarray(resliced)
            return resliced

        def opacity(self, value):
            self.GetProperty().SetOpacity(value)

        def tolerance(self, value):
            self.picker.SetTolerance(value)

        def copy(self):
            im_actor = ImActor()
            im_actor.input_connection(self.output)
            im_actor.SetDisplayExtent(*self.GetDisplayExtent())
            im_actor.opacity(self.GetOpacity())
            im_actor.tolerance(self.picker.GetTolerance())
            if interpolation == 'nearest':
                im_actor.SetInterpolate(False)
            else:
                im_actor.SetInterpolate(True)
                im_actor.GetMapper().BorderOn()
            return im_actor

        def shallow_copy(self):
            # TODO rename copy to shallow_copy
            self.copy()

    r1, r2 = value_range

    image_actor = ImActor()
    if nb_components == 1:
        lut = lookup_colormap
        if lookup_colormap is None:
            # Create a black/white lookup table.
            lut = colormap_lookup_table((r1, r2), (0, 0), (0, 0), (0, 1))

        plane_colors = ImageMapToColors()
        plane_colors.SetOutputFormatToRGB()
        plane_colors.SetLookupTable(lut)
        plane_colors.SetInputConnection(image_resliced.GetOutputPort())
        plane_colors.Update()
        image_actor.input_connection(plane_colors)
    else:
        image_actor.input_connection(image_resliced)
    image_actor.display()
    image_actor.opacity(opacity)
    image_actor.tolerance(picking_tol)

    if interpolation == 'nearest':
        image_actor.SetInterpolate(False)
    else:
        image_actor.SetInterpolate(True)

    image_actor.GetMapper().BorderOn()

    return image_actor


def surface(vertices, faces=None, colors=None, smooth=None, subdivision=3):
    """Generate a surface actor from an array of vertices.

    The color and smoothness of the surface can be customized by specifying
    the type of subdivision algorithm and the number of subdivisions.

    Parameters
    ----------
    vertices : array, shape (X, Y, Z)
        The point cloud defining the surface.
    faces : array
        An array of precomputed triangulation for the point cloud.
        It is an optional parameter, it is computed locally if None
    colors : (N, 3) array
        Specifies the colors associated with each vertex in the
        vertices array. Range should be 0 to 1.
        Optional parameter, if not passed, all vertices
        are colored white
    smooth : string - "loop" or "butterfly"
        Defines the type of subdivision to be used
        for smoothing the surface
    subdivision : integer, default = 3
        Defines the number of subdivisions to do for
        each triangulation of the point cloud.
        The higher the value, smoother the surface
        but at the cost of higher computation

    Returns
    -------
    surface_actor : vtkActor
        A vtkActor visualizing the final surface
        computed from the point cloud is returned.

    """
    from scipy.spatial import Delaunay
    points = Points()
    points.SetData(numpy_support.numpy_to_vtk(vertices))
    triangle_poly_data = PolyData()
    triangle_poly_data.SetPoints(points)

    if colors is not None:
        triangle_poly_data.GetPointData().\
            SetScalars(numpy_to_vtk_colors(255 * colors))

    if faces is None:
        tri = Delaunay(vertices[:, [0, 1]])
        faces = np.array(tri.simplices, dtype='i8')

    set_polydata_triangles(triangle_poly_data, faces)

    clean_poly_data = CleanPolyData()
    clean_poly_data.SetInputData(triangle_poly_data)

    mapper = PolyDataMapper()
    surface_actor = Actor()

    if smooth is None:
        mapper.SetInputData(triangle_poly_data)
        surface_actor.SetMapper(mapper)

    elif smooth == "loop":
        smooth_loop = LoopSubdivisionFilter()
        smooth_loop.SetNumberOfSubdivisions(subdivision)
        smooth_loop.SetInputConnection(clean_poly_data.GetOutputPort())
        mapper.SetInputConnection(smooth_loop.GetOutputPort())
        surface_actor.SetMapper(mapper)

    elif smooth == "butterfly":
        smooth_butterfly = ButterflySubdivisionFilter()
        smooth_butterfly.SetNumberOfSubdivisions(subdivision)
        smooth_butterfly.SetInputConnection(clean_poly_data.GetOutputPort())
        mapper.SetInputConnection(smooth_butterfly.GetOutputPort())
        surface_actor.SetMapper(mapper)

    return surface_actor


def contour_from_roi(data, affine=None,
                     color=np.array([1, 0, 0]), opacity=1):
    """Generate surface actor from a binary ROI.

    The color and opacity of the surface can be customized.

    Parameters
    ----------
    data : array, shape (X, Y, Z)
        An ROI file that will be binarized and displayed.
    affine : array, shape (4, 4)
        Grid to space (usually RAS 1mm) transformation matrix. Default is None.
        If None then the identity matrix is used.
    color : (1, 3) ndarray
        RGB values in [0,1].
    opacity : float
        Opacity of surface between 0 and 1.

    Returns
    -------
    contour_assembly : vtkAssembly
        ROI surface object displayed in space
        coordinates as calculated by the affine parameter.

    """
    if data.ndim != 3:
        raise ValueError('Only 3D arrays are currently supported.')

    nb_components = 1

    data = (data > 0) * 1
    vol = np.interp(data, xp=[data.min(), data.max()], fp=[0, 255])
    vol = vol.astype('uint8')

    im = ImageData()
    di, dj, dk = vol.shape[:3]
    im.SetDimensions(di, dj, dk)
    voxsz = (1., 1., 1.)
    # im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
    im.AllocateScalars(VTK_UNSIGNED_CHAR, nb_components)

    # copy data
    vol = np.swapaxes(vol, 0, 2)
    vol = np.ascontiguousarray(vol)

    vol = vol.ravel()

    uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
    im.GetPointData().SetScalars(uchar_array)

    if affine is None:
        affine = np.eye(4)

    # Set the transform (identity if none given)
    transform = Transform()
    transform_matrix = Matrix4x4()
    transform_matrix.DeepCopy((
        affine[0][0], affine[0][1], affine[0][2], affine[0][3],
        affine[1][0], affine[1][1], affine[1][2], affine[1][3],
        affine[2][0], affine[2][1], affine[2][2], affine[2][3],
        affine[3][0], affine[3][1], affine[3][2], affine[3][3]))
    transform.SetMatrix(transform_matrix)
    transform.Inverse()

    # Set the reslicing
    image_resliced = ImageReslice()
    set_input(image_resliced, im)
    image_resliced.SetResliceTransform(transform)
    image_resliced.AutoCropOutputOn()

    # Adding this will allow to support anisotropic voxels
    # and also gives the opportunity to slice per voxel coordinates

    rzs = affine[:3, :3]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    image_resliced.SetOutputSpacing(*zooms)

    image_resliced.SetInterpolationModeToLinear()
    image_resliced.Update()

    skin_extractor = ContourFilter()
    skin_extractor.SetInputData(image_resliced.GetOutput())

    skin_extractor.SetValue(0, 1)
    skin_normals = PolyDataNormals()
    skin_normals.SetInputConnection(skin_extractor.GetOutputPort())
    skin_normals.SetFeatureAngle(60.0)

    skin_mapper = PolyDataMapper()
    skin_mapper.SetInputConnection(skin_normals.GetOutputPort())
    skin_mapper.ScalarVisibilityOff()

    skin_actor = Actor()

    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetColor(color[0], color[1], color[2])
    skin_actor.GetProperty().SetOpacity(opacity)

    return skin_actor


def contour_from_label(data, affine=None, color=None):
    """Generate surface actor from a labeled Array.

    The color and opacity of individual surfaces can be customized.

    Parameters
    ----------
    data : array, shape (X, Y, Z)
        A labeled array file that will be binarized and displayed.
    affine : array, shape (4, 4)
        Grid to space (usually RAS 1mm) transformation matrix. Default is None.
        If None then the identity matrix is used.
    color : (N, 3) or (N, 4) ndarray
        RGB/RGBA values in [0,1]. Default is None.
        If None then random colors are used.
        Alpha channel is set to 1 by default.

    Returns
    -------
    contour_assembly : vtkAssembly
        Array surface object displayed in space
        coordinates as calculated by the affine parameter
        in the order of their roi ids.

    """
    unique_roi_id = np.delete(np.unique(data), 0)

    nb_surfaces = len(unique_roi_id)

    unique_roi_surfaces = Assembly()

    if color is None:
        color = np.random.rand(nb_surfaces, 3)
    elif color.shape != (nb_surfaces, 3) and color.shape != (nb_surfaces, 4):
        raise ValueError("Incorrect color array shape")

    if color.shape == (nb_surfaces, 4):
        opacity = color[:, -1]
        color = color[:, :-1]
    else:
        opacity = np.ones((nb_surfaces, 1)).astype(float)

    for i, roi_id in enumerate(unique_roi_id):
        roi_data = np.isin(data, roi_id).astype(int)
        roi_surface = contour_from_roi(roi_data, affine,
                                       color=color[i],
                                       opacity=opacity[i])
        unique_roi_surfaces.AddPart(roi_surface)

    return unique_roi_surfaces


def streamtube(lines, colors=None, opacity=1, linewidth=0.1, tube_sides=9,
               lod=True, lod_points=10 ** 4, lod_points_size=3,
               spline_subdiv=None, lookup_colormap=None):
    """Use streamtubes to visualize polylines.

    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays

    colors : array (N, 3), list of arrays, tuple (3,), array (K,)
        If None or False, a standard orientation colormap is used for every
        line.
        If one tuple of color is used. Then all streamlines will have the same
        colour.
        If an array (N, 3) is given, where N is equal to the number of lines.
        Then every line is coloured with a different RGB color.
        If a list of RGB arrays is given then every point of every line takes
        a different color.
        If an array (K, 3) is given, where K is the number of points of all
        lines then every point is colored with a different RGB color.
        If an array (K,) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (L,) is given, where L is the number of streamlines then
        these are considered as the values to be used by the colormap per
        streamline.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.

    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque). Default is 1.
    linewidth : float, optional
        Default is 0.01.
    tube_sides : int, optional
        Default is 9.
    lod : bool, optional
        Use vtkLODActor(level of detail) rather than vtkActor. Default is True.
        Level of detail actors do not render the full geometry when the
        frame rate is low.
    lod_points : int, optional
        Number of points to be used when LOD is in effect. Default is 10000.
    lod_points_size : int, optional
        Size of points when lod is in effect. Default is 3.
    spline_subdiv : int, optional
        Number of splines subdivision to smooth streamtubes. Default is None.
    lookup_colormap : vtkLookupTable, optional
        Add a default lookup table to the colormap. Default is None which calls
        :func:`fury.actor.colormap_lookup_table`.

    Examples
    --------
    >>> import numpy as np
    >>> from fury import actor, window
    >>> scene = window.Scene()
    >>> lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    >>> colors = np.random.rand(2, 3)
    >>> c = actor.streamtube(lines, colors)
    >>> scene.add(c)
    >>> #window.show(scene)

    Notes
    -----
    Streamtubes can be heavy on GPU when loading many streamlines and
    therefore, you may experience slow rendering time depending on system GPU.
    A solution to this problem is to reduce the number of points in each
    streamline. In Dipy we provide an algorithm that will reduce the number of
    points on the straighter parts of the streamline but keep more points on
    the curvier parts. This can be used in the following way::

        from dipy.tracking.distances import approx_polygon_track
        lines = [approx_polygon_track(line, 0.2) for line in lines]

    Alternatively we suggest using the ``line`` actor which is much more
    efficient.

    See Also
    --------
    :func:`fury.actor.line`

    """
    # Poly data with lines and colors
    poly_data, color_is_scalar = lines_to_vtk_polydata(lines, colors)
    next_input = poly_data

    # Set Normals
    poly_normals = set_input(PolyDataNormals(), next_input)
    poly_normals.ComputeCellNormalsOn()
    poly_normals.ComputePointNormalsOn()
    poly_normals.ConsistencyOn()
    poly_normals.AutoOrientNormalsOn()
    poly_normals.Update()
    next_input = poly_normals.GetOutputPort()

    # Spline interpolation
    if (spline_subdiv is not None) and (spline_subdiv > 0):
        spline_filter = set_input(SplineFilter(), next_input)
        spline_filter.SetSubdivideToSpecified()
        spline_filter.SetNumberOfSubdivisions(spline_subdiv)
        spline_filter.Update()
        next_input = spline_filter.GetOutputPort()

    # Add thickness to the resulting lines
    tube_filter = set_input(TubeFilter(), next_input)
    tube_filter.SetNumberOfSides(tube_sides)
    tube_filter.SetRadius(linewidth)
    # TODO using the line above we will be able to visualize
    # streamtubes of varying radius
    # tube_filter.SetVaryRadiusToVaryRadiusByScalar()
    tube_filter.CappingOn()
    tube_filter.Update()
    next_input = tube_filter.GetOutputPort()

    # Poly mapper
    poly_mapper = set_input(PolyDataMapper(), next_input)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.SetScalarModeToUsePointFieldData()
    poly_mapper.SelectColorArray("colors")
    poly_mapper.Update()

    # Color Scale with a lookup table
    if color_is_scalar:
        if lookup_colormap is None:
            lookup_colormap = colormap_lookup_table()
        poly_mapper.SetLookupTable(lookup_colormap)
        poly_mapper.UseLookupTableScalarRangeOn()
        poly_mapper.Update()

    # Set Actor
    if lod:
        actor = LODActor()
        actor.SetNumberOfCloudPoints(lod_points)
        actor.GetProperty().SetPointSize(lod_points_size)
    else:
        actor = Actor()

    actor.SetMapper(poly_mapper)

    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetOpacity(opacity)

    return actor


def line(lines, colors=None, opacity=1, linewidth=1,
         spline_subdiv=None, lod=True, lod_points=10 ** 4, lod_points_size=3,
         lookup_colormap=None, depth_cue=False, fake_tube=False):
    """Create an actor for one or more lines.

    Parameters
    ------------
    lines :  list of arrays

    colors : array (N, 3), list of arrays, tuple (3,), array (K,)
        If None or False, a standard orientation colormap is used for every
        line.
        If one tuple of color is used. Then all streamlines will have the same
        colour.
        If an array (N, 3) is given, where N is equal to the number of lines.
        Then every line is coloured with a different RGB color.
        If a list of RGB arrays is given then every point of every line takes
        a different color.
        If an array (K, 3) is given, where K is the number of points of all
        lines then every point is colored with a different RGB color.
        If an array (K,) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (L,) is given, where L is the number of streamlines then
        these are considered as the values to be used by the colormap per
        streamline.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.

    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque). Default is 1.

    linewidth : float, optional
        Line thickness. Default is 1.
    spline_subdiv : int, optional
        Number of splines subdivision to smooth streamtubes. Default is None
        which means no subdivision.
    lod : bool, optional
        Use vtkLODActor(level of detail) rather than vtkActor. Default is True.
        Level of detail actors do not render the full geometry when the
        frame rate is low.
    lod_points : int, optional
        Number of points to be used when LOD is in effect. Default is 10000.
    lod_points_size : int
        Size of points when lod is in effect. Default is 3.
    lookup_colormap : vtkLookupTable, optional
        Add a default lookup table to the colormap. Default is None which calls
        :func:`fury.actor.colormap_lookup_table`.
    depth_cue : boolean, optional
        Add a size depth cue so that lines shrink with distance to the camera.
        Works best with linewidth <= 1.
    fake_tube: boolean, optional
        Add shading to lines to approximate the look of tubes.

    Returns
    ----------
    v : vtkActor or vtkLODActor object
        Line.

    Examples
    ----------
    >>> from fury import actor, window
    >>> scene = window.Scene()
    >>> lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    >>> colors = np.random.rand(2, 3)
    >>> c = actor.line(lines, colors)
    >>> scene.add(c)
    >>> #window.show(scene)

    """
    # Poly data with lines and colors
    poly_data, color_is_scalar = lines_to_vtk_polydata(lines, colors)
    next_input = poly_data

    # use spline interpolation
    if (spline_subdiv is not None) and (spline_subdiv > 0):
        spline_filter = set_input(SplineFilter(), next_input)
        spline_filter.SetSubdivideToSpecified()
        spline_filter.SetNumberOfSubdivisions(spline_subdiv)
        spline_filter.Update()
        next_input = spline_filter.GetOutputPort()

    poly_mapper = set_input(PolyDataMapper(), next_input)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.SetScalarModeToUsePointFieldData()
    poly_mapper.SelectColorArray("colors")
    poly_mapper.Update()

    # Color Scale with a lookup table
    if color_is_scalar:
        if lookup_colormap is None:
            lookup_colormap = colormap_lookup_table()

        poly_mapper.SetLookupTable(lookup_colormap)
        poly_mapper.UseLookupTableScalarRangeOn()
        poly_mapper.Update()

    # Set Actor
    if lod:
        actor = LODActor()
        actor.SetNumberOfCloudPoints(lod_points)
        actor.GetProperty().SetPointSize(lod_points_size)
    else:
        actor = Actor()

    actor.SetMapper(poly_mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetOpacity(opacity)

    if depth_cue:
        def callback(_caller, _event, calldata=None):
            program = calldata
            if program is not None:
                program.SetUniformf("linewidth", linewidth)

        replace_shader_in_actor(actor, "geometry", load("line.geom"))
        add_shader_callback(actor, callback)

    if fake_tube:
        actor.GetProperty().SetRenderLinesAsTubes(True)

    return actor


def scalar_bar(lookup_table=None, title=" "):
    """ Default scalar bar actor for a given colormap (colorbar)

    Parameters
    ----------
    lookup_table : vtkLookupTable or None
        If None then ``colormap_lookup_table`` is called with default options.
    title : str

    Returns
    -------
    scalar_bar : vtkScalarBarActor

    See Also
    --------
    :func:`fury.actor.colormap_lookup_table`

    """
    lookup_table_copy = LookupTable()
    if lookup_table is None:
        lookup_table = colormap_lookup_table()
    # Deepcopy the lookup_table because sometimes vtkPolyDataMapper deletes it
    lookup_table_copy.DeepCopy(lookup_table)
    scalar_bar = ScalarBarActor()
    scalar_bar.SetTitle(title)
    scalar_bar.SetLookupTable(lookup_table_copy)
    scalar_bar.SetNumberOfLabels(6)

    return scalar_bar


def axes(scale=(1, 1, 1), colorx=(1, 0, 0), colory=(0, 1, 0), colorz=(0, 0, 1),
         opacity=1):
    """ Create an actor with the coordinate's system axes where
    red = x, green = y, blue = z.

    Parameters
    ----------
    scale : tuple (3,)
        Axes size e.g. (100, 100, 100). Default is (1, 1, 1).
    colorx : tuple (3,)
        x-axis color. Default red (1, 0, 0).
    colory : tuple (3,)
        y-axis color. Default green (0, 1, 0).
    colorz : tuple (3,)
        z-axis color. Default blue (0, 0, 1).
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque). Default is 1.

    Returns
    -------
    vtkActor
    """

    centers = np.zeros((3, 3))
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([colorx + (opacity,),
                       colory + (opacity,),
                       colorz + (opacity,)])

    scales = np.asarray(scale)
    return arrow(centers, dirs, colors, scales)


def odf_slicer(odfs, affine=None, mask=None, sphere=None, scale=0.5,
               norm=True, radial_scale=True, opacity=1.0, colormap=None,
               global_cm=False, B_matrix=None):
    """
    Create an actor for rendering a grid of ODFs given an array of
    spherical function (SF) or spherical harmonics (SH) coefficients.

    Parameters
    ----------
    odfs : ndarray
        4D ODFs array in SF or SH coefficients. If SH coefficients,
        `B_matrix` must be supplied.
    affine : array
        4x4 transformation array from native coordinates to world coordinates.
    mask : ndarray
        3D mask to apply to ODF field.
    sphere : dipy Sphere
        The sphere used for SH to SF projection. If None, a default sphere
        of 100 vertices will be used.
    scale : float
        Multiplicative factor to apply to ODF amplitudes.
    norm : bool
        Normalize SF amplitudes so that the maximum
        ODF amplitude per voxel along a direction is 1.
    radial_scale : bool
        Scale sphere points by ODF values.
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque).
    colormap : None or str or tuple
        The name of the colormap to use. Matplotlib colormaps are supported
        (e.g., 'inferno'). A plain color can be supplied as a RGB tuple in
        range [0, 255]. If None then a RGB colormap is used.
    global_cm : bool
        If True the colormap will be applied in all ODFs. If False
        it will be applied individually at each voxel.
    B_matrix : ndarray (n_coeffs, n_vertices)
        Optional SH to SF matrix for projecting `odfs` given in SH
        coefficents on the `sphere`. If None, then the input is assumed
        to be expressed in SF coefficients.

    Returns
    ---------
    actor : OdfSlicerActor
        vtkActor representing the ODF field.
    """
    # first we check if the input array is 4D
    n_dims = len(odfs.shape)
    if n_dims != 4:
        raise ValueError('Invalid number of dimensions for odfs. Expected 4 '
                         'dimensions, got {0} dimensions.'.format(n_dims))

    # we generate indices for all nonzero voxels
    valid_odf_mask = np.abs(odfs).max(axis=-1) > 0.
    if mask is not None:
        valid_odf_mask = np.logical_and(valid_odf_mask, mask)
    indices = np.nonzero(valid_odf_mask)
    shape = odfs.shape[:-1]

    if sphere is None:
        # Use a default sphere with 100 vertices
        vertices, faces = fp.prim_sphere('repulsion100')
    else:
        vertices = sphere.vertices
        faces = fix_winding_order(vertices, sphere.faces, clockwise=True)

    if B_matrix is None:
        if len(vertices) != odfs.shape[-1]:
            raise ValueError('Invalid nunber of SF coefficients. '
                             'Expected {0}, got {1}.'
                             .format(len(vertices), odfs.shape[-1]))
    else:
        if len(vertices) != B_matrix.shape[1]:
            raise ValueError('Invalid nunber of SH coefficients. '
                             'Expected {0}, got {1}.'
                             .format(len(vertices), B_matrix.shape[1]))

    # create and return an instance of OdfSlicerActor
    return OdfSlicerActor(odfs[indices], vertices, faces, indices, scale, norm,
                          radial_scale, shape, global_cm, colormap, opacity,
                          affine, B_matrix)


def _makeNd(array, ndim):
    """Pad as many 1s at the beginning of array's shape as are need to give
    array ndim dimensions.
    """
    new_shape = (1,) * (ndim - array.ndim) + array.shape
    return array.reshape(new_shape)


def _roll_evals(evals, axis=-1):
    """Check evals shape.

    Helper function to check that the evals provided to functions calculating
    tensor statistics have the right shape

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor. shape should be (...,3).

    axis : int
        The axis of the array which contains the 3 eigenvals. Default: -1

    Returns
    -------
    evals : array-like
        Eigenvalues of a diffusion tensor, rolled so that the 3 eigenvals are
        the last axis.

    """
    if evals.shape[-1] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[-1])
        raise ValueError(msg)

    evals = np.rollaxis(evals, axis)

    return evals

def _fa(evals, axis=-1):
    r"""Return Fractional anisotropy (FA) of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    fa : array
        Calculated FA. Range is 0 <= FA <= 1.

    Notes
    -----
    FA is calculated using the following equation:

    .. math::

        FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                    \lambda_3)^2+(\lambda_2-\lambda_3)^2}{\lambda_1^2+
                    \lambda_2^2+\lambda_3^2}}

    """
    evals = _roll_evals(evals, axis)
    # Make sure not to get nans
    all_zero = (evals == 0).all(axis=0)
    ev1, ev2, ev3 = evals
    fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                        (ev2 - ev3) ** 2 +
                        (ev3 - ev1) ** 2) /
                 ((evals * evals).sum(0) + all_zero))

    return fa


def _color_fa(fa, evecs):
    r""" Color fractional anisotropy of diffusion tensor

    Parameters
    ----------
    fa : array-like
        Array of the fractional anisotropy (can be 1D, 2D or 3D)

    evecs : array-like
        eigen vectors from the tensor model

    Returns
    -------
    rgb : Array with 3 channels for each color as the last dimension.
        Colormap of the FA with red for the x value, y for the green
        value and z for the blue value.

    Notes
    -----

    It is computed from the clipped FA between 0 and 1 using the following
    formula

    .. math::

        rgb = abs(max(\vec{e})) \times fa
    """

    if (fa.shape != evecs[..., 0, 0].shape) or ((3, 3) != evecs.shape[-2:]):
        raise ValueError("Wrong number of dimensions for evecs")

    return np.abs(evecs[..., 0]) * np.clip(fa, 0, 1)[..., None]





def tensor_slicer(evals, evecs, affine=None, mask=None, sphere=None, scale=2.2,
                  norm=True, opacity=1., scalar_colors=None):
    """Slice many tensors as ellipsoids in native or world coordinates.

    Parameters
    ----------
    evals : (3,) or (X, 3) or (X, Y, 3) or (X, Y, Z, 3) ndarray
        eigenvalues
    evecs : (3, 3) or (X, 3, 3) or (X, Y, 3, 3) or (X, Y, Z, 3, 3) ndarray
        eigenvectors
    affine : array
        4x4 transformation array from native coordinates to world coordinates*
    mask : ndarray
        3D mask
    sphere : Sphere
        a sphere
    scale : float
        Distance between spheres.
    norm : bool
        Normalize `sphere_values`.
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque). Default is 1.
    scalar_colors : (3,) or (X, 3) or (X, Y, 3) or (X, Y, Z, 3) ndarray
        RGB colors used to show the tensors
        Default None, color the ellipsoids using ``color_fa``

    Returns
    ---------
    actor : vtkActor
        Ellipsoid

    """
    if not evals.shape == evecs.shape[:-1]:
        raise RuntimeError(
            "Eigenvalues shape {} is incompatible with eigenvectors' {}."
            " Please provide eigenvalue and"
            " eigenvector arrays that have compatible dimensions."
            .format(evals.shape, evecs.shape))

    if mask is None:
        mask = np.ones(evals.shape[:3], dtype=bool)
    else:
        mask = mask.astype(bool)

    szx, szy, szz = evals.shape[:3]

    class TensorSlicerActor(LODActor):
        def __init__(self):
            self.mapper = None

        def display_extent(self, x1, x2, y1, y2, z1, z2):
            tmp_mask = np.zeros(evals.shape[:3], dtype=bool)
            tmp_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = True
            tmp_mask = np.bitwise_and(tmp_mask, mask)

            self.mapper = _tensor_slicer_mapper(evals=evals,
                                                evecs=evecs,
                                                affine=affine,
                                                mask=tmp_mask,
                                                sphere=sphere,
                                                scale=scale,
                                                norm=norm,
                                                scalar_colors=scalar_colors)
            self.SetMapper(self.mapper)

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(0, szx - 1, 0, szy - 1,
                                    int(np.floor(szz/2)), int(np.floor(szz/2)))
            if x is not None:
                self.display_extent(x, x, 0, szy - 1, 0, szz - 1)
            if y is not None:
                self.display_extent(0, szx - 1, y, y, 0, szz - 1)
            if z is not None:
                self.display_extent(0, szx - 1, 0, szy - 1, z, z)

    tensor_actor = TensorSlicerActor()
    tensor_actor.display_extent(0, szx - 1, 0, szy - 1,
                                int(np.floor(szz/2)), int(np.floor(szz/2)))

    tensor_actor.GetProperty().SetOpacity(opacity)

    return tensor_actor


def _tensor_slicer_mapper(evals, evecs, affine=None, mask=None, sphere=None,
                          scale=2.2, norm=True, scalar_colors=None):
    """Return Helper function for slicing tensor fields.

    Parameters
    ----------
    evals : (3,) or (X, 3) or (X, Y, 3) or (X, Y, Z, 3) ndarray
        eigenvalues
    evecs : (3, 3) or (X, 3, 3) or (X, Y, 3, 3) or (X, Y, Z, 3, 3) ndarray
        eigenvectors
    affine : array
        4x4 transformation array from native coordinates to world coordinates
    mask : ndarray
        3D mask
    sphere : Sphere
        a sphere
    scale : float
        Distance between spheres.
    norm : bool
        Normalize `sphere_values`.
    scalar_colors : (3,) or (X, 3) or (X, Y, 3) or (X, Y, Z, 3) ndarray
        RGB colors used to show the tensors
        Default None, color the ellipsoids using ``color_fa``

    Returns
    ---------
    mapper : vtkPolyDataMapper
        Ellipsoid mapper

    """
    mask = np.ones(evals.shape[:3]) if mask is None else mask

    ijk = np.ascontiguousarray(np.array(np.nonzero(mask)).T)
    if len(ijk) == 0:
        return None

    if affine is not None:
        ijk = np.ascontiguousarray(apply_affine(affine, ijk))

    faces = np.asarray(sphere.faces, dtype=int)
    vertices = sphere.vertices

    if scalar_colors is None:
        #from dipy.reconst.dti import color_fa, fractional_anisotropy
        cfa = _color_fa(_fa(evals), evecs)
    else:
        cfa = _makeNd(scalar_colors, 4)

    cols = np.zeros((ijk.shape[0],) + sphere.vertices.shape,
                    dtype='f4')

    all_xyz = []
    all_faces = []
    for (k, center) in enumerate(ijk):
        ea = evals[tuple(center.astype(int))]
        if norm:
            ea /= ea.max()
        ea = np.diag(ea.copy())

        ev = evecs[tuple(center.astype(int))].copy()
        xyz = np.dot(ev, np.dot(ea, vertices.T))

        xyz = xyz.T
        all_xyz.append(scale * xyz + center)
        all_faces.append(faces + k * xyz.shape[0])

        cols[k, ...] = np.interp(cfa[tuple(center.astype(int))], [0, 1],
                                 [0, 255]).astype('ubyte')

    all_xyz = np.ascontiguousarray(np.concatenate(all_xyz))
    all_xyz_vtk = numpy_support.numpy_to_vtk(all_xyz, deep=True)

    points = Points()
    points.SetData(all_xyz_vtk)

    all_faces = np.concatenate(all_faces)

    cols = np.ascontiguousarray(
        np.reshape(cols, (cols.shape[0] * cols.shape[1],
                   cols.shape[2])), dtype='f4')

    vtk_colors = numpy_support.numpy_to_vtk(
        cols,
        deep=True,
        array_type=VTK_UNSIGNED_CHAR)

    vtk_colors.SetName("colors")

    polydata = PolyData()
    polydata.SetPoints(points)
    set_polydata_triangles(polydata, all_faces)
    polydata.GetPointData().SetScalars(vtk_colors)

    mapper = PolyDataMapper()
    mapper.SetInputData(polydata)

    return mapper


def peak_slicer(peaks_dirs, peaks_values=None, mask=None, affine=None,
                colors=(1, 0, 0), opacity=1., linewidth=1, lod=False,
                lod_points=10 ** 4, lod_points_size=3, symmetric=True):
    """Visualize peak directions as given from ``peaks_from_model``.

    Parameters
    ----------
    peaks_dirs : ndarray
        Peak directions. The shape of the array can be (M, 3) or (X, M, 3) or
        (X, Y, M, 3) or (X, Y, Z, M, 3)
    peaks_values : ndarray
        Peak values. The shape of the array can be (M, ) or (X, M) or
        (X, Y, M) or (X, Y, Z, M)
    affine : array
        4x4 transformation array from native coordinates to world coordinates
    mask : ndarray
        3D mask
    colors : tuple or None
        Default red color. If None then every peak gets an orientation color
        in similarity to a DEC map.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque)
    linewidth : float, optional
        Line thickness. Default is 1.
    lod : bool
        Use vtkLODActor(level of detail) rather than vtkActor.
        Default is False. Level of detail actors do not render the full
        geometry when the frame rate is low.
    lod_points : int
        Number of points to be used when LOD is in effect. Default is 10000.
    lod_points_size : int
        Size of points when lod is in effect. Default is 3.
    symmetric: bool, optional
        If True, peaks are drawn for both peaks_dirs and -peaks_dirs. Else,
        peaks are only drawn for directions given by peaks_dirs. Default is
        True.

    Returns
    -------
    vtkActor

    See Also
    --------
    fury.actor.odf_slicer

    """
    peaks_dirs = np.asarray(peaks_dirs)
    if peaks_dirs.ndim > 5:
        raise ValueError("Wrong shape")

    peaks_dirs = _makeNd(peaks_dirs, 5)
    if peaks_values is not None:
        peaks_values = _makeNd(peaks_values, 4)

    grid_shape = np.array(peaks_dirs.shape[:3])

    if mask is None:
        mask = np.ones(grid_shape).astype(bool)

    class PeakSlicerActor(LODActor):
        def __init__(self):
            self.line = None

        def display_extent(self, x1, x2, y1, y2, z1, z2):

            tmp_mask = np.zeros(grid_shape, dtype=bool)
            tmp_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = True
            tmp_mask = np.bitwise_and(tmp_mask, mask)

            ijk = np.ascontiguousarray(np.array(np.nonzero(tmp_mask)).T)
            if len(ijk) == 0:
                self.SetMapper(None)
                return
            if affine is not None:
                ijk_trans = np.ascontiguousarray(apply_affine(affine, ijk))
            list_dirs = []
            for index, center in enumerate(ijk):
                # center = tuple(center)
                if affine is None:
                    xyz = center[:, None]
                else:
                    xyz = ijk_trans[index][:, None]
                xyz = xyz.T
                for i in range(peaks_dirs[tuple(center)].shape[-2]):

                    if peaks_values is not None:
                        pv = peaks_values[tuple(center)][i]
                    else:
                        pv = 1.
                    if symmetric:
                        dirs = np.vstack(
                            (-peaks_dirs[tuple(center)][i] * pv + xyz,
                             peaks_dirs[tuple(center)][i] * pv + xyz))
                    else:
                        dirs = np.vstack(
                            (xyz, peaks_dirs[tuple(center)][i] * pv + xyz))
                    list_dirs.append(dirs)

            self.line = line(list_dirs, colors=colors,
                             opacity=opacity, linewidth=linewidth,
                             lod=lod, lod_points=lod_points,
                             lod_points_size=lod_points_size)

            self.SetProperty(self.line.GetProperty())
            self.SetMapper(self.line.GetMapper())

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(0, szx - 1, 0, szy - 1,
                                    int(np.floor(szz/2)), int(np.floor(szz/2)))
            if x is not None:
                self.display_extent(x, x, 0, szy - 1, 0, szz - 1)
            if y is not None:
                self.display_extent(0, szx - 1, y, y, 0, szz - 1)
            if z is not None:
                self.display_extent(0, szx - 1, 0, szy - 1, z, z)

    peak_actor = PeakSlicerActor()

    szx, szy, szz = grid_shape
    peak_actor.display_extent(0, szx - 1, 0, szy - 1,
                              int(np.floor(szz / 2)), int(np.floor(szz / 2)))

    return peak_actor


def peak(peaks_dirs, peaks_values=None, mask=None, affine=None, colors=None,
         linewidth=1, lookup_colormap=None):
    """Visualize peak directions as given from ``peaks_from_model``.

    Parameters
    ----------
    peaks_dirs : ndarray
        Peak directions. The shape of the array should be (X, Y, Z, D, 3).
    peaks_values : ndarray, optional
        Peak values. The shape of the array should be (X, Y, Z, D).
    affine : array, optional
        4x4 transformation array from native coordinates to world coordinates.
    mask : ndarray, optional
        3D mask
    colors : tuple or None, optional
        Default None. If None then every peak gets an orientation color
        in similarity to a DEC map.
    lookup_colormap : vtkLookupTable, optional
        Add a default lookup table to the colormap. Default is None which calls
        :func:`fury.actor.colormap_lookup_table`.
    linewidth : float, optional
        Line thickness. Default is 1.

    Returns
    -------
    actor : PeakActor
        vtkActor or vtkLODActor representing the peaks directions and/or
        magnitudes.

    Examples
    ----------
    >>> from fury import actor, window
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> peak_dirs = np.random.rand(3, 3, 3, 3, 3)
    >>> c = actor.peak(peak_dirs)
    >>> scene.add(c)
    >>> #window.show(scene)

    """
    if peaks_dirs.ndim != 5:
        raise ValueError('Invalid peak directions. The shape of the structure '
                         'must be (XxYxZxDx3). Your data has {} dimensions.'
                         ''.format(peaks_dirs.ndim))
    if peaks_dirs.shape[4] != 3:
        raise ValueError('Invalid peak directions. The shape of the last '
                         'dimension must be 3. Your data has a last dimension '
                         'of {}.'.format(peaks_dirs.shape[4]))

    dirs_shape = peaks_dirs.shape

    if peaks_values is not None:
        if peaks_values.ndim != 4:
            raise ValueError('Invalid peak values. The shape of the structure '
                             'must be (XxYxZxD). Your data has {} dimensions.'
                             ''.format(peaks_values.ndim))
        vals_shape = peaks_values.shape
        if vals_shape != dirs_shape[:4]:
            raise ValueError('Invalid peak values. The shape of the values '
                             'must coincide with the shape of the directions.')

    valid_mask = np.abs(peaks_dirs).max(axis=(-2, -1)) > 0
    if mask is not None:
        if mask.ndim != 3:
            warnings.warn('Invalid mask. The mask must be a 3D array. The '
                          'passed mask has {} dimensions. Ignoring passed '
                          'mask.'.format(mask.ndim), UserWarning)
        elif mask.shape != dirs_shape[:3]:
            warnings.warn('Invalid mask. The shape of the mask must coincide '
                          'with the shape of the directions. Ignoring passed '
                          'mask.', UserWarning)
        else:
            valid_mask = np.logical_and(valid_mask, mask)
    indices = np.nonzero(valid_mask)

    return PeakActor(peaks_dirs, indices, values=peaks_values, affine=affine,
                     colors=colors, lookup_colormap=lookup_colormap,
                     linewidth=linewidth)


def dots(points, color=(1, 0, 0), opacity=1, dot_size=5):
    """Create one or more 3d points.

    Parameters
    ----------
    points : ndarray, (N, 3)
    color : tuple (3,)
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque)
    dot_size : int

    Returns
    --------
    vtkActor

    See Also
    ---------
    fury.actor.point

    """
    if points.ndim == 2:
        points_no = points.shape[0]
    else:
        points_no = 1

    polyVertexPoints = Points()
    polyVertexPoints.SetNumberOfPoints(points_no)
    aPolyVertex = PolyVertex()
    aPolyVertex.GetPointIds().SetNumberOfIds(points_no)

    cnt = 0
    if points.ndim > 1:
        for point in points:
            polyVertexPoints.InsertPoint(cnt, point[0], point[1], point[2])
            aPolyVertex.GetPointIds().SetId(cnt, cnt)
            cnt += 1
    else:
        polyVertexPoints.InsertPoint(cnt, points[0], points[1], points[2])
        aPolyVertex.GetPointIds().SetId(cnt, cnt)
        cnt += 1

    aPolyVertexGrid = UnstructuredGrid()
    aPolyVertexGrid.Allocate(1, 1)
    aPolyVertexGrid.InsertNextCell(aPolyVertex.GetCellType(),
                                   aPolyVertex.GetPointIds())

    aPolyVertexGrid.SetPoints(polyVertexPoints)
    aPolyVertexMapper = DataSetMapper()
    aPolyVertexMapper.SetInputData(aPolyVertexGrid)
    aPolyVertexActor = Actor()
    aPolyVertexActor.SetMapper(aPolyVertexMapper)

    aPolyVertexActor.GetProperty().SetColor(color)
    aPolyVertexActor.GetProperty().SetOpacity(opacity)
    aPolyVertexActor.GetProperty().SetPointSize(dot_size)
    return aPolyVertexActor


def point(points, colors, point_radius=0.1, phi=8, theta=8, opacity=1.):
    """Visualize points as sphere glyphs

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    colors : ndarray (N,3) or tuple (3,)
    point_radius : float
    phi : int
    theta : int
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque). Default is 1.

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> pts = np.random.rand(5, 3)
    >>> point_actor = actor.point(pts, window.colors.coral)
    >>> scene.add(point_actor)
    >>> # window.show(scene)

    """
    return sphere(centers=points, colors=colors, radii=point_radius, phi=phi,
                  theta=theta, vertices=None, faces=None, opacity=opacity)


def sphere(centers, colors, radii=1., phi=16, theta=16,
           vertices=None, faces=None, opacity=1):
    """Visualize one or many spheres with different colors and radii

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Spheres positions
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    radii : float or ndarray, shape (N,)
        Sphere radius
    phi : int
    theta : int
    vertices : ndarray, shape (N, 3)
        The point cloud defining the sphere.
    faces : ndarray, shape (M, 3)
        If faces is None then a sphere is created based on theta and phi angles
        If not then a sphere is created with the provided vertices and faces.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque). Default is 1.


    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> sphere_actor = actor.sphere(centers, window.colors.coral)
    >>> scene.add(sphere_actor)
    >>> # window.show(scene)

    """
    src = SphereSource() if faces is None else None

    if src is not None:
        src.SetRadius(1)
        src.SetThetaResolution(theta)
        src.SetPhiResolution(phi)

    actor = repeat_sources(centers=centers, colors=colors,
                           active_scalars=radii, source=src,
                           vertices=vertices, faces=faces)

    actor.GetProperty().SetOpacity(opacity)

    return actor


def cylinder(centers, directions, colors, radius=0.05, heights=1,
             capped=False, resolution=6, vertices=None, faces=None):
    """Visualize one or many cylinder with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Cylinder positions
    directions : ndarray, shape (N, 3)
        The orientation vector of the cylinder.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    radius : float
        cylinder radius, default: 1
    heights : ndarray, shape (N)
        The height of the arrow.
    capped : bool
        Turn on/off whether to cap cylinder with polygons. Default (False)
    resolution: int
        Number of facets used to define cylinder.
    vertices : ndarray, shape (N, 3)
        The point cloud defining the sphere.
    faces : ndarray, shape (M, 3)
        If faces is None then a sphere is created based on theta and phi angles
        If not then a sphere is created with the provided vertices and faces.

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> dirs = np.random.rand(5, 3)
    >>> heights = np.random.rand(5)
    >>> actor = actor.cylinder(centers, dirs, (1, 1, 1), heights=heights)
    >>> scene.add(actor)
    >>> # window.show(scene)

    """
    src = CylinderSource() if faces is None else None

    if src is not None:
        src.SetCapping(capped)
        src.SetResolution(resolution)
        src.SetRadius(radius)

    actor = repeat_sources(centers=centers, colors=colors,
                           directions=directions,
                           active_scalars=heights, source=src,
                           vertices=vertices, faces=faces)

    return actor


def square(centers, directions=(1, 0, 0), colors=(1, 0, 0), scales=1):
    """Visualize one or many squares with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Square positions
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the square.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N,3) or tuple (3,), optional
        Square size on each direction (x, y), default(1)

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> dirs = np.random.rand(5, 3)
    >>> sq_actor = actor.square(centers, dirs)
    >>> scene.add(sq_actor)
    >>> # window.show(scene)

    """
    verts, faces = fp.prim_square()
    res = fp.repeat_primitive(verts, faces, directions=directions,
                              centers=centers, colors=colors, scales=scales)

    big_verts, big_faces, big_colors, _ = res
    sq_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    sq_actor.GetProperty().BackfaceCullingOff()
    return sq_actor


def rectangle(centers, directions=(1, 0, 0), colors=(1, 0, 0),
              scales=(1, 2, 0)):
    """Visualize one or many rectangles with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Rectangle positions
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the rectangle.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N,3) or tuple (3,), optional
        Rectangle size on each direction (x, y), default(1)

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> dirs = np.random.rand(5, 3)
    >>> rect_actor = actor.rectangle(centers, dirs)
    >>> scene.add(rect_actor)
    >>> # window.show(scene)

    """
    return square(centers=centers, directions=directions, colors=colors,
                  scales=scales)


@deprecated_params(['size', 'heights'], ['scales', 'scales'],
                   since='0.6', until='0.8')
def box(centers, directions=(1, 0, 0), colors=(1, 0, 0), scales=(1, 2, 3)):
    """Visualize one or many boxes with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Box positions
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the box.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N,3) or tuple (3,), optional
        Box size on each direction (x, y), default(1)

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> dirs = np.random.rand(5, 3)
    >>> box_actor = actor.box(centers, dirs, (1, 1, 1))
    >>> scene.add(box_actor)
    >>> # window.show(scene)

    """
    verts, faces = fp.prim_box()
    res = fp.repeat_primitive(verts, faces, directions=directions,
                              centers=centers, colors=colors, scales=scales)

    big_verts, big_faces, big_colors, _ = res
    box_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    return box_actor


@deprecated_params('heights', 'scales', since='0.6', until='0.8')
def cube(centers, directions=(1, 0, 0), colors=(1, 0, 0), scales=1):
    """Visualize one or many cubes with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Cube positions
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the cube.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N,3) or tuple (3,), optional
        Cube size, default=1

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> dirs = np.random.rand(5, 3)
    >>> cube_actor = actor.cube(centers, dirs)
    >>> scene.add(cube_actor)
    >>> # window.show(scene)

    """
    return box(centers=centers, directions=directions, colors=colors,
               scales=scales)


def arrow(centers, directions, colors, heights=1., resolution=10,
          tip_length=0.35, tip_radius=0.1, shaft_radius=0.03,
          vertices=None, faces=None):
    """Visualize one or many arrows with differents features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Arrow positions
    directions : ndarray, shape (N, 3)
        The orientation vector of the arrow.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    heights : ndarray, shape (N)
        The height of the arrow.
    resolution : int
        The resolution of the arrow.
    tip_length : float
        The tip size of the arrow (default: 0.35)
    tip_radius : float
        the tip radius of the arrow (default: 0.1)
    shaft_radius : float
        The shaft radius of the arrow (default: 0.03)
    vertices : ndarray, shape (N, 3)
        The point cloud defining the arrow.
    faces : ndarray, shape (M, 3)
        If faces is None then a arrow is created based on directions, heights
        and resolution. If not then a arrow is created with the provided
        vertices and faces.

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> directions = np.random.rand(5, 3)
    >>> heights = np.random.rand(5)
    >>> arrow_actor = actor.arrow(centers, directions, (1, 1, 1), heights)
    >>> scene.add(arrow_actor)
    >>> # window.show(scene)

    """
    src = ArrowSource() if faces is None else None

    if src is not None:
        src.SetTipResolution(resolution)
        src.SetShaftResolution(resolution)
        src.SetTipLength(tip_length)
        src.SetTipRadius(tip_radius)
        src.SetShaftRadius(shaft_radius)

    actor = repeat_sources(centers=centers, directions=directions,
                           colors=colors, active_scalars=heights, source=src,
                           vertices=vertices, faces=faces)
    return actor


def cone(centers, directions, colors, heights=1., resolution=10,
         vertices=None, faces=None):
    """Visualize one or many cones with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Cone positions
    directions : ndarray, shape (N, 3)
        The orientation vector of the cone.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    heights : ndarray, shape (N)
        The height of the cone.
    resolution : int
        The resolution of the cone.
    vertices : ndarray, shape (N, 3)
        The point cloud defining the cone.
    faces : ndarray, shape (M, 3)
        If faces is None then a cone is created based on directions, heights
        and resolution. If not then a cone is created with the provided
        vertices and faces.

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> directions = np.random.rand(5, 3)
    >>> heights = np.random.rand(5)
    >>> cone_actor = actor.cone(centers, directions, (1, 1, 1), heights)
    >>> scene.add(cone_actor)
    >>> # window.show(scene)

    """
    src = ConeSource() if faces is None else None

    if src is not None:
        src.SetResolution(resolution)

    actor = repeat_sources(centers=centers, directions=directions,
                           colors=colors, active_scalars=heights, source=src,
                           vertices=vertices, faces=faces)
    return actor


def triangularprism(centers, directions=(1, 0, 0), colors=(1, 0, 0),
                    scales=1):
    """Visualize one or many regular triangular prisms with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Triangular prism positions
    directions : ndarray, shape (N, 3)
        The orientation vector(s) of the triangular prism(s)
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N,3) or tuple (3,), optional
        Triangular prism size on each direction (x, y), default(1)

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(3, 3)
    >>> dirs = np.random.rand(3, 3)
    >>> colors = np.random.rand(3, 3)
    >>> scales = np.random.rand(3, 1)
    >>> actor = actor.triangularprism(centers, dirs, colors, scales)
    >>> scene.add(actor)
    >>> # window.show(scene)

    """
    verts, faces = fp.prim_triangularprism()
    res = fp.repeat_primitive(verts, faces, directions=directions,
                              centers=centers, colors=colors, scales=scales)
    big_verts, big_faces, big_colors, _ = res
    tri_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    return tri_actor


def rhombicuboctahedron(centers, directions=(1, 0, 0), colors=(1, 0, 0),
                        scales=1):
    """Visualize one or many rhombicuboctahedron with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Rhombicuboctahedron positions
    directions : ndarray, shape (N, 3)
        The orientation vector(s) of the Rhombicuboctahedron(s)
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N,3) or tuple (3,), optional
        Rhombicuboctahedron size on each direction (x, y), default(1)

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(3, 3)
    >>> dirs = np.random.rand(3, 3)
    >>> colors = np.random.rand(3, 3)
    >>> scales = np.random.rand(3, 1)
    >>> actor = actor.rhombicuboctahedron(centers, dirs, colors, scales)
    >>> scene.add(actor)
    >>> # window.show(scene)

    """
    verts, faces = fp.prim_rhombicuboctahedron()
    res = fp.repeat_primitive(verts, faces, directions=directions,
                              centers=centers, colors=colors, scales=scales)
    big_verts, big_faces, big_colors, _ = res
    rcoh_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    return rcoh_actor


def pentagonalprism(centers, directions=(1, 0, 0), colors=(1, 0, 0),
                    scales=1):
    """Visualize one or many pentagonal prisms with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3), optional
        Pentagonal prism positions
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the pentagonal prism.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N,3) or tuple (3,), optional
        Pentagonal prism size on each direction (x, y), default(1)

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> import numpy as np
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(3, 3)
    >>> dirs = np.random.rand(3, 3)
    >>> colors = np.random.rand(3, 3)
    >>> scales = np.random.rand(3, 1)
    >>> actor_pentagonal = actor.pentagonalprism(centers, dirs, colors, scales)
    >>> scene.add(actor_pentagonal)
    >>> # window.show(scene)

    """
    verts, faces = fp.prim_pentagonalprism()
    res = fp.repeat_primitive(verts, faces, directions=directions,
                              centers=centers, colors=colors, scales=scales)

    big_verts, big_faces, big_colors, _ = res
    pent_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    return pent_actor


def octagonalprism(centers, directions=(1, 0, 0), colors=(1, 0, 0),
                   scales=1):
    """Visualize one or many octagonal prisms with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Octagonal prism positions
    directions : ndarray, shape (N, 3)
        The orientation vector of the octagonal prism.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N,3) or tuple (3,), optional
        Octagonal prism size on each direction (x, y), default(1)

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(3, 3)
    >>> dirs = np.random.rand(3, 3)
    >>> colors = np.random.rand(3, 3)
    >>> scales = np.random.rand(3, 1)
    >>> actor = actor.octagonalprism(centers, dirs, colors, scales)
    >>> scene.add(actor)
    >>> # window.show(scene)

    """
    verts, faces = fp.prim_octagonalprism()
    res = fp.repeat_primitive(verts, faces, directions=directions,
                              centers=centers, colors=colors, scales=scales)

    big_verts, big_faces, big_colors, _ = res
    oct_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    return oct_actor


def frustum(centers, directions=(1, 0, 0), colors=(0, 1, 0), scales=1):
    """Visualize one or many frustum pyramids with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Frustum pyramid positions
    directions : ndarray, shape (N, 3)
        The orientation vector of the frustum pyramid.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : int or ndarray (N,3) or tuple (3,), optional
        Frustum pyramid size on each direction (x, y), default(1)
    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(4, 3)
    >>> dirs = np.random.rand(4, 3)
    >>> colors = np.random.rand(4, 3)
    >>> scales = np.random.rand(4, 1)
    >>> actor = actor.frustum(centers, dirs, colors, scales)
    >>> scene.add(actor)
    >>> # window.show(scene)

    """
    verts, faces = fp.prim_frustum()
    res = fp.repeat_primitive(verts, faces, directions=directions,
                              centers=centers, colors=colors, scales=scales)

    big_verts, big_faces, big_colors, _ = res
    frustum_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    return frustum_actor


def superquadric(centers, roundness=(1, 1), directions=(1, 0, 0),
                 colors=(1, 0, 0), scales=1):
    """Visualize one or many superquadrics with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Superquadrics positions
    roundness : ndarray, shape (N, 2) or tuple/list (2,), optional
        parameters (Phi and Theta) that control the shape of the superquadric
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the cone.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : ndarray, shape (N) or (N,3) or float or int, optional
        The height of the cone.

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(3, 3) * 10
    >>> directions = np.random.rand(3, 3)
    >>> scales = np.random.rand(3)
    >>> colors = np.random.rand(3, 3)
    >>> roundness = np.array([[1, 1], [1, 2], [2, 1]])
    >>> sq_actor = actor.superquadric(centers, roundness=roundness,
    ...                               directions=directions,
    ...                               colors=colors, scales=scales)
    >>> scene.add(sq_actor)
    >>> # window.show(scene)

    """
    def have_2_dimensions(arr):
        return all(isinstance(i, (list, tuple, np.ndarray)) for i in arr)

    # reshape roundness to a valid numpy array
    if (isinstance(roundness, (tuple, list, np.ndarray)) and
       len(roundness) == 2 and not have_2_dimensions(roundness)):
        roundness = np.array([roundness] * centers.shape[0])
    elif isinstance(roundness, np.ndarray) and len(roundness) == 1:
        roundness = np.repeat(roundness, centers.shape[0], axis=0)
    else:
        roundness = np.array(roundness)

    res = fp.repeat_primitive_function(func=fp.prim_superquadric,
                                       centers=centers,
                                       func_args=roundness,
                                       directions=directions,
                                       colors=colors, scales=scales)

    big_verts, big_faces, big_colors, _ = res
    actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    return actor


def billboard(centers, colors=(0, 1, 0), scales=1, vs_dec=None, vs_impl=None,
              fs_dec=None, fs_impl=None, gs_dec=None, gs_impl=None):
    """Create a billboard actor.

    Billboards are 2D elements incrusted in a 3D world. It offers you the
    possibility to draw differents shapes/elements at the shader level.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Superquadrics positions
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : ndarray, shape (N) or (N,3) or float or int, optional
        The height of the cone.
    vs_dec : str or list of str, optional
        vertex shaders code that contains all variable/function delarations
    vs_impl : str or list of str, optional
        vertex shaders code that contains all variable/function implementation
    fs_dec : str or list of str, optional
        Fragment shaders code that contains all variable/function delarations
    fs_impl : str or list of str, optional
        Fragment shaders code that contains all variable/function
        implementation
    gs_dec : str or list of str, optional
        Geometry shaders code that contains all variable/function delarations
    gs_impl : str or list of str, optional
        Geometry shaders code that contains all variable/function
        mplementation

    Returns
    -------
    vtkActor

    """
    verts, faces = fp.prim_square()
    res = fp.repeat_primitive(verts, faces, centers=centers, colors=colors,
                              scales=scales)

    big_verts, big_faces, big_colors, big_centers = res

    sq_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    sq_actor.GetMapper().SetVBOShiftScaleMethod(False)
    sq_actor.GetProperty().BackfaceCullingOff()
    attribute_to_actor(sq_actor, big_centers, 'center')

    def get_code(glsl_code):
        code = ""
        if not glsl_code:
            return code

        if not all(isinstance(i, (str)) for i in glsl_code):
            raise IOError("The only supported format are string or filename,"
                          "list of string or filename")

        if isinstance(glsl_code, str):
            code += "\n"
            code += load(glsl_code) if op.isfile(glsl_code) else glsl_code
            return code

        for content in glsl_code:
            code += "\n"
            code += load(content) if op.isfile(content) else content
        return code

    vs_dec_code = get_code(vs_dec) + "\n" + load("billboard_dec.vert")
    vs_impl_code = get_code(vs_impl) + "\n" + load("billboard_impl.vert")
    fs_dec_code = get_code(fs_dec) + "\n" + load("billboard_dec.frag")
    fs_impl_code = load("billboard_impl.frag") + "\n" + get_code(fs_impl)
    gs_dec_code = get_code(gs_dec)
    gs_impl_code = get_code(gs_impl)

    shader_to_actor(sq_actor, "vertex", impl_code=vs_impl_code,
                    decl_code=vs_dec_code)
    shader_to_actor(sq_actor, "fragment", decl_code=fs_dec_code)
    shader_to_actor(sq_actor, "fragment", impl_code=fs_impl_code,
                    block="light")
    shader_to_actor(sq_actor, "geometry", impl_code=gs_impl_code,
                    decl_code=gs_dec_code, block="output")

    return sq_actor


def label(text='Origin', pos=(0, 0, 0), scale=(0.2, 0.2, 0.2),
          color=(1, 1, 1)):
    """Create a label actor.

    This actor will always face the camera

    Parameters
    ----------
    text : str
        Text for the label.
    pos : (3,) array_like, optional
        Left down position of the label.
    scale : (3,) array_like
        Changes the size of the label.
    color : (3,) array_like
        Label color as ``(r,g,b)`` tuple.

    Returns
    -------
    l : vtkActor object
        Label.

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> l = actor.label(text='Hello')
    >>> scene.add(l)
    >>> #window.show(scene)

    """
    atext = VectorText()
    atext.SetText(text)

    textm = PolyDataMapper()
    textm.SetInputConnection(atext.GetOutputPort())

    texta = Follower()
    texta.SetMapper(textm)
    texta.SetScale(scale)

    texta.GetProperty().SetColor(color)
    texta.SetPosition(pos)

    return texta


def text_3d(text, position=(0, 0, 0), color=(1, 1, 1),
            font_size=12, font_family='Arial', justification='left',
            vertical_justification="bottom",
            bold=False, italic=False, shadow=False):
    """ Generate 2D text that lives in the 3D world

    Parameters
    ----------
    text : str
    position : tuple
    color : tuple
    font_size : int
    font_family : str
    justification : str
        Left, center or right (default left)
    vertical_justification : str
        Bottom, middle or top (default bottom)
    bold : bool
    italic : bool
    shadow : bool

    Returns
    -------
    Text3D
    """

    class Text3D(TextActor3D):

        def message(self, text):
            self.set_message(text)

        def set_message(self, text):
            self.SetInput(text)
            self._update_user_matrix()

        def get_message(self):
            return self.GetInput()

        def font_size(self, size):
            self.GetTextProperty().SetFontSize(24)
            text_actor.SetScale((1./24.*size,)*3)
            self._update_user_matrix()

        def font_family(self, _family='Arial'):
            self.GetTextProperty().SetFontFamilyToArial()
            # self._update_user_matrix()

        def justification(self, justification):
            tprop = self.GetTextProperty()
            if justification == 'left':
                tprop.SetJustificationToLeft()
            elif justification == 'center':
                tprop.SetJustificationToCentered()
            elif justification == 'right':
                tprop.SetJustificationToRight()
            else:
                raise ValueError("Unknown justification: '{}'"
                                 .format(justification))

            self._update_user_matrix()

        def vertical_justification(self, justification):
            tprop = self.GetTextProperty()
            if justification == 'top':
                tprop.SetVerticalJustificationToTop()
            elif justification == 'middle':
                tprop.SetVerticalJustificationToCentered()
            elif justification == 'bottom':
                tprop.SetVerticalJustificationToBottom()
            else:
                raise ValueError("Unknown vertical justification: '{}'"
                                 .format(justification))

            self._update_user_matrix()

        def font_style(self, bold=False, italic=False, shadow=False):
            tprop = self.GetTextProperty()
            if bold:
                tprop.BoldOn()
            else:
                tprop.BoldOff()
            if italic:
                tprop.ItalicOn()
            else:
                tprop.ItalicOff()
            if shadow:
                tprop.ShadowOn()
            else:
                tprop.ShadowOff()

            self._update_user_matrix()

        def color(self, color):
            self.GetTextProperty().SetColor(*color)

        def set_position(self, position):
            self.SetPosition(position)

        def get_position(self):
            return self.GetPosition()

        def _update_user_matrix(self):
            """ Text justification of vtkTextActor3D doesn't seem to be
            working, so we do it manually. Yeah!
            """
            user_matrix = np.eye(4)

            text_bounds = [0, 0, 0, 0]
            self.GetBoundingBox(text_bounds)

            tprop = self.GetTextProperty()
            if tprop.GetJustification() == VTK_TEXT_LEFT:
                user_matrix[:3, -1] += (-text_bounds[0], 0, 0)
            elif tprop.GetJustification() == VTK_TEXT_CENTERED:
                tm = -(text_bounds[0] + (text_bounds[1] - text_bounds[0]) / 2.)
                user_matrix[:3, -1] += (tm, 0, 0)
            elif tprop.GetJustification() == VTK_TEXT_RIGHT:
                user_matrix[:3, -1] += (-text_bounds[1], 0, 0)

            if tprop.GetVerticalJustification() == VTK_TEXT_BOTTOM:
                user_matrix[:3, -1] += (0, -text_bounds[2], 0)
            elif tprop.GetVerticalJustification() == VTK_TEXT_CENTERED:
                tm = -(text_bounds[2] + (text_bounds[3] - text_bounds[2]) / 2.)
                user_matrix[:3, -1] += (0, tm, 0)
            elif tprop.GetVerticalJustification() == VTK_TEXT_TOP:
                user_matrix[:3, -1] += (0, -text_bounds[3], 0)

            user_matrix[:3, -1] *= self.GetScale()
            self.SetUserMatrix(numpy_to_vtk_matrix(user_matrix))

    text_actor = Text3D()
    text_actor.message(text)
    text_actor.font_size(font_size)
    text_actor.set_position(position)
    text_actor.font_family(font_family)
    text_actor.font_style(bold, italic, shadow)
    text_actor.color(color)
    text_actor.justification(justification)
    text_actor.vertical_justification(vertical_justification)

    return text_actor


class Container(object):
    """ Provides functionalities for grouping multiple actors using a given
    layout.

    Attributes
    ----------
    anchor : 3-tuple of float
        Anchor of this container used when laying out items in a container.
        The anchor point is relative to the center of the container.
        Default: (0, 0, 0).

    padding : 6-tuple of float
        Padding around this container bounding box. The 6-tuple represents
        (pad_x_neg, pad_x_pos, pad_y_neg, pad_y_pos, pad_z_neg, pad_z_pos).
        Default: (0, 0, 0, 0, 0, 0)

    """

    def __init__(self, layout=layout.Layout()):
        """

        Parameters
        ----------
        layout : ``fury.layout.Layout`` object
            Items of this container will be arranged according to `layout`.
        """
        self.layout = layout
        self._items = []
        self._need_update = True
        self._position = np.zeros(3)
        self._visibility = True
        self.anchor = np.zeros(3)
        self.padding = np.zeros(6)

    @property
    def items(self):
        if self._need_update:
            self.update()

        return self._items

    def add(self, *items, **kwargs):
        """Adds some items to this container.

        Parameters
        ----------
        items : `vtkProp3D` objects
            Items to add to this container.
        borrow : bool
            If True the items are added as-is, otherwise a shallow copy is
            made first. If you intend to reuse the items elsewhere you
            should set `borrow=False`. Default: True.
        """
        self._need_update = True

        for item in items:
            if not kwargs.get('borrow', True):
                item = shallow_copy(item)

            self._items.append(item)

    def clear(self):
        """ Clears all items of this container. """
        self._need_update = True
        del self._items[:]

    def update(self):
        """ Updates the position of the items of this container. """
        self.layout.apply(self._items)
        self._need_update = False

    def add_to_scene(self, ren):
        """ Adds the items of this container to a given scene. """
        for item in self.items:
            if isinstance(item, Container):
                item.add_to_scene(ren)
            else:
                ren.add(item)

    def GetBounds(self):
        """ Get the bounds of the container. """
        bounds = np.zeros(6)    # x1, x2, y1, y2, z1, z2
        bounds[::2] = np.inf    # x1, y1, z1
        bounds[1::2] = -np.inf  # x2, y2, z2

        for item in self.items:
            item_bounds = item.GetBounds()
            bounds[::2] = np.minimum(bounds[::2], item_bounds[::2])
            bounds[1::2] = np.maximum(bounds[1::2], item_bounds[1::2])

        # Add padding, if any.
        bounds[::2] -= self.padding[::2]
        bounds[1::2] += self.padding[1::2]

        return tuple(bounds)

    def GetVisibility(self):
        return self._visibility

    def SetVisibility(self, visibility):
        self._visibility = visibility
        for item in self.items:
            item.SetVisibility(visibility)

    def GetPosition(self):
        return self._position

    def AddPosition(self, position):
        self._position += position
        for item in self.items:
            item.AddPosition(position)

    def SetPosition(self, position):
        self.AddPosition(np.array(position) - self._position)

    def GetCenter(self):
        """ Get the center of the bounding box. """
        x1, x2, y1, y2, z1, z2 = self.GetBounds()
        return ((x1+x2)/2., (y1+y2)/2., (z1+z2)/2.)

    def GetLength(self):
        """ Get the length of bounding box diagonal. """
        x1, x2, y1, y2, z1, z2 = self.GetBounds()
        width, height, depth = x2-x1, y2-y1, z2-z1
        return np.sqrt(np.sum([width**2, height**2, depth**2]))

    def NewInstance(self):
        return Container(layout=self.layout)

    def ShallowCopy(self, other):
        self._position = other._position.copy()
        self.anchor = other.anchor
        self.clear()
        self.add(*other._items, borrow=False)
        self.update()

    def __len__(self):
        return len(self._items)


def grid(actors, captions=None, caption_offset=(0, -100, 0), cell_padding=0,
         cell_shape="rect", aspect_ratio=16/9., dim=None):
    """ Creates a grid of actors that lies in the xy-plane.

    Parameters
    ----------
    actors : list of `vtkProp3D` objects
        Actors to be layout in a grid manner.
    captions : list of `vtkProp3D` objects or list of str
        Objects serving as captions (can be any `vtkProp3D` object, not
        necessarily text). There should be one caption per actor. By
        default, there are no captions.
    caption_offset : tuple of float (optional)
        Tells where to position the caption w.r.t. the center of its
        associated actor. Default: (0, -100, 0).
    cell_padding : tuple of 2 floats or float
        Each grid cell will be padded according to (pad_x, pad_y) i.e.
        horizontally and vertically. Padding is evenly distributed on each
        side of the cell. If a single float is provided then both pad_x and
        pad_y will have the same value.
    cell_shape : str
        Specifies the desired shape of every grid cell.
        'rect' ensures the cells are the tightest.
        'square' ensures the cells are as wide as high.
        'diagonal' ensures the content of the cells can be rotated without
        colliding with content of the neighboring cells.
    aspect_ratio : float
        Aspect ratio of the grid (width/height). Default: 16:9.
    dim : tuple of int
        Dimension (nb_rows, nb_cols) of the grid. If provided,
        `aspect_ratio` will be ignored.

    Returns
    -------
    ``fury.actor.Container`` object
        Object that represents the grid containing all the actors and
        captions, if any.
    """
    grid_layout = layout.GridLayout(cell_padding=cell_padding,
                                    cell_shape=cell_shape,
                                    aspect_ratio=aspect_ratio, dim=dim)
    grid = Container(layout=grid_layout)

    if captions is not None:
        actors_with_caption = []
        for actor, caption in zip(actors, captions):

            actor_center = np.array(actor.GetCenter())

            # Offset accordingly the caption w.r.t.
            # the center of the associated actor.
            if isinstance(caption, str):
                caption = text_3d(caption, justification='center')
            else:
                caption = shallow_copy(caption)
            caption.SetPosition(actor_center + caption_offset)

            actor_with_caption = Container()
            actor_with_caption.add(actor, caption)

            # We change the anchor of the container so
            # the actor will be centered in the
            # grid cell.
            actor_with_caption.anchor = actor_center - \
                actor_with_caption.GetCenter()
            actors_with_caption.append(actor_with_caption)

        actors = actors_with_caption

    grid.add(*actors)
    return grid


def figure(pic, interpolation='nearest'):
    """Return a figure as an image actor.

    Parameters
    ----------
    pic : filename or numpy RGBA array
    interpolation : str
        Options are nearest, linear or cubic. Default is nearest.

    Returns
    -------
    image_actor : vtkImageActor
    """

    if isinstance(pic, str):
        vtk_image_data = load_image(pic, True)
    else:

        if pic.ndim == 3 and pic.shape[2] == 4:

            vtk_image_data = ImageData()
            vtk_image_data.AllocateScalars(VTK_UNSIGNED_CHAR, 4)

            # width, height
            vtk_image_data.SetDimensions(pic.shape[1], pic.shape[0], 1)
            vtk_image_data.SetExtent(0, pic.shape[1] - 1,
                                     0, pic.shape[0] - 1,
                                     0, 0)
            pic_tmp = np.swapaxes(pic, 0, 1)
            pic_tmp = pic.reshape(pic.shape[1] * pic.shape[0], 4)
            pic_tmp = np.ascontiguousarray(pic_tmp)
            uchar_array = numpy_support.numpy_to_vtk(pic_tmp, deep=True)
            vtk_image_data.GetPointData().SetScalars(uchar_array)

    image_actor = ImageActor()
    image_actor.SetInputData(vtk_image_data)

    if interpolation == 'nearest':
        image_actor.GetProperty().SetInterpolationTypeToNearest()

    if interpolation == 'linear':
        image_actor.GetProperty().SetInterpolationTypeToLinear()

    if interpolation == 'cubic':
        image_actor.GetProperty().SetInterpolationTypeToCubic()

    image_actor.Update()
    return image_actor


def texture(rgb, interp=True):
    """Map an RGB or RGBA texture on a plane.

    Parameters
    ----------
    rgb : ndarray
        Input 2D RGB or RGBA array. Dtype should be uint8.
    interp : bool
        Interpolate between grid centers. Default True.

    Returns
    -------
    vtkActor
    """
    arr = rgb
    grid = rgb_to_vtk(np.ascontiguousarray(arr))

    Y, X = arr.shape[:2]

    # Get vertices and triangles, then scale it
    vertices, triangles = fp.prim_square()
    vertices *= np.array([[X, Y, 0]])

    # Create a polydata
    my_polydata = PolyData()
    set_polydata_vertices(my_polydata, vertices)
    set_polydata_triangles(my_polydata, triangles)

    # Create texture object
    texture = Texture()
    texture.SetInputDataObject(grid)
    # texture.UseSRGBColorSpaceOn()
    # texture.SetPremultipliedAlpha(True)
    if interp:
        texture.InterpolateOn()

    # Map texture coordinates
    map_to_sphere = TextureMapToPlane()
    map_to_sphere.SetInputData(my_polydata)

    # Create mapper and set the mapped texture as input
    mapper = PolyDataMapper()
    mapper.SetInputConnection(map_to_sphere.GetOutputPort())
    mapper.Update()

    # Create actor and set the mapper and the texture
    act = Actor()
    act.SetMapper(mapper)
    act.SetTexture(texture)

    return act


def texture_update(texture_actor, arr):
    """
    Updates texture of an actor by updating the vtkImageData
    assigned to the vtkTexture object.

    Parameters
    ----------
    texture_actor: vtkActor
        Actor whose texture is to be updated.
    arr : ndarray
        Input 2D image in the form of RGB or RGBA array.
        This is the new image to be rendered on the actor.
        Dtype should be uint8.

    Implementation
    --------------
    Check docs/examples/viz_video_on_plane.py
    """
    grid = texture_actor.GetTexture().GetInput()
    dim = arr.shape[-1]
    img_data = np.flip(arr.swapaxes(0, 1), axis=1)\
                 .reshape((-1, dim), order='F')
    vtkarr = numpy_support.numpy_to_vtk(img_data, deep=False)
    grid.GetPointData().SetScalars(vtkarr)


def _textured_sphere_source(theta=60, phi=60):
    tss = TexturedSphereSource()
    tss.SetThetaResolution(theta)
    tss.SetPhiResolution(phi)

    return tss


def texture_on_sphere(rgb, theta=60, phi=60, interpolate=True):

    tss = _textured_sphere_source(theta=theta, phi=phi)
    earthMapper = PolyDataMapper()
    earthMapper.SetInputConnection(tss.GetOutputPort())

    earthActor = Actor()
    earthActor.SetMapper(earthMapper)

    atext = Texture()
    grid = rgb_to_vtk(rgb)
    atext.SetInputDataObject(grid)
    if interpolate:
        atext.InterpolateOn()
    earthActor.SetTexture(atext)

    return earthActor


def texture_2d(rgb, interp=False):
    """ Create 2D texture from array

    Parameters
    ----------
    rgb : ndarray
        Input 2D RGB or RGBA array. Dtype should be uint8.
    interp : bool
        Interpolate between grid centers. Default True.

    Returns
    -------
    vtkTexturedActor
    """

    arr = rgb
    Y, X = arr.shape[:2]
    size = (X, Y)
    grid = rgb_to_vtk(np.ascontiguousarray(arr))

    texture_polydata = PolyData()
    texture_points = Points()
    texture_points.SetNumberOfPoints(4)

    polys = CellArray()
    polys.InsertNextCell(4)
    polys.InsertCellPoint(0)
    polys.InsertCellPoint(1)
    polys.InsertCellPoint(2)
    polys.InsertCellPoint(3)
    texture_polydata.SetPolys(polys)

    tc = FloatArray()
    tc.SetNumberOfComponents(2)
    tc.SetNumberOfTuples(4)
    tc.InsertComponent(0, 0, 0.0)
    tc.InsertComponent(0, 1, 0.0)
    tc.InsertComponent(1, 0, 1.0)
    tc.InsertComponent(1, 1, 0.0)
    tc.InsertComponent(2, 0, 1.0)
    tc.InsertComponent(2, 1, 1.0)
    tc.InsertComponent(3, 0, 0.0)
    tc.InsertComponent(3, 1, 1.0)
    texture_polydata.GetPointData().SetTCoords(tc)

    texture_points.SetPoint(0, 0, 0, 0.0)
    texture_points.SetPoint(1, size[0], 0, 0.0)
    texture_points.SetPoint(2, size[0], size[1], 0.0)
    texture_points.SetPoint(3, 0, size[1], 0.0)
    texture_polydata.SetPoints(texture_points)

    texture_mapper = PolyDataMapper2D()
    texture_mapper = set_input(texture_mapper,
                               texture_polydata)

    act = TexturedActor2D()
    act.SetMapper(texture_mapper)

    tex = Texture()
    tex.SetInputDataObject(grid)
    if interp:
        tex.InterpolateOn()
    tex.Update()
    act.SetTexture(tex)
    return act


def sdf(centers, directions=(1, 0, 0), colors=(1, 0, 0), primitives='torus',
        scales=1):
    """Create a SDF primitive based actor

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        SDF primitive positions
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    directions : ndarray, shape (N, 3)
        The orientation vector of the SDF primitive.
    primitives : str, list, tuple, np.ndarray
        The primitive of choice to be rendered.
        Options are sphere, torus and ellipsoid. Default is torus.
    scales : float
        The size of the SDF primitive

    Returns
    -------
    vtkActor
    """

    prims = {'sphere': 1, 'torus': 2, 'ellipsoid': 3, 'capsule': 4}

    verts, faces = fp.prim_box()
    repeated = fp.repeat_primitive(verts, faces, centers=centers,
                                   colors=colors, directions=directions,
                                   scales=scales)

    rep_verts, rep_faces, rep_colors, rep_centers = repeated
    box_actor = get_actor_from_primitive(rep_verts, rep_faces, rep_colors)
    box_actor.GetMapper().SetVBOShiftScaleMethod(False)

    if isinstance(primitives,  (list, tuple, np.ndarray)):
        primlist = [prims[prim] for prim in primitives]
        if len(primitives) < len(centers):
            primlist = primlist + [2] * (len(centers) - len(primitives))
            warnings.warn("Not enough primitives provided,\
             defaulting to torus", category=UserWarning)
        rep_prims = np.repeat(primlist, verts.shape[0])
    else:
        rep_prims = np.repeat(prims[primitives], rep_centers.shape[0], axis=0)

    if isinstance(scales, (list, tuple, np.ndarray)):
        rep_scales = np.repeat(scales, verts.shape[0])
    else:
        rep_scales = np.repeat(scales, rep_centers.shape[0], axis=0)

    if isinstance(directions, (list, tuple, np.ndarray)) and \
            len(directions) == 3:
        rep_directions = np.repeat(directions, rep_centers.shape[0], axis=0)
    else:
        rep_directions = np.repeat(directions, verts.shape[0], axis=0)

    attribute_to_actor(box_actor, rep_centers, 'center')
    attribute_to_actor(box_actor, rep_prims, 'primitive')
    attribute_to_actor(box_actor, rep_scales, 'scale')
    attribute_to_actor(box_actor, rep_directions, 'direction')

    vs_dec_code = load("sdf_dec.vert")
    vs_impl_code = load("sdf_impl.vert")
    fs_dec_code = load("sdf_dec.frag")
    fs_impl_code = load("sdf_impl.frag")

    shader_to_actor(box_actor, "vertex", impl_code=vs_impl_code,
                    decl_code=vs_dec_code)
    shader_to_actor(box_actor, "fragment", decl_code=fs_dec_code)
    shader_to_actor(box_actor, "fragment", impl_code=fs_impl_code,
                    block="light")
    return box_actor


def markers(
        centers,
        colors=(0, 1, 0),
        scales=1,
        marker='3d',
        marker_opacity=.8,
        edge_width=.0,
        edge_color=(255, 255, 255),
        edge_opacity=.8
):
    """Create a marker actor with different shapes.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : ndarray, shape (N) or (N,3) or float or int, optional
    marker : str or a list
        Available markers are: '3d', 'o', 's', 'd', '^', 'p', 'h', 's6',
        'x', '+', optional
    marker_opacity : float, optional
    edge_width : int, optional
    edge_color : ndarray, shape (3), optional

    Returns
    -------
    vtkActor

    """

    n_markers = centers.shape[0]
    verts, faces = fp.prim_square()
    res = fp.repeat_primitive(verts, faces, centers=centers, colors=colors,
                              scales=scales)

    big_verts, big_faces, big_colors, big_centers = res
    sq_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    sq_actor.GetMapper().SetVBOShiftScaleMethod(False)
    sq_actor.GetProperty().BackfaceCullingOff()

    attribute_to_actor(sq_actor, big_centers, 'center')
    marker2id = {
        'o': 0, 's': 1, 'd': 2, '^': 3, 'p': 4,
        'h': 5, 's6': 6, 'x': 7, '+': 8, '3d': 0}

    vs_dec_code = load("billboard_dec.vert")
    vs_dec_code += f'\n{load("marker_billboard_dec.vert")}'
    vs_impl_code = load("billboard_impl.vert")
    vs_impl_code += f'\n{load("marker_billboard_impl.vert")}'

    fs_dec_code = load('billboard_dec.frag')
    fs_dec_code += f'\n{load("marker_billboard_dec.frag")}'
    fs_impl_code = load('billboard_impl.frag')

    if marker == '3d':
        fs_impl_code += f'{load("billboard_spheres_impl.frag")}'
    else:
        fs_impl_code += f'{load("marker_billboard_impl.frag")}'
        if isinstance(marker, str):
            list_of_markers = np.ones(n_markers)*marker2id[marker]
        else:
            list_of_markers = [marker2id[i] for i in marker]

        list_of_markers = np.repeat(list_of_markers, 4).astype('float')
        attribute_to_actor(
            sq_actor,
            list_of_markers, 'marker')

    def callback(
        _caller, _event, calldata=None,
            uniform_type='f', uniform_name=None, value=None):
        program = calldata
        if program is not None:
            program.__getattribute__(f'SetUniform{uniform_type}')(
                uniform_name, value)

    add_shader_callback(
        sq_actor, partial(
            callback, uniform_type='f', uniform_name='edgeWidth',
            value=edge_width))
    add_shader_callback(
        sq_actor, partial(
            callback, uniform_type='f', uniform_name='markerOpacity',
            value=marker_opacity))
    add_shader_callback(
        sq_actor, partial(
            callback, uniform_type='f', uniform_name='edgeOpacity',
            value=edge_opacity))
    add_shader_callback(
        sq_actor, partial(
            callback, uniform_type='3f', uniform_name='edgeColor',
            value=edge_color))

    shader_to_actor(sq_actor, "vertex", impl_code=vs_impl_code,
                    decl_code=vs_dec_code)
    shader_to_actor(sq_actor, "fragment", decl_code=fs_dec_code)
    shader_to_actor(sq_actor, "fragment", impl_code=fs_impl_code,
                    block="light")

    return sq_actor
