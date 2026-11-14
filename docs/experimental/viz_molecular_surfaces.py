"""
======================================================================
Molecular Surfaces
======================================================================

There are three types of molecular Surfaces:

a. Van der Waals
b. Solvent Accessible
c. Solvent Excluded

This program aims to implement all the surfaces, currently the first
two i.e. Van der Waals and Solvent Accessible are implemented.
This program is based on the paper "Generating Triangulated
Macromolecular Surfaces by Euclidean Distance Transform" by Dong Xu
and Yang Zhang.

Importing necessary modules
"""
import urllib
import os
from vtk.util.numpy_support import numpy_to_vtk
import vtk
from fury import window, utils
import numpy as np


###############################################################################
# Downloading the PDB file whose model is to be rendered.
# User can change the pdb_code depending on which protein they want to
# visualize
pdb_code = '1pgb'
downloadurl = "https://files.rcsb.org/download/"
pdbfn = pdb_code + ".cif"
flag = 0
if not os.path.isfile(pdbfn):
    flag = 1
    url = downloadurl + pdbfn
    outfnm = os.path.join(pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
    except Exception:
        print("Error in downloading the file!")

###############################################################################
# ``table`` will help in accessing the atomic information such as radius of the
# atoms.
table = vtk.vtkPeriodicTable()

# atomic_nums will store the atomic number of the atoms
atomic_nums = []

# coords will store the coordinates of the centers of the atoms
coords = []

###############################################################################
# Accessing and parsing the PDB file.
pdbfile = open(pdbfn, 'r')
pdb_lines = pdbfile.readlines()

# parsing pdbx file
for line in pdb_lines:
    _line = line.split()
    try:
        if _line[0] == 'ATOM' or _line[0] == 'HETATM':

            # obtain coordinates of atom
            coorX, coorY, coorZ = float(_line[10]), float(_line[11]), \
                                  float(_line[12])
            # obtain the atomic number of atom
            atomic_num = table.GetAtomicNumber(_line[2])

            coords += [[coorX, coorY, coorZ]]
            atomic_nums += [atomic_num]
    except Exception:
        continue

coords = np.array(coords)

# stores the radii of atoms
radii = np.array([table.GetVDWRadius(atomic_num) for
                  atomic_num in atomic_nums])

# scaling coordinates of atom centers
# obtaining the maximum difference along x, y and z axes
diff_x = max(coords[:, 0]) - min(coords[:, 0])
diff_y = max(coords[:, 1]) - min(coords[:, 1])
diff_z = max(coords[:, 2]) - min(coords[:, 2])

# selecting difference with the largest magnitude
diff = max(diff_x, diff_z, diff_y)

# selecting the minimum value corresponding to the difference
# for example if diff_x is the greatest in magnitude, mini will be
# assigned the value of minimum value from x coordinates.
if (diff == diff_x):
    mini = min(coords[:, 0])
elif (diff == diff_y):
    mini = min(coords[:, 1])
else:
    mini = min(coords[:, 2])

###############################################################################
# resolution of the grid, higher the resolution, better the surface
num_grid_points = 256

###############################################################################
# Scaling:
# subtracting mini from atomic centers and dividing it by diff
# multiplying it by num_grid_points to scale it inside the grid,
# dividing by 1.5 to ensure it's well inside the grid to ensure
# that points don't lie outside the bounding box while constructing
# SAS and VDW surfaces.
newcoords = (coords-mini)/diff*(num_grid_points//1.8)

# probe radius
probe_radius = 1.4
# scaling radius
newradii = (radii+probe_radius)/diff*(num_grid_points//1.8)

# for translating the atoms
xmean = np.mean(newcoords[:, 0])
ymean = np.mean(newcoords[:, 1])
zmean = np.mean(newcoords[:, 2])

newcoords[:, 0] = newcoords[:, 0] + (num_grid_points-1)//2 - xmean
newcoords[:, 1] = newcoords[:, 1] + (num_grid_points-1)//2 - ymean
newcoords[:, 2] = newcoords[:, 2] + (num_grid_points-1)//2 - zmean

# converting coordinates into integer coordinates
newcoords = np.array(newcoords, dtype=int)

###############################################################################
# constructing the volume grid
grid_coords = np.zeros((num_grid_points**3, 3))
x = np.arange(num_grid_points)
y = np.arange(num_grid_points)
z = np.arange(num_grid_points)
x, y, z = np.meshgrid(x, y, z)
x = x.reshape(-1)
y = y.reshape(-1)
z = z.reshape(-1)
grid_coords[:] = np.vstack([x, y, z]).T

# constructing grid polydata
polydata = vtk.vtkPolyData()
points = utils.numpy_to_vtk_points(grid_coords)
polydata.SetPoints(points)

# locating points within rp + ri, newcoords is list of atom centers
pointTree = vtk.vtkOctreePointLocator()
pointTree.SetDataSet(polydata)
pointTree.BuildLocator()

ids = []

colors = []

###############################################################################
# Getting ids of voxels which are part of SAS
for i, coord in enumerate(newcoords):
    result = vtk.vtkIdList()
    pointTree.FindPointsWithinRadius(newradii[i], coord, result)
    k = result.GetNumberOfIds()
    for j in range(k):
        ids += [result.GetId(j)]

# removing overlapping voxels
ids = np.unique(np.array(ids))
newcoords2 = grid_coords[ids]

data = np.zeros(num_grid_points**3, dtype='f8')
data[ids] = 100

# Smoothing via gaussian filer
# import scipy
# data = scipy.ndimage.gaussian_filter(data, sigma=1)

###############################################################################
# Creating a vtkImage
imdata = vtk.vtkImageData()
depthArray = numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)

imdata.SetDimensions(num_grid_points, num_grid_points, num_grid_points)
spacing = 1
imdata.SetSpacing([spacing, spacing, spacing])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

###############################################################################
# Volummetric Model
colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(2, 0, 1, 1)  # Cyan

opacity = vtk.vtkPiecewiseFunction()

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(opacity)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.SetIndependentComponents(2)

volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
volumeMapper.SetInputData(imdata)
volumeMapper.SetBlendModeToMaximumIntensity()
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)


###############################################################################
# Generating isosurface from 3D image data (volume)
# Can use vtkMarchingcubes but vtkFlyingEdges3D is much faster and there's no
# perceptible difference in the quality of meshes created.

# mc = vtk.vtkMarchingCubes()
mc = vtk.vtkFlyingEdges3D()
mc.SetInputData(imdata)
mc.ComputeNormalsOn()
mc.ComputeGradientsOn()
# threshold = 400
# mc.SetValue(0, threshold)
mc.GenerateValues(1, 1, 1)
mc.Update()

###############################################################################
# Smoothing is performed on the isosuface by the following two filters
# successively:
# 1. vtkWindowedSincPolyDataFilter
# 2. vtkSmoothPolyDataFilter

# Smoothing by using vtkWindowedSincPolyDataFilter
smoothing_iterations = 32
pass_band = 0.001
feature_angle = 150.0
smoother1 = vtk.vtkWindowedSincPolyDataFilter()
smoother1.SetInputConnection(mc.GetOutputPort())
smoother1.SetNumberOfIterations(smoothing_iterations)
# smoother1.BoundarySmoothingOff()
# smoother1.FeatureEdgeSmoothingOff()
# smoother1.NormalizeCoordinatesOn()
smoother1.SetFeatureAngle(feature_angle)
smoother1.SetPassBand(pass_band)
smoother1.NonManifoldSmoothingOn()
smoother1.NormalizeCoordinatesOn()
smoother1.Update()


###############################################################################
# Smoothing by using vtkSmoothPolyDataFilter

smoother2 = vtk.vtkSmoothPolyDataFilter()
smoother2.SetInputConnection(mc.GetOutputPort())
smoother2.SetNumberOfIterations(150)
# smoother2.SetRelaxationFactor(0.001)
smoother2.FeatureEdgeSmoothingOn()
smoother2.BoundarySmoothingOn()
smoother2.Update()

# Save the isosurface in an stl file
# writer = vtk.vtkSTLWriter()
# writer.SetInputConnection(smoother2.GetOutputPort())
# writer.SetFileTypeToBinary()
# writer.SetFileName(pdbx_code+ ".stl")
# writer.Write()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(smoother2.GetOutputPort())
mapper.ScalarVisibilityOff()
sas_actor = vtk.vtkActor()
sas_actor.SetMapper(mapper)


scene = window.Scene()

# Solvent Accessible Surface actor
scene.add(sas_actor)

# Volummetric model
# volumeMapper.SetBlendModeToIsoSurface()
# volumeMapper.SetBlendModeToComposite()
# scene.AddVolume(volume)

# Voxels forming the molecular surface
# scene.add(actor.markers(grid_coords, scales=0.2, colors=np.random.rand(3),
#                         marker_opacity=0.2))

# Voxels forming the grid
# scene.add(actor.markers(newcoords2, scales=0.4, marker_opacity=1,
#           colors=np.random.rand(3))


###############################################################################
# Delete the PDB file if it's downloaded from the internet
if flag:
    os.remove(outfnm)


interactive = True
if interactive:
    window.show(scene, size=(600, 600))
