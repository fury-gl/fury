import vtkmodules.vtkCommonCore as ccvtk
import vtkmodules.vtkCommonDataModel as cdmvtk
import vtkmodules.vtkCommonExecutionModel as cemvtk
import vtkmodules.vtkCommonTransforms as ctvtk
import vtkmodules.vtkCommonMath as cmvtk
import vtkmodules.vtkDomainsChemistry as dcvtk
import vtkmodules.vtkDomainsChemistryOpenGL2 as dcovtk
import vtkmodules.vtkFiltersCore as fcvtk
import vtkmodules.vtkFiltersGeneral as fgvtk
import vtkmodules.vtkFiltersHybrid as fhvtk
import vtkmodules.vtkFiltersModeling as fmvtk
import vtkmodules.vtkFiltersSources as fsvtk
import vtkmodules.vtkFiltersTexture as ftvtk
import vtkmodules.vtkImagingCore as icvtk
import vtkmodules.vtkInteractionStyle as isvtk
import vtkmodules.vtkIOImage as ioivtk
import vtkmodules.vtkIOLegacy as iolvtk
import vtkmodules.vtkIOPLY as ioplyvtk
import vtkmodules.vtkIOGeometry as iogvtk
import vtkmodules.vtkIOMINC as iomincvtk
import vtkmodules.vtkIOXML as ioxmlvtk
import vtkmodules.vtkRenderingAnnotation as ravtk
import vtkmodules.vtkRenderingCore as rcvtk
import vtkmodules.vtkRenderingFreeType as rftvtk
import vtkmodules.vtkRenderingLOD as rlodvtk
import vtkmodules.vtkRenderingOpenGL2 as roglvtk


from vtkmodules.util import numpy_support, colors
from vtkmodules.util.misc import calldata_type


VTK_9_PLUS = ccvtk.vtkVersion.GetVTKMajorVersion() >= 9
VTK_VERSION = ccvtk.vtkVersion.GetVTKVersion()

##############################################################
#  vtkCommonCore Module
Command = ccvtk.vtkCommand
LookupTable = ccvtk.vtkLookupTable
Points = ccvtk.vtkPoints
IdTypeArray = ccvtk.vtkIdTypeArray
FloatArray = ccvtk.vtkFloatArray
DoubleArray = ccvtk.vtkDoubleArray
StringArray = ccvtk.vtkStringArray
UnsignedCharArray = ccvtk.vtkUnsignedCharArray
VTK_OBJECT = ccvtk.VTK_OBJECT
VTK_ID_TYPE = ccvtk.VTK_ID_TYPE
VTK_DOUBLE = ccvtk.VTK_DOUBLE
VTK_FLOAT = ccvtk.VTK_FLOAT
VTK_TEXT_LEFT = ccvtk.VTK_TEXT_LEFT
VTK_TEXT_RIGHT = ccvtk.VTK_TEXT_RIGHT
VTK_TEXT_BOTTOM = ccvtk.VTK_TEXT_BOTTOM
VTK_TEXT_TOP = ccvtk.VTK_TEXT_TOP
VTK_TEXT_CENTERED = ccvtk.VTK_TEXT_CENTERED
VTK_UNSIGNED_CHAR = ccvtk.VTK_UNSIGNED_CHAR
VTK_UNSIGNED_SHORT = ccvtk.VTK_UNSIGNED_SHORT
VTK_UNSIGNED_INT = ccvtk.VTK_UNSIGNED_INT

##############################################################
#  vtkCommonExecutionModel Module
AlgorithmOutput = cemvtk.vtkAlgorithmOutput

##############################################################
#  vtkRenderingCore Module
Renderer = rcvtk.vtkRenderer
Volume = rcvtk.vtkVolume
Actor2D = rcvtk.vtkActor2D
Actor = rcvtk.vtkActor
RenderWindow = rcvtk.vtkRenderWindow
RenderWindowInteractor = rcvtk.vtkRenderWindowInteractor
InteractorEventRecorder = rcvtk.vtkInteractorEventRecorder
WindowToImageFilter = rcvtk.vtkWindowToImageFilter
InteractorStyle = rcvtk.vtkInteractorStyle
PropPicker = rcvtk.vtkPropPicker
PointPicker = rcvtk.vtkPointPicker
CellPicker = rcvtk.vtkCellPicker
WorldPointPicker = rcvtk.vtkWorldPointPicker
HardwareSelector = rcvtk.vtkHardwareSelector
ImageActor = rcvtk.vtkImageActor
PolyDataMapper = rcvtk.vtkPolyDataMapper
PolyDataMapper2D = rcvtk.vtkPolyDataMapper2D
Assembly = rcvtk.vtkAssembly
DataSetMapper = rcvtk.vtkDataSetMapper
Texture = rcvtk.vtkTexture
TexturedActor2D = rcvtk.vtkTexturedActor2D
Follower = rcvtk.vtkFollower
TextActor = rcvtk.vtkTextActor
TextActor3D = rcvtk.vtkTextActor3D
Property2D = rcvtk.vtkProperty2D

##############################################################
#  vtkRenderingFreeType Module
VectorText = rftvtk.vtkVectorText

##############################################################
#  vtkRenderingLOD Module
LODActor = rlodvtk.vtkLODActor

##############################################################
#  vtkRenderingAnnotation Module
ScalarBarActor = ravtk.vtkScalarBarActor

##############################################################
#  vtkRenderingOpenGL2 Module
Shader = roglvtk.vtkShader

##############################################################
#  vtkInteractionStyle Module
InteractorStyleImage = isvtk.vtkInteractorStyleImage
InteractorStyleTrackballActor = isvtk.vtkInteractorStyleTrackballActor
InteractorStyleTrackballCamera = isvtk.vtkInteractorStyleTrackballCamera
InteractorStyleUser = isvtk.vtkInteractorStyleUser

##############################################################
#  vtkFiltersCore Module
CleanPolyData = fcvtk.vtkCleanPolyData
PolyDataNormals = fcvtk.vtkPolyDataNormals
ContourFilter = fcvtk.vtkContourFilter
TubeFilter = fcvtk.vtkTubeFilter
Glyph3D = fcvtk.vtkGlyph3D

##############################################################
#  vtkFiltersGeneral Module
SplineFilter = fgvtk.vtkSplineFilter

##############################################################
#  vtkFiltersHybrid Module
RenderLargeImage = fhvtk.vtkRenderLargeImage

##############################################################
#  vtkFiltersModeling Module
LoopSubdivisionFilter = fmvtk.vtkLoopSubdivisionFilter
ButterflySubdivisionFilter = fmvtk.vtkButterflySubdivisionFilter
OutlineFilter = fmvtk.vtkOutlineFilter

##############################################################
#  vtkFiltersTexture Module
TextureMapToPlane = ftvtk.vtkTextureMapToPlane

##############################################################
#  vtkFiltersSource Module
SphereSource = fsvtk.vtkSphereSource
CylinderSource = fsvtk.vtkCylinderSource
ArrowSource = fsvtk.vtkArrowSource
ConeSource = fsvtk.vtkConeSource
DiskSource = fsvtk.vtkDiskSource
TexturedSphereSource = fsvtk.vtkTexturedSphereSource
RegularPolygonSource = fsvtk.vtkRegularPolygonSource

##############################################################
#  vtkCommonDataModel Module
PolyData = cdmvtk.vtkPolyData
ImageData = cdmvtk.vtkImageData
DataObject = cdmvtk.vtkDataObject
CellArray = cdmvtk.vtkCellArray
PolyVertex = cdmvtk.vtkPolyVertex
UnstructuredGrid = cdmvtk.vtkUnstructuredGrid
Polygon = cdmvtk.vtkPolygon
DataObject = cdmvtk.vtkDataObject
Molecule = cdmvtk.vtkMolecule
DataSetAttributes = cdmvtk.vtkDataSetAttributes

##############################################################
#  vtkCommonTransforms Module
Transform = ctvtk.vtkTransform

##############################################################
#  vtkCommonTransforms Module
Matrix4x4 = cmvtk.vtkMatrix4x4
Matrix3x3 = cmvtk.vtkMatrix3x3

##############################################################
#  vtkImagingCore Module
ImageReslice = icvtk.vtkImageReslice
ImageMapToColors = icvtk.vtkImageMapToColors

##############################################################
#  vtkIOImage vtkIOLegacy, vtkIOPLY, vtkIOGeometry,
# vtkIOMINC Modules
PNGReader = ioivtk.vtkPNGReader
BMPReader = ioivtk.vtkBMPReader
JPEGReader = ioivtk.vtkJPEGReader
TIFFReader = ioivtk.vtkTIFFReader
PLYReader = ioplyvtk.vtkPLYReader
STLReader = iogvtk.vtkSTLReader
OBJReader = iogvtk.vtkOBJReader
MNIObjectReader = iomincvtk.vtkMNIObjectReader
PolyDataReader = iolvtk.vtkPolyDataReader
XMLPolyDataReader = ioxmlvtk.vtkXMLPolyDataReader
PNGWriter = ioivtk.vtkPNGWriter
BMPWriter = ioivtk.vtkBMPWriter
JPEGWriter = ioivtk.vtkJPEGWriter
TIFFWriter = ioivtk.vtkTIFFWriter
PLYWriter = ioplyvtk.vtkPLYWriter
STLWriter = iogvtk.vtkSTLWriter
MNIObjectWriter = iomincvtk.vtkMNIObjectWriter
PolyDataWriter = iolvtk.vtkPolyDataWriter
XMLPolyDataWriter = ioxmlvtk.vtkXMLPolyDataWriter

##############################################################
#  vtkDomainsChemistry  and vtkDomainsChemistryOpenGL2 Module
SimpleBondPerceiver = dcvtk.vtkSimpleBondPerceiver
ProteinRibbonFilter = dcvtk.vtkProteinRibbonFilter
PeriodicTable = dcvtk.vtkPeriodicTable
OpenGLMoleculeMapper = dcovtk.vtkOpenGLMoleculeMapper
