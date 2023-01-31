import vtkmodules.vtkCommonCore as ccvtk    # type: ignore
import vtkmodules.vtkCommonDataModel as cdmvtk    # type: ignore
import vtkmodules.vtkCommonExecutionModel as cemvtk  # type: ignore
import vtkmodules.vtkCommonTransforms as ctvtk   # type: ignore
import vtkmodules.vtkCommonMath as cmvtk   # type: ignore
import vtkmodules.vtkDomainsChemistry as dcvtk   # type: ignore
import vtkmodules.vtkDomainsChemistryOpenGL2 as dcovtk   # type: ignore
import vtkmodules.vtkFiltersCore as fcvtk  # type: ignore
import vtkmodules.vtkFiltersGeneral as fgvtk   # type: ignore
import vtkmodules.vtkFiltersHybrid as fhvtk   # type: ignore
import vtkmodules.vtkFiltersModeling as fmvtk   # type: ignore
import vtkmodules.vtkFiltersSources as fsvtk   # type: ignore
import vtkmodules.vtkFiltersTexture as ftvtk   # type: ignore
import vtkmodules.vtkImagingCore as icvtk   # type: ignore
import vtkmodules.vtkInteractionStyle as isvtk   # type: ignore
import vtkmodules.vtkIOImage as ioivtk   # type: ignore
import vtkmodules.vtkIOLegacy as iolvtk   # type: ignore
import vtkmodules.vtkIOPLY as ioplyvtk   # type: ignore
import vtkmodules.vtkIOGeometry as iogvtk   # type: ignore
import vtkmodules.vtkIOMINC as iomincvtk   # type: ignore
import vtkmodules.vtkIOXML as ioxmlvtk   # type: ignore
import vtkmodules.vtkRenderingAnnotation as ravtk   # type: ignore
import vtkmodules.vtkRenderingCore as rcvtk   # type: ignore
import vtkmodules.vtkRenderingFreeType as rftvtk   # type: ignore
import vtkmodules.vtkRenderingLOD as rlodvtk   # type: ignore
import vtkmodules.vtkRenderingOpenGL2 as roglvtk   # type: ignore


from vtkmodules.util import numpy_support, colors  # type: ignore # noqa: F401
from vtkmodules.util.misc import calldata_type   # type: ignore # noqa: F401


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
VTK_INT = ccvtk.VTK_INT
VTK_DOUBLE = ccvtk.VTK_DOUBLE
VTK_FLOAT = ccvtk.VTK_FLOAT
VTK_TEXT_LEFT = ccvtk.VTK_TEXT_LEFT
VTK_TEXT_RIGHT = ccvtk.VTK_TEXT_RIGHT
VTK_TEXT_BOTTOM = ccvtk.VTK_TEXT_BOTTOM
VTK_TEXT_TOP = ccvtk.VTK_TEXT_TOP
VTK_TEXT_CENTERED = ccvtk.VTK_TEXT_CENTERED
VTK_UNSIGNED_CHAR = ccvtk.VTK_UNSIGNED_CHAR
VTK_UNSIGNED_INT = ccvtk.VTK_UNSIGNED_INT
VTK_UNSIGNED_SHORT = ccvtk.VTK_UNSIGNED_SHORT

##############################################################
#  vtkCommonExecutionModel Module
AlgorithmOutput = cemvtk.vtkAlgorithmOutput

##############################################################
#  vtkRenderingCore Module
Renderer = rcvtk.vtkRenderer
Skybox = rcvtk.vtkSkybox
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
Camera = rcvtk.vtkCamera

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
OpenGLRenderer = roglvtk.vtkOpenGLRenderer
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
TransformPolyDataFilter = fgvtk.vtkTransformPolyDataFilter

##############################################################
#  vtkFiltersHybrid Module
RenderLargeImage = fhvtk.vtkRenderLargeImage

##############################################################
#  vtkFiltersModeling Module
LoopSubdivisionFilter = fmvtk.vtkLoopSubdivisionFilter
ButterflySubdivisionFilter = fmvtk.vtkButterflySubdivisionFilter
OutlineFilter = fmvtk.vtkOutlineFilter
LinearExtrusionFilter = fmvtk.vtkLinearExtrusionFilter

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
ImageFlip = icvtk.vtkImageFlip
ImageReslice = icvtk.vtkImageReslice
ImageMapToColors = icvtk.vtkImageMapToColors

##############################################################
#  vtkIOImage vtkIOLegacy, vtkIOPLY, vtkIOGeometry,
# vtkIOMINC Modules
ImageReader2Factory = ioivtk.vtkImageReader2Factory
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
