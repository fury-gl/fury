from vtkmodules.util import colors, numpy_support  # type: ignore # noqa: F401
from vtkmodules.util.misc import calldata_type  # type: ignore # noqa: F401
import vtkmodules.vtkCommonCore as ccvtk  # type: ignore
import vtkmodules.vtkCommonDataModel as cdmvtk  # type: ignore
import vtkmodules.vtkCommonExecutionModel as cemvtk  # type: ignore
import vtkmodules.vtkCommonMath as cmvtk  # type: ignore
import vtkmodules.vtkCommonTransforms as ctvtk  # type: ignore
import vtkmodules.vtkDomainsChemistry as dcvtk  # type: ignore
import vtkmodules.vtkDomainsChemistryOpenGL2 as dcovtk  # type: ignore
import vtkmodules.vtkFiltersCore as fcvtk  # type: ignore
import vtkmodules.vtkFiltersGeneral as fgvtk  # type: ignore
import vtkmodules.vtkFiltersHybrid as fhvtk  # type: ignore
import vtkmodules.vtkFiltersModeling as fmvtk  # type: ignore
import vtkmodules.vtkFiltersSources as fsvtk  # type: ignore
import vtkmodules.vtkFiltersTexture as ftvtk  # type: ignore
import vtkmodules.vtkIOGeometry as iogvtk  # type: ignore
import vtkmodules.vtkIOImage as ioivtk  # type: ignore
import vtkmodules.vtkIOLegacy as iolvtk  # type: ignore
import vtkmodules.vtkIOMINC as iomincvtk  # type: ignore
import vtkmodules.vtkIOPLY as ioplyvtk  # type: ignore
import vtkmodules.vtkIOXML as ioxmlvtk  # type: ignore
import vtkmodules.vtkImagingCore as icvtk  # type: ignore
import vtkmodules.vtkInteractionStyle as isvtk  # type: ignore
import vtkmodules.vtkRenderingAnnotation as ravtk  # type: ignore
import vtkmodules.vtkRenderingCore as rcvtk  # type: ignore
import vtkmodules.vtkRenderingFreeType as rftvtk  # type: ignore
import vtkmodules.vtkRenderingLOD as rlodvtk  # type: ignore
import vtkmodules.vtkRenderingOpenGL2 as roglvtk  # type: ignore

VTK_VERSION = ccvtk.vtkVersion.GetVTKVersion()

##############################################################
#  vtkCommonCore Module
#: class for callback/observer methods
Command = ccvtk.vtkCommand
#: class for LookupTable methods
LookupTable = ccvtk.vtkLookupTable
#: class for Points methods
Points = ccvtk.vtkPoints
#: class for IdTypeArray methods
IdTypeArray = ccvtk.vtkIdTypeArray
#: class for FloatArray methods
FloatArray = ccvtk.vtkFloatArray
#: class for DoubleArray methods
DoubleArray = ccvtk.vtkDoubleArray
#: class for StringArray methods
StringArray = ccvtk.vtkStringArray
#: class for UnsignedCharArray
UnsignedCharArray = ccvtk.vtkUnsignedCharArray
#: class for VTK_OBJECT
VTK_OBJECT = ccvtk.VTK_OBJECT
#: class for VTK_ID_TYPE
VTK_ID_TYPE = ccvtk.VTK_ID_TYPE
#: class for VTK_INT
VTK_INT = ccvtk.VTK_INT
#: class for VTK_DOUBLE
VTK_DOUBLE = ccvtk.VTK_DOUBLE
#: class for VTK_FLOAT
VTK_FLOAT = ccvtk.VTK_FLOAT
#: class for VTK_TEXT_LEFT
VTK_TEXT_LEFT = ccvtk.VTK_TEXT_LEFT
#: class for VTK_TEXT_RIGHT
VTK_TEXT_RIGHT = ccvtk.VTK_TEXT_RIGHT
#: class for VTK_TEXT_BOTTOM
VTK_TEXT_BOTTOM = ccvtk.VTK_TEXT_BOTTOM
#: class for VTK_TEXT_TOP
VTK_TEXT_TOP = ccvtk.VTK_TEXT_TOP
#: class for VTK_TEXT_CENTERED
VTK_TEXT_CENTERED = ccvtk.VTK_TEXT_CENTERED
#: class for VTK_UNSIGNED_CHAR
VTK_UNSIGNED_CHAR = ccvtk.VTK_UNSIGNED_CHAR
#: class for VTK_UNSIGNED_INT
VTK_UNSIGNED_INT = ccvtk.VTK_UNSIGNED_INT
#: class for VTK_UNSIGNED_SHORT
VTK_UNSIGNED_SHORT = ccvtk.VTK_UNSIGNED_SHORT

##############################################################
#  vtkCommonExecutionModel Module
#: class for AlgorithmOutput
AlgorithmOutput = cemvtk.vtkAlgorithmOutput

##############################################################
#  vtkRenderingCore Module
#: class for Renderer
Renderer = rcvtk.vtkRenderer
#: class for Skybox
Skybox = rcvtk.vtkSkybox
#: class for Volume
Volume = rcvtk.vtkVolume
#: class for Actor2D
Actor2D = rcvtk.vtkActor2D
#: class for Actor
Actor = rcvtk.vtkActor
#: class for RenderWindow
RenderWindow = rcvtk.vtkRenderWindow
#: class for RenderWindowInteractor
RenderWindowInteractor = rcvtk.vtkRenderWindowInteractor
#: class for InteractorEventRecorder
InteractorEventRecorder = rcvtk.vtkInteractorEventRecorder
#: class for WindowToImageFilter
WindowToImageFilter = rcvtk.vtkWindowToImageFilter
#: class for InteractorStyle
InteractorStyle = rcvtk.vtkInteractorStyle
#: class for PropPicker
PropPicker = rcvtk.vtkPropPicker
#: class for PointPicker
PointPicker = rcvtk.vtkPointPicker
#: class for CellPicker
CellPicker = rcvtk.vtkCellPicker
#: class for WorldPointPicker
WorldPointPicker = rcvtk.vtkWorldPointPicker
#: class for HardwareSelector
HardwareSelector = rcvtk.vtkHardwareSelector
#: class for ImageActor
ImageActor = rcvtk.vtkImageActor
#: class for PolyDataMapper
PolyDataMapper = rcvtk.vtkPolyDataMapper
#: class for PolyDataMapper2D
PolyDataMapper2D = rcvtk.vtkPolyDataMapper2D
#: class for Assembly
Assembly = rcvtk.vtkAssembly
#: class for DataSetMapper
DataSetMapper = rcvtk.vtkDataSetMapper
#: class for Texture
Texture = rcvtk.vtkTexture
#: class for TexturedActor2D
TexturedActor2D = rcvtk.vtkTexturedActor2D
#: class for Follower
Follower = rcvtk.vtkFollower
#: class for TextActor
TextActor = rcvtk.vtkTextActor
#: class for TextActor3D
TextActor3D = rcvtk.vtkTextActor3D
#: class for Property2D
Property2D = rcvtk.vtkProperty2D
#: class for Camera
Camera = rcvtk.vtkCamera

##############################################################
#  vtkRenderingFreeType Module
#: class for VectorText
VectorText = rftvtk.vtkVectorText

##############################################################
#  vtkRenderingLOD Module
#: class for LODActor
LODActor = rlodvtk.vtkLODActor

##############################################################
#  vtkRenderingAnnotation Module
#: class for ScalarBarActor
ScalarBarActor = ravtk.vtkScalarBarActor

##############################################################
#  vtkRenderingOpenGL2 Module
#: class for OpenGLRenderer
OpenGLRenderer = roglvtk.vtkOpenGLRenderer
#: class for Shader
Shader = roglvtk.vtkShader
#: class for RenderPassCollection
RenderPassCollection = roglvtk.vtkRenderPassCollection
#: class for DefaultRenderPass
DefaultRenderPass = roglvtk.vtkDefaultPass
#: class for SequencePass
SequencePass = roglvtk.vtkSequencePass
#: class for SSAAPass
SSAAPass = roglvtk.vtkSSAAPass
#: class for CameraPass
CameraPass = roglvtk.vtkCameraPass

##############################################################
#  vtkInteractionStyle Module
#: class for InteractorStyleImage
InteractorStyleImage = isvtk.vtkInteractorStyleImage
#: class for InteractorStyleTrackballActor
InteractorStyleTrackballActor = isvtk.vtkInteractorStyleTrackballActor
#: class for InteractorStyleTrackballCamera
InteractorStyleTrackballCamera = isvtk.vtkInteractorStyleTrackballCamera
#: class for InteractorStyleUser
InteractorStyleUser = isvtk.vtkInteractorStyleUser

##############################################################
#  vtkFiltersCore Module
#: class for CleanPolyData
CleanPolyData = fcvtk.vtkCleanPolyData
#: class for PolyDataNormals
PolyDataNormals = fcvtk.vtkPolyDataNormals
#: class for ContourFilter
ContourFilter = fcvtk.vtkContourFilter
#: class for TubeFilter
TubeFilter = fcvtk.vtkTubeFilter
#: class for Glyph3D
Glyph3D = fcvtk.vtkGlyph3D
#: class for TriangleFilter
TriangleFilter = fcvtk.vtkTriangleFilter

##############################################################
#  vtkFiltersGeneral Module
#: class for SplineFilter
SplineFilter = fgvtk.vtkSplineFilter
#: class for TransformPolyDataFilter
TransformPolyDataFilter = fgvtk.vtkTransformPolyDataFilter

##############################################################
#  vtkFiltersHybrid Module
#: class for RenderLargeImage
RenderLargeImage = fhvtk.vtkRenderLargeImage

##############################################################
#  vtkFiltersModeling Module
#: class for LoopSubdivisionFilter
LoopSubdivisionFilter = fmvtk.vtkLoopSubdivisionFilter
#: class for ButterflySubdivisionFilter
ButterflySubdivisionFilter = fmvtk.vtkButterflySubdivisionFilter
#: class for OutlineFilter
OutlineFilter = fmvtk.vtkOutlineFilter
#: class for LinearExtrusionFilter
LinearExtrusionFilter = fmvtk.vtkLinearExtrusionFilter

##############################################################
#  vtkFiltersTexture Module
#: class for TextureMapToPlane
TextureMapToPlane = ftvtk.vtkTextureMapToPlane

##############################################################
#  vtkFiltersSource Module
#: class for SphereSource
SphereSource = fsvtk.vtkSphereSource
#: class for CylinderSource
CylinderSource = fsvtk.vtkCylinderSource
#: class for ArrowSource
ArrowSource = fsvtk.vtkArrowSource
#: class for ConeSource
ConeSource = fsvtk.vtkConeSource
#: class for DiskSource
DiskSource = fsvtk.vtkDiskSource
#: class for TexturedSphereSource
TexturedSphereSource = fsvtk.vtkTexturedSphereSource
#: class for RegularPolygonSource
RegularPolygonSource = fsvtk.vtkRegularPolygonSource

##############################################################
#  vtkCommonDataModel Module
#: class for PolyData
PolyData = cdmvtk.vtkPolyData
#: class for ImageData
ImageData = cdmvtk.vtkImageData
#: class for DataObject
DataObject = cdmvtk.vtkDataObject
#: class for CellArray
CellArray = cdmvtk.vtkCellArray
#: class for PolyVertex
PolyVertex = cdmvtk.vtkPolyVertex
#: class for UnstructuredGrid
UnstructuredGrid = cdmvtk.vtkUnstructuredGrid
#: class for Polygon
Polygon = cdmvtk.vtkPolygon
#: class for DataObject
DataObject = cdmvtk.vtkDataObject
#: class for Molecule
Molecule = cdmvtk.vtkMolecule
#: class for DataSetAttributes
DataSetAttributes = cdmvtk.vtkDataSetAttributes

##############################################################
#  vtkCommonTransforms Module
#: class for Transform
Transform = ctvtk.vtkTransform

##############################################################
#  vtkCommonTransforms Module
#: class for Matrix4x4
Matrix4x4 = cmvtk.vtkMatrix4x4
#: class for Matrix3x3
Matrix3x3 = cmvtk.vtkMatrix3x3

##############################################################
#  vtkImagingCore Module
#: class for ImageFlip
ImageFlip = icvtk.vtkImageFlip
#: class for ImageReslice
ImageReslice = icvtk.vtkImageReslice
#: class for ImageMapToColors
ImageMapToColors = icvtk.vtkImageMapToColors

##############################################################
#  vtkIOImage vtkIOLegacy, vtkIOPLY, vtkIOGeometry,
# vtkIOMINC Modules
#: class for ImageReader2Factory
ImageReader2Factory = ioivtk.vtkImageReader2Factory
#: class for PNGReader
PNGReader = ioivtk.vtkPNGReader
#: class for BMPReader
BMPReader = ioivtk.vtkBMPReader
#: class for JPEGReader
JPEGReader = ioivtk.vtkJPEGReader
#: class for TIFFReader
TIFFReader = ioivtk.vtkTIFFReader
#: class for PLYReader
PLYReader = ioplyvtk.vtkPLYReader
#: class for STLReader
STLReader = iogvtk.vtkSTLReader
#: class for OBJReader
OBJReader = iogvtk.vtkOBJReader
#: class for MNIObjectReader
MNIObjectReader = iomincvtk.vtkMNIObjectReader
#: class for PolyDataReader
PolyDataReader = iolvtk.vtkPolyDataReader
#: class for XMLPolyDataReader
XMLPolyDataReader = ioxmlvtk.vtkXMLPolyDataReader
#: class for PNGWriter
PNGWriter = ioivtk.vtkPNGWriter
#: class for BMPWriter
BMPWriter = ioivtk.vtkBMPWriter
#: class for JPEGWriter
JPEGWriter = ioivtk.vtkJPEGWriter
#: class for TIFFWriter
TIFFWriter = ioivtk.vtkTIFFWriter
#: class for PLYWriter
PLYWriter = ioplyvtk.vtkPLYWriter
#: class for STLWriter
STLWriter = iogvtk.vtkSTLWriter
#: class for MNIObjectWriter
MNIObjectWriter = iomincvtk.vtkMNIObjectWriter
#: class for PolyDataWriter
PolyDataWriter = iolvtk.vtkPolyDataWriter
#: class for XMLPolyDataWriter
XMLPolyDataWriter = ioxmlvtk.vtkXMLPolyDataWriter

##############################################################
#  vtkDomainsChemistry  and vtkDomainsChemistryOpenGL2 Module
#: class for SimpleBondPerceiver
SimpleBondPerceiver = dcvtk.vtkSimpleBondPerceiver
#: class for ProteinRibbonFilter
ProteinRibbonFilter = dcvtk.vtkProteinRibbonFilter
#: class for PeriodicTable
PeriodicTable = dcvtk.vtkPeriodicTable
#: class for OpenGLMoleculeMapper
OpenGLMoleculeMapper = dcovtk.vtkOpenGLMoleculeMapper
