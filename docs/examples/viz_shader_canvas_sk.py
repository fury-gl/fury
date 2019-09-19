import vtk
from vtk.util import numpy_support
import numpy as np
from fury import actor, window, ui
from fury.utils import numpy_to_vtk_points, numpy_to_vtk_colors, set_polydata_colors
from fury.utils import set_input, rotate, set_polydata_vertices
from fury.utils import get_actor_from_polydata, set_polydata_triangles


def rectangle2(centers, colors, use_vertices=False, size=(2, 2)):
    """Visualize one or many spheres with different colors and radii

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    """
    if np.array(colors).ndim == 1:
        colors = np.tile(colors, (len(centers), 1))

    pts = numpy_to_vtk_points(np.ascontiguousarray(centers))
    cols = numpy_to_vtk_colors(255 * np.ascontiguousarray(colors))
    cols.SetName('colors')

    polydata_centers = vtk.vtkPolyData()

    polydata_centers.SetPoints(pts)
    polydata_centers.GetPointData().AddArray(cols)
    print("NB pts: ", polydata_centers.GetNumberOfPoints())
    print("NB arrays: ", polydata_centers.GetPointData().GetNumberOfArrays())
    for i in range(polydata_centers.GetPointData().GetNumberOfArrays()):
        print("Array {0}: {1}".format(i, polydata_centers.GetPointData().GetArrayName(i)))

    for i in range(polydata_centers.GetCellData().GetNumberOfArrays()):
        print("Cell {0}: {1}".format(i, polydata_centers.GetCellData().GetArrayName(i)))

    print("Array pts: {}".format(polydata_centers.GetPoints().GetData().GetName()))

    glyph = vtk.vtkGlyph3D()
    if use_vertices:
        scale = 1
        my_polydata = vtk.vtkPolyData()
        my_vertices = np.array([[0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [1.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0]])

        my_vertices -= np.array([0.5, 0.5, 0])

        my_vertices = scale * my_vertices

        my_triangles = np.array([[0, 1, 2],
                                 [2, 3, 0]], dtype='i8')

        set_polydata_vertices(my_polydata, my_vertices)
        set_polydata_triangles(my_polydata, my_triangles)

        glyph.SetSourceData(my_polydata)
    else:
        src = vtk.vtkPlaneSource()
        src.SetXResolution(size[0])
        src.SetYResolution(size[1])
        glyph.SetSourceConnection(src.GetOutputPort())

    glyph.SetInputData(polydata_centers)
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(glyph.GetOutput())
    mapper.SetScalarModeToUsePointFieldData()

    mapper.SelectColorArray('colors')

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def rectangle(size = (1, 1)):

    X, Y = size

    # Setup four points
    points = vtk.vtkPoints()
    points.InsertNextPoint(-X/2, -Y/2, 0)
    points.InsertNextPoint(-X/2, Y/2, 0)
    points.InsertNextPoint(X/2, Y/2, 0)
    points.InsertNextPoint(X/2, -Y/2, 0)

    # Create the polygon
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)

    # Add the polygon to a list of polygons
    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)

    # Create a PolyData
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper = set_input(mapper, polygonPolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

def square(scale=1):
    my_polydata = vtk.vtkPolyData()

    my_vertices = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0]])

    my_vertices -= np.array([0.5, 0.5, 0])

    my_vertices = scale * my_vertices

    my_triangles = np.array([[0, 1, 2],
                             [2, 3, 0]], dtype='i8')

    set_polydata_vertices(my_polydata, my_vertices)
    set_polydata_triangles(my_polydata, my_triangles)

    # vertex_filter = vtk.vtkGlyph3D()
    # vertex_filter.SetInputData(my_polydata)
    # vertex_filter.Update()

    # polydata = vtk.vtkPolyData()
    # polydata.ShallowCopy(vertex_filter.GetOutput())

    actor = get_actor_from_polydata(my_polydata)
    # actor.GetProperty().SetPointSize(100)
    # actor.GetProperty().SetRenderPointsAsSpheres(True)
    return actor


def cube():
    my_polydata = vtk.vtkPolyData()

    my_vertices = np.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0]])

    my_vertices -= 0.5

    my_triangles = np.array([[0, 6, 4],
                            [0, 2, 6],
                            [0, 3, 2],
                            [0, 1, 3],
                            [2, 7, 6],
                            [2, 3, 7],
                            [4, 6, 7],
                            [4, 7, 5],
                            [0, 4, 5],
                            [0, 5, 1],
                            [1, 5, 7],
                            [1, 7, 3]], dtype='i8')

    set_polydata_vertices(my_polydata, my_vertices)
    set_polydata_triangles(my_polydata, my_triangles)
    return get_actor_from_polydata(my_polydata)


def disk():

    np.random.seed(42)
    n_points = 1
    centers = np.random.rand(n_points, 3)
    colors = 255 * np.random.rand(n_points, 3)

    vtk_points = numpy_to_vtk_points(centers)

    points_polydata = vtk.vtkPolyData()
    points_polydata.SetPoints(vtk_points)

    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(points_polydata)
    vertex_filter.Update()

    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(vertex_filter.GetOutput())

    # set_polydata_colors(polydata, colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    points_actor = vtk.vtkActor()
    points_actor.SetMapper(mapper)
    points_actor.GetProperty().SetPointSize(1000)
    points_actor.GetProperty().SetRenderPointsAsSpheres(True)

    return points_actor

scene = window.Scene()
# scene.projection('parallel')
scene.add(actor.axes())
# scene.background((1, 1, 1))
showm = window.ShowManager(scene, size=(1920, 1080), order_transparent=True, interactor_style='custom')

obj = 'square'

if obj == 'square':

    sq = square()
    sq.GetProperty().BackfaceCullingOff()
    scene.add(sq)
    mapper = sq.GetMapper()

if obj == 'rectangle':

    rec = rectangle(size=(100, 100))
    scene.add(rec)
    mapper = rec.GetMapper()

if obj == 'rectangle2':
    n_points = 3
    # centers = np.random.rand(n_points, 3)
    # print(centers)
    centers = np.array([[1., 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 2, 0]])
    colors = 255 * np.random.rand(n_points, 3)
    rec = rectangle2(centers=centers, colors=colors)
    scene.add(rec)
    mapper = rec.GetMapper()
    mapper.MapDataArrayToVertexAttribute(
        'my_centers',
        'Points',
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        -1)

if obj == 'cube':

    # rec.SetPosition(100, 0, 0)
    cu = cube()
    scene.add(cu)
    scene.background((1, 1, 1))
    # window.show(scene)
    mapper = cu.GetMapper()

if obj == 'disk':

    dis = disk()
    scene.add(dis)
    mapper = dis.GetMapper()


mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Light::Dec',
    True,
    '''
    //VTK::Light::Dec
    uniform float time;
    ''',
    False
)

import itertools
counter = itertools.count(start=1)

global timer

timer = 0

def timer_callback(obj, event):

    global timer
    timer += 1.0
    # print(timer)
    showm.render()
    # scene.azimuth(10)


@window.vtk.calldata_type(window.vtk.VTK_OBJECT)
def vtk_shader_callback(caller, event, calldata=None):
    program = calldata
    global timer
    if program is not None:
        try:
            program.SetUniformf("time", timer)
        except ValueError:
            pass


mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)


# TODO: Create Fragment Shader Canvas
mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    "//VTK::Normal::Dec",
    True,
    "//VTK::Normal::Dec\n"
    "out vec4 myVertexMC;\n"
    "in vec3 my_centers[3]; // now declare our attribute\n"
    "out vec3 my_centers_out[3];",
    False
  )
mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    "//VTK::Normal::Impl",
    True,
    "//VTK::Normal::Impl\n"
    "  myVertexMC = vertexMC;\n"
    "my_centers_out = my_centers;",
    False
  )
mapper.AddShaderReplacement(
      vtk.vtkShader.Fragment,
      "//VTK::Normal::Dec",
      True,
      "//VTK::Normal::Dec\n"
      "  varying vec4 myVertexMC;\n"
      "  varying vec3  my_centers_out[3];",
      False
  )
mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Light::Impl',
    True,
    '''
    //VTK::Light::Impl
    if (myVertexMC == vec4(0.5, 0.5, 0.5, 1.0))
        {fragOutput0 = vec4(1., 1., 1., 1.); return;}
    vec3 rColor = vec3(.9, .0, .3);
    vec3 gColor = vec3(.0, .9, .3);
    vec3 bColor = vec3(.0, .3, .9);
    vec3 yColor = vec3(.9, .9, .3);

    float tm = .2; // speed
    float vcm = 5;

    float a = sin(myVertexMC.y * vcm - time * tm) / 2.;
    float b = cos(myVertexMC.y * vcm - time * tm) / 2.;
    float c = sin(myVertexMC.y * vcm - time * tm + 3.14) / 2.;
    float d = cos(myVertexMC.y * vcm - time * tm + 3.14) / 2.;

    float div = 0.01; // default 0.01

    float e = div / abs(myVertexMC.x + a);
    float f = div / abs(myVertexMC.x + b);
    float g = div / abs(myVertexMC.x + c);
    float h = div / abs(myVertexMC.x + d);

    vec3 destColor = rColor * e + gColor * f + bColor * g + yColor * h;
    fragOutput0 = vec4(destColor, 1.);
    //fragOutput0 = vec4(1 - myVertexMC.x, 1 - myVertexMC.y, 0, 1.);
    //fragOutput0 = vec4(myVertexMC.x, 0, 0, 1.);
    //vec2 p = vertexVCVSOutput.xy; //- vec2(1.5,0.5);
    vec2 p = myVertexMC.xy;
    fragOutput0 = vec4(p, 0., 1.);
    //bc
    //float z = 1.0 - length(p)/0.05;
    //if (z < 0.0) {fragOutput0 = vec4(0., 1., 0., 1.);return;}
    //fragOutput0 = vec4(1., 0., 0., 1.);
    if (length(p - vec2(0, 0)) < 0.2) {
        fragOutput0 = vec4(1, 0., 0., 1.);

    }

    if (length(p - vec2(1, 1)) < 0.2) {
        fragOutput0 = vec4(1, 0., 0., 1.);
    }


    ''',
    False
)

showm.initialize()
# showm.add_timer_callback(True, 100, timer_callback)

showm.initialize()
showm.start()