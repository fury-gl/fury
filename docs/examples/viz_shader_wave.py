
import numpy as np
from fury import actor, window, ui
from fury.utils import numpy_to_vtk_points, set_polydata_colors
from fury.utils import set_input, rotate, set_polydata_vertices
from fury.utils import get_actor_from_polydata
import vtk

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
# scene.background((1, 1, 1))
showm = window.ShowManager(scene, size=(1920, 1080), order_transparent=True, interactor_style='custom')

rec = rectangle(size=(100, 100))
scene.add(rec)
mapper = rec.GetMapper()

# cu = cube()
# scene.add(cu)
# mapper = cu.GetMapper()

# dis = disk()
# scene.add(dis)
# mapper = dis.GetMapper()

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
    vtk.vtkShader.Fragment,
    '//VTK::Light::Impl',
    True,
    '''
    //VTK::Light::Impl
    vec3 rColor = vec3(.9, .0, .3);
    vec3 gColor = vec3(.0, .9, .3);
    vec3 bColor = vec3(.0, .3, .9);
    vec3 yColor = vec3(.9, .9, .3);

    float tm = .2; // speed
    float vcm = 5;

    float a = sin(normalVCVSOutput.y * vcm - time * tm) / 2.;
    float b = cos(normalVCVSOutput.y * vcm - time * tm) / 2.;
    float c = sin(normalVCVSOutput.y * vcm - time * tm + 3.14) / 2.;
    float d = cos(normalVCVSOutput.y * vcm - time * tm + 3.14) / 2.;

    float div = 0.1; // default 0.01

    float e = div / abs(normalVCVSOutput.x + a);
    float f = div / abs(normalVCVSOutput.x + b);
    float g = div / abs(normalVCVSOutput.x + c);
    float h = div / abs(normalVCVSOutput.x + d);

    vec3 destColor = rColor * e + gColor * f + bColor * g + yColor * h;
    fragOutput0 = vec4(destColor, 1.);
    ''',
    False
)

showm.initialize()
showm.add_timer_callback(True, 200, timer_callback)

showm.initialize()
showm.start()