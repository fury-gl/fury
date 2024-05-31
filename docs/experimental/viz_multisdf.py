import numpy as np
import vtk
from vtk.util import numpy_support

from fury import actor, window
import fury.primitive as fp
import fury.shaders as fs
from fury.utils import get_actor_from_primitive

verts, faces = fp.prim_box()

centers = np.array([[0, 0, 0], [8, 0, 0]])
repeated = fp.repeat_primitive(verts, faces, centers=centers, scales=5)

rep_verts, rep_faces, rep_colors, rep_centers = repeated
sdfactor = get_actor_from_primitive(rep_verts, rep_faces)

vtk_center = numpy_support.numpy_to_vtk(rep_centers)
vtk_center.SetNumberOfComponents(3)
vtk_center.SetName('center')
sdfactor.GetMapper().GetInput().GetPointData().AddArray(vtk_center)

vs_dec_code = fs.load('sdf_dec.vert')
vs_impl_code = fs.load('sdf_impl.vert')
fs_dec_code = fs.load('sdfmulti_dec.frag')
fs_impl_code = fs.load('sdfmulti_impl.frag')

mapper = sdfactor.GetMapper()
mapper.MapDataArrayToVertexAttribute(
    'center', 'center', vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex, '//VTK::ValuePass::Dec', True, vs_dec_code, False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex, '//VTK::ValuePass::Impl', True, vs_impl_code, False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment, '//VTK::ValuePass::Dec', True, fs_dec_code, False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment, '//VTK::Light::Impl', True, fs_impl_code, False
)

scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
centers = np.array([[0, 0, 0]])


global timer
timer = 0


def timer_callback(obj, event):
    global timer
    timer += 1.0
    showm.render()


@window.vtk.calldata_type(window.vtk.VTK_OBJECT)
def vtk_shader_callback(caller, event, calldata=None):
    program = calldata
    global timer
    if program is not None:
        try:
            program.SetUniformf('time', timer)
        except ValueError:
            pass


mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)

showm = window.ShowManager(scene, reset_camera=False)


showm.add_timer_callback(True, 10, timer_callback)


scene.add(sdfactor)
scene.add(actor.axes())

showm.start()
