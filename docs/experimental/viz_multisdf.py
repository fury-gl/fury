from fury import actor, window
import numpy as np
import fury.primitive as fp
import fury.shaders as fs
from vtk.util import numpy_support
from fury.utils import get_actor_from_primitive
import vtk

verts, faces = fp.prim_box()

centers = np.array([[0, 0, 0]])
repeated = fp.repeat_primitive(verts, faces, centers=centers, scale=5)

rep_verts, rep_faces, rep_colors, rep_centers = repeated
sdfactor = get_actor_from_primitive(rep_verts, rep_faces)

vtk_center = numpy_support.numpy_to_vtk(rep_centers)
vtk_center.SetNumberOfComponents(3)
vtk_center.SetName("center")
sdfactor.GetMapper().GetInput().GetPointData().AddArray(vtk_center)

vs_dec_code = fs.load("sdf_dec.vert")
vs_impl_code = fs.load("sdf_impl.vert")
fs_dec_code = fs.load("sdfmulti_dec.frag")
fs_impl_code = fs.load("sdfmulti_impl.frag")

mapper = sdfactor.GetMapper()
mapper.MapDataArrayToVertexAttribute(
    "center", "center", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex, "//VTK::ValuePass::Dec", True,
    vs_dec_code, False)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex, "//VTK::ValuePass::Impl", True,
    vs_impl_code, False)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment, "//VTK::ValuePass::Dec", True,
    fs_dec_code, False)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment, "//VTK::Light::Impl", True,
    fs_impl_code, False)

scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
centers = np.array([[0, 0, 0]])
scene.add(sdfactor)

scene.add(actor.axes())
window.show(scene, size=(1920, 1080))
