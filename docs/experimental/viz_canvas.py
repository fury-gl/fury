from fury import actor, window
from fury.utils import (get_actor_from_polydata, numpy_to_vtk_colors,
                        numpy_to_vtk_points, set_polydata_triangles,
                        set_polydata_vertices, set_polydata_colors)
import fury.primitive as fp
from vtk.util import numpy_support


import numpy as np
import vtk


def test_spheres_on_canvas():

    scene = window.Scene()
    showm = window.ShowManager(scene, reset_camera=False)

    # colors = 255 * np.array([
    #     [.85, .07, .21], [.56, .14, .85], [.16, .65, .20], [.95, .73, .06],
    #     [.95, .55, .05], [.62, .42, .75], [.26, .58, .85], [.24, .82, .95],
    #     [.95, .78, .25], [.85, .58, .35], [1., 1., 1.]
    # ])
    n_points = 5
    colors = 255 * np.random.rand(n_points, 3)
    # n_points = colors.shape[0]
    np.random.seed(42)
    centers = np.random.rand(n_points, 3)
    radius = np.random.rand(n_points) * 5

    polydata = vtk.vtkPolyData()

    verts, faces = fp.prim_square()

    big_verts = np.tile(verts, (centers.shape[0], 1))
    big_cents = np.repeat(centers, verts.shape[0], axis=0)

    big_verts += big_cents

    #print(big_verts)

    big_scales = np.repeat(radius, verts.shape[0], axis=0)

    #print(big_scales)

    big_verts *= big_scales[:, np.newaxis]

    #print(big_verts)

    tris = np.array([[0, 1, 2], [2, 3, 0]], dtype='i8')

    big_tris = np.tile(tris, (centers.shape[0], 1))
    shifts = np.repeat(np.arange(0, centers.shape[0] * verts.shape[0],
                                 verts.shape[0]), tris.shape[0])

    big_tris += shifts[:, np.newaxis]

    #print(big_tris)

    big_cols = np.repeat(colors, verts.shape[0], axis=0)

    #print(big_cols)

    big_centers = np.repeat(centers, verts.shape[0], axis=0)

    #print(big_centers)

    big_centers *= big_scales[:, np.newaxis]

    #print(big_centers)

    set_polydata_vertices(polydata, big_verts)
    set_polydata_triangles(polydata, big_tris)
    set_polydata_colors(polydata, big_cols)

    vtk_centers = numpy_support.numpy_to_vtk(big_centers, deep=True)
    vtk_centers.SetNumberOfComponents(3)
    vtk_centers.SetName("center")
    polydata.GetPointData().AddArray(vtk_centers)

    canvas_actor = get_actor_from_polydata(polydata)
    canvas_actor.GetProperty().BackfaceCullingOff()

    scene.add(canvas_actor)

    mapper = canvas_actor.GetMapper()

    mapper.MapDataArrayToVertexAttribute(
        "center", "center", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        "//VTK::ValuePass::Dec",
        True,
        """
        //VTK::ValuePass::Dec
        in vec3 center;

        uniform mat4 Ext_mat;

        out vec3 centeredVertexMC;
        out vec3 cameraPosition;
        out vec3 viewUp;

        """,
        False
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        "//VTK::ValuePass::Impl",
        True,
        """
        //VTK::ValuePass::Impl
        centeredVertexMC = vertexMC.xyz - center;
        float scalingFactor = 1. / abs(centeredVertexMC.x);
        centeredVertexMC *= scalingFactor;

        vec3 CameraRight_worldspace = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
        vec3 CameraUp_worldspace = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);

        vec3 vertexPosition_worldspace = center + CameraRight_worldspace * 0.5 * vertexMC.x + CameraUp_worldspace * -0.5 * vertexMC.y;
        gl_Position = MCDCMatrix * vec4(vertexPosition_worldspace, 1.);

        """,
        False
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::ValuePass::Dec",
        True,
        """
        //VTK::ValuePass::Dec
        in vec3 centeredVertexMC;
        in vec3 cameraPosition;
        in vec3 viewUp;

        uniform vec3 Ext_camPos;
        uniform vec3 Ext_viewUp;
        """,
        False
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::Light::Impl",
        True,
        """
        // Renaming variables passed from the Vertex Shader
        vec3 color = vertexColorVSOutput.rgb;
        vec3 point = centeredVertexMC;
        fragOutput0 = vec4(color, 0.7);
        /*
        // Comparing camera position from vertex shader and python
        float dist = distance(cameraPosition, Ext_camPos);
        if(dist < .0001)
            fragOutput0 = vec4(1, 0, 0, 1);
        else
            fragOutput0 = vec4(0, 1, 0, 1);


        // Comparing view up from vertex shader and python
        float dist = distance(viewUp, Ext_viewUp);
        if(dist < .0001)
            fragOutput0 = vec4(1, 0, 0, 1);
        else
            fragOutput0 = vec4(0, 1, 0, 1);
        */
        float len = length(point);
        // VTK Fake Spheres
        float radius = 1.;
        if(len > radius)
          discard;
        vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
        vec3 direction = normalize(vec3(1., 1., 1.));
        float df = max(0, dot(direction, normalizedPoint));
        float sf = pow(df, 24);
        fragOutput0 = vec4(max(df * color, sf * vec3(1)), 1);
        """,
        False
    )

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def vtk_shader_callback(caller, event, calldata=None):
        res = scene.size()
        camera = scene.GetActiveCamera()
        cam_pos = camera.GetPosition()
        foc_pnt = camera.GetFocalPoint()
        view_up = camera.GetViewUp()
        cam_light_mat = camera.GetCameraLightTransformMatrix()
        #comp_proj_mat = camera.GetCompositeProjectionTransformMatrix()
        exp_proj_mat = camera.GetExplicitProjectionTransformMatrix()
        eye_mat = camera.GetEyeTransformMatrix()
        model_mat = camera.GetModelTransformMatrix()
        model_view_mat = camera.GetModelViewTransformMatrix()
        proj_mat = camera.GetProjectionTransformMatrix(scene)
        view_mat = camera.GetViewTransformMatrix()
        mat = view_mat
        np.set_printoptions(precision=3, suppress=True)
        np_mat = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                np_mat[i, j] = mat.GetElement(i, j)
        program = calldata
        if program is not None:
            #print("\nCamera position: {}".format(cam_pos))
            #print("Focal point: {}".format(foc_pnt))
            #print("View up: {}".format(view_up))
            #print(mat)
            #print(np_mat)
            #print(np.dot(-np_mat[:3, 3], np_mat[:3, :3]))
            #a = np.array(cam_pos) - np.array(foc_pnt)
            #print(a / np.linalg.norm(a))
            #print(cam_light_mat)
            ##print(comp_proj_mat)
            #print(exp_proj_mat)
            #print(eye_mat)
            #print(model_mat)
            #print(model_view_mat)
            #print(proj_mat)
            #print(view_mat)
            program.SetUniform2f("Ext_res", res)
            program.SetUniform3f("Ext_camPos", cam_pos)
            program.SetUniform3f("Ext_focPnt", foc_pnt)
            program.SetUniform3f("Ext_viewUp", view_up)
            program.SetUniformMatrix("Ext_mat", mat)

    mapper.AddObserver(vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)

    global timer
    timer = 0

    def timer_callback(obj, event):
        global timer
        timer += 1.
        showm.render()
        scene.azimuth(2)
        #scene.elevation(5)
        #scene.roll(5)

    label = vtk.vtkOpenGLBillboardTextActor3D()
    label.SetInput("FURY Rocks!!!")
    label.SetPosition(1., 1., 1)
    label.GetTextProperty().SetFontSize(40)
    label.GetTextProperty().SetColor(.5, .5, .5)
    # TODO: Get Billboard's mapper
    #l_mapper = label.GetActors()

    #scene.add(label)
    scene.add(actor.axes())

    scene.background((1, 1, 1))

    #scene.set_camera(position=(1.5, 2.5, 15), focal_point=(1.5, 2.5, 1.5), view_up=(0, 1, 0))
    scene.set_camera(position=(1.5, 2.5, 25), focal_point=(0, 0, 0), view_up=(0, 1, 0))
    showm.initialize()
    showm.add_timer_callback(True, 100, timer_callback)
    showm.start()


def test_fireballs_on_canvas():
    scene = window.Scene()
    showm = window.ShowManager(scene)

    # colors = 255 * np.array([
    #     [.85, .07, .21], [.56, .14, .85], [.16, .65, .20], [.95, .73, .06],
    #     [.95, .55, .05], [.62, .42, .75], [.26, .58, .85], [.24, .82, .95],
    #     [.95, .78, .25], [.85, .58, .35], [1., 1., 1.]
    # ])
    colors = np.random.rand(1000000, 3) * 255
    n_points = colors.shape[0]
    np.random.seed(42)
    centers = 500 * np.random.rand(n_points, 3) - 250

    radius = .5 * np.ones(n_points)

    polydata = vtk.vtkPolyData()

    verts = np.array([[0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0]])
    verts -= np.array([0.5, 0.5, 0])

    big_verts = np.tile(verts, (centers.shape[0], 1))
    big_cents = np.repeat(centers, verts.shape[0], axis=0)

    big_verts += big_cents

    big_scales = np.repeat(radius, verts.shape[0], axis=0)

    big_verts *= big_scales[:, np.newaxis]

    tris = np.array([[0, 1, 2], [2, 3, 0]], dtype='i8')

    big_tris = np.tile(tris, (centers.shape[0], 1))
    shifts = np.repeat(np.arange(0, centers.shape[0] * verts.shape[0],
                                 verts.shape[0]), tris.shape[0])

    big_tris += shifts[:, np.newaxis]

    big_cols = np.repeat(colors, verts.shape[0], axis=0)

    big_centers = np.repeat(centers, verts.shape[0], axis=0)

    big_centers *= big_scales[:, np.newaxis]

    set_polydata_vertices(polydata, big_verts)
    set_polydata_triangles(polydata, big_tris)
    set_polydata_colors(polydata, big_cols)

    vtk_centers = numpy_support.numpy_to_vtk(big_centers, deep=True)
    vtk_centers.SetNumberOfComponents(3)
    vtk_centers.SetName("center")
    polydata.GetPointData().AddArray(vtk_centers)

    canvas_actor = get_actor_from_polydata(polydata)
    canvas_actor.GetProperty().BackfaceCullingOff()

    scene.add(canvas_actor)

    mapper = canvas_actor.GetMapper()

    mapper.MapDataArrayToVertexAttribute(
        "center", "center", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        "//VTK::ValuePass::Dec",
        True,
        """
        //VTK::ValuePass::Dec
        in vec3 center;
        out vec3 centeredVertexMC;
        """,
        False
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        "//VTK::ValuePass::Impl",
        True,
        """
        //VTK::ValuePass::Impl
        centeredVertexMC = vertexMC.xyz - center;
        float scalingFactor = 1. / abs(centeredVertexMC.x);
        centeredVertexMC *= scalingFactor;

        vec3 CameraRight_worldspace = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
        vec3 CameraUp_worldspace = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);

        vec3 vertexPosition_worldspace = center + CameraRight_worldspace * 0.5 * vertexMC.x + CameraUp_worldspace * -0.5 * vertexMC.y;
        gl_Position = MCDCMatrix * vec4(vertexPosition_worldspace, 1.);
        """,
        False
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::ValuePass::Dec",
        True,
        """
        //VTK::ValuePass::Dec
        in vec3 centeredVertexMC;
        uniform float time;

        float snoise(vec3 uv, float res) {
            const vec3 s = vec3(1e0, 1e2, 1e3);
            uv *= res;
            vec3 uv0 = floor(mod(uv, res)) * s;
            vec3 uv1 = floor(mod(uv + vec3(1.), res)) * s;
            vec3 f = fract(uv);
            f = f * f * (3. - 2. * f);
            vec4 v = vec4(uv0.x + uv0.y + uv0.z, uv1.x + uv0.y + uv0.z,
                uv0.x + uv1.y + uv0.z, uv1.x + uv1.y + uv0.z);
            vec4 r = fract(sin(v * 1e-1) * 1e3);
            float r0 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
            r = fract(sin((v + uv1.z - uv0.z) * 1e-1) * 1e3);
            float r1 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
            return mix(r0, r1, f.z) * 2. - 1.;
        }
        """,
        False
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::Light::Impl",
        True,
        """
        // Renaming variables passed from the Vertex Shader
        vec3 color = vertexColorVSOutput.rgb;
        vec3 point = centeredVertexMC;
        float len = length(point);
        float fColor = 2. - 2. * len;
        vec3 coord = vec3(atan(point.x, point.y) / 6.2832 + .5, len * .4, .5);
        for(int i = 1; i <= 7; i++) {
            float power = pow(2., float(i));
            fColor += (1.5 / power) * snoise(coord +
                vec3(0., -time * .005, time * .001), power * 16.);
        }
        if(fColor < 0) discard;
        //color = vec3(fColor);
        color *= fColor;
        fragOutput0 = vec4(color, 1.);
        """,
        False
    )

    global timer
    timer = 0

    def timer_callback(obj, event):
        global timer
        timer += 1.
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

    mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent,
                       vtk_shader_callback)

    showm.initialize()
    showm.add_timer_callback(True, 100, timer_callback)
    showm.start()



# test_spheres_on_canvas()
test_fireballs_on_canvas()