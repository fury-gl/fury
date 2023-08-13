import numpy as np

from fury.actor import billboard, cube, sphere
from fury.actors.effect_manager import texture_to_actor, shader_custom_uniforms, window_to_texture
from fury.shaders import shader_to_actor, shader_apply_effects
from fury.window import Scene, ShowManager, gl_disable_depth
from vtk import vtkOpenGLState

width, height = (1200, 800)
cam_pos = np.array([5.0, 7.0, -1.0])

scene = Scene()
scene.set_camera(position=(cam_pos[0], cam_pos[1], cam_pos[2]),
                 focal_point=(0.0,
                              0.0,
                              0.0),
                 view_up=(0.0, 0.0, 0.0))

manager = ShowManager(
    scene,
    "demo",
    (width,
     height))

manager.initialize()

off_scene = Scene()
off_scene.set_camera(*(scene.get_camera()))
off_scene.GetActiveCamera().SetClippingRange(0.1, 10.0)

off_manager = ShowManager(
    off_scene,
    "demo",
    (width,
     height))

off_manager.window.SetOffScreenRendering(True)

off_manager.initialize()

center = np.array([[0.0, 0.0, 0.0]]) + 0.0*np.array([cam_pos/np.linalg.norm(cam_pos)])


#hardcoding the billboard to be fullscreen and centered
# scale_factor_2 = np.abs(np.linalg.norm(center[0] - cam_pos))*np.sin(np.deg2rad(scene.camera().GetViewAngle()/2.0))
# print(scale_factor_2)

tex_impl = """
        //gl_FragDepth = gl_FragCoord.x/res.x;
        //gl_FragDepth = gl_FragCoord.z;
        fragOutput0 = vec4(0.0, 1.0, 0.0, 1.0);
        """

tex_impl_2 = """
        // Turning screen coordinates to texture coordinates
        vec2 res_factor = vec2(res.y/res.x, 1.0);
        vec2 tex_2 = gl_FragCoord.xy/res;
        vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
        //float tex = texture(screenTexture, renorm_tex).b;
        vec4 tex = texture(screenTexture, renorm_tex);

        fragOutput0 = vec4(tex.rgb, 1.0);
        //fragOutput0 = vec4(tex_2, 0.0, 1.0);
        """

scale_factor_2 = 4

scale = scale_factor_2*np.array([[width/height, 1.0, 0.0]])

bill = billboard(center, scales=scale, colors = (1.0, 0.0, 0.0), fs_impl=tex_impl)
cube_actor = cube(center) 
shader_to_actor(cube_actor, "fragment", tex_impl)
shader_custom_uniforms(cube_actor, "fragment").SetUniform2f("res", [width, height])
# shader_apply_effects(off_manager.window, cube_actor, gl_disable_depth)


sphere_actor = sphere(center + np.array([[1.0, 1.0, 1.0]]), colors = (0.0, 1.0, 0.0))
shader_to_actor(sphere_actor, "fragment", tex_impl)
shader_custom_uniforms(cube_actor, "fragment").SetUniform2f("res", [width, height])
# shader_apply_effects(off_manager.window, sphere_actor, gl_disable_depth)



bill_2 = billboard(center, scales=scale, colors = (1.0, 0.0, 0.0), fs_impl=tex_impl_2)
shader_custom_uniforms(bill_2, "fragment").SetUniform2f("res", [width, height])

manager.scene.add(bill_2)



# def callback(obj = None, event = None):
#     pos, fp, vu = manager.scene.get_camera()
#     scale_factor_2 = np.abs(np.linalg.norm(center[0] - np.array(pos)))*np.sin(np.deg2rad(scene.camera().GetViewAngle()/2.0))
#     scale = scale_factor_2*np.array([[width/height, 1.0, 0.0]])
#     bill.SetScale(scale[0, 0], scale[0, 1], scale[0, 2])
#     bill.Modified()

# callback()



off_manager.scene.add(sphere_actor)
off_manager.scene.add(cube_actor)

gl_state = vtkOpenGLState()
gl_state.vtkglDepthFunc(519)



# off_manager.scene.add(bill)
# manager.add_iren_callback(callback, "RenderEvent")

off_manager.render()
window_to_texture(off_manager.window, "screenTexture", bill_2, d_type="zbuffer")

manager.start()