import numpy as np

from fury.actor import billboard
from fury.actors.effect_manager import texture_to_actor, shader_custom_uniforms
from fury.window import Scene, ShowManager

width, height = (1200, 800)
cam_pos = np.array([0.0, 0.0, -1.0])

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

center = np.array([[0.0, 0.0, 20.0]])


#hardcoding the billboard to be fullscreen and centered
# scale_factor_2 = np.abs(np.linalg.norm(center[0] - cam_pos))*np.sin(np.deg2rad(scene.camera().GetViewAngle()/2.0))
# print(scale_factor_2)

tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 res_factor = vec2(res.y/res.x, 1.0);
        vec2 tex_2 = gl_FragCoord.xy/res;
        vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
        vec4 tex = texture(screenTexture, tex_2);

        fragOutput0 = vec4(tex);
        //fragOutput0 = vec4(tex_2, 0.0, 1.0);
        """

scale_factor_2 = 4

scale = scale_factor_2*np.array([[width/height, 1.0, 0.0]])

bill = billboard(center, scales=scale, colors = (1.0, 0.0, 0.0), fs_impl=tex_impl)

texture_to_actor("C:\\Users\\Lampada\\Desktop\\fraser.png", "screenTexture", bill)
shader_custom_uniforms(bill, "fragment").SetUniform2f("res", [width, height])



# def callback(obj = None, event = None):
#     pos, fp, vu = manager.scene.get_camera()
#     scale_factor_2 = np.abs(np.linalg.norm(center[0] - np.array(pos)))*np.sin(np.deg2rad(scene.camera().GetViewAngle()/2.0))
#     scale = scale_factor_2*np.array([[width/height, 1.0, 0.0]])
#     bill.SetScale(scale[0, 0], scale[0, 1], scale[0, 2])
#     bill.Modified()

# callback()

manager.scene.add(bill)
# manager.add_iren_callback(callback, "RenderEvent")

manager.start()