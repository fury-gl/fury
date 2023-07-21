import os
import numpy as np
from fury.actor import billboard
from fury.postprocessing import (colormap_to_texture, 
                                 window_to_texture)
from fury.shaders import (import_fury_shader,
                          compose_shader,
                          attribute_to_actor,
                          shader_apply_effects,
                          shader_custom_uniforms)
from fury.window import (ShowManager, 
                         Scene, 
                         gl_disable_depth, 
                         gl_set_additive_blending)
from matplotlib import colormaps

class EffectManager():
    """Class that manages the application of post-processing effects on actors.

    Parameters
    ----------
    manager : ShowManager
        Target manager that will render post processed actors."""
    def __init__(self, manager : ShowManager):
        self.scene = Scene()
        pos, focal, vu = manager.scene.get_camera()
        self.scene.set_camera(pos, focal, vu)
        self.scene.set_camera()
        self.on_manager = manager
        self.off_manager = ShowManager(self.scene, 
                                   size=manager.size)
        self.off_manager.window.SetOffScreenRendering(True)
        self.off_manager.initialize()

    

    def kde(self, center, points : np.ndarray, sigmas, scale = 1, opacity = 1.0, colormap = "viridis", custom_colormap : np.array = None):
        if not isinstance(sigmas, np.ndarray):
            sigmas = np.array(sigmas)
        if sigmas.shape[0] != 1 and sigmas.shape[0] != points.shape[0]:
            raise IndexError("sigmas size must be one or points size.")

        varying_dec = """
        varying float out_sigma;
        varying float out_scale;
        """

        kde_dec = import_fury_shader(os.path.join("utils", "normal_distribution.glsl"))

        kde_impl = """
        float current_kde = kde(normalizedVertexMCVSOutput*out_scale, out_sigma);
        color = vec3(current_kde);
        fragOutput0 = vec4(color, 1.0);
        """

        kde_vs_dec = """
        in float in_sigma;
        varying float out_sigma;

        in float in_scale;
        varying float out_scale;
        """


        kde_vs_impl = """
        out_sigma = in_sigma;
        out_scale = in_scale;
        """

        tex_dec = import_fury_shader(os.path.join("effects", "color_mapping.glsl"))

        tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 res_factor = vec2(res.y/res.x, 1.0);
        vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
        float intensity = texture(screenTexture, renorm_tex).r;
        color = color_mapping(intensity, colormapTexture).rgb;

        if(intensity<=0.0){
            discard;
        }else{
            fragOutput0 = vec4(color, opacity);
        }
        """

        fs_dec = compose_shader([varying_dec, kde_dec])

        # Scales parameter will be defined by the empirical rule:
        # 2*1*sima  = 68.27% of data inside the curve
        # 2*2*sigma = 95.45% of data inside the curve
        # 2*3*sigma = 99.73% of data inside the curve
        scales = 2*3.0*np.copy(sigmas)

        bill = billboard(
        points,
        (0.0,
        0.0,
        1.0),
        scales=scales,
        fs_dec=fs_dec,
        fs_impl=kde_impl,
        vs_dec=kde_vs_dec,
        vs_impl=kde_vs_impl)

        # Blending and uniforms setup
        window = self.off_manager.window
        shader_apply_effects(window, bill, gl_disable_depth)
        shader_apply_effects(window, bill, gl_set_additive_blending)

        attribute_to_actor(bill, np.repeat(sigmas, 4), "in_sigma")
        attribute_to_actor(bill, np.repeat(scales, 4), "in_scale")

        self.off_manager.scene.add(bill)

        self.off_manager.render()

        # Render to second billboard for color map post-processing.
        textured_billboard = billboard(center, scales=scale, fs_dec=tex_dec, fs_impl=tex_impl)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniform2f("res", self.off_manager.size)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniformf("opacity", opacity)

        # Disables the texture warnings
        textured_billboard.GetProperty().GlobalWarningDisplayOff() 

        if custom_colormap == None:
            cmap = colormaps[colormap]
            cmap = np.array([cmap(i) for i in np.arange(0.0, 1.0, 1/256)])
        else:
            cmap = custom_colormap

        colormap_to_texture(cmap, "colormapTexture", textured_billboard)

        def event_callback(obj, event):
            pos, focal, vu = self.on_manager.scene.get_camera()
            self.off_manager.scene.set_camera(pos, focal, vu)
            self.off_manager.scene.Modified()
            self.off_manager.render()

            window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate")

        window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate")
        
        self.on_manager.add_iren_callback(event_callback, "RenderEvent")

        return textured_billboard
    