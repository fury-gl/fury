import os
import numpy as np
from fury.actor import billboard
from fury.colormap import create_colormap
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

class EffectManager():
    """Class that manages the application of post-processing effects on actors.

    Parameters
    ----------
    manager : ShowManager
        Target manager that will render post processed actors."""
    def __init__(self, manager : ShowManager):
        self.scene = Scene()
        cam_params = manager.scene.get_camera()
        self.scene.set_camera(*cam_params)
        self.on_manager = manager
        self.off_manager = ShowManager(self.scene, 
                                   size=manager.size)
        self.off_manager.window.SetOffScreenRendering(True)
        self.off_manager.initialize()
        self._n_active_effects = 0
        self._active_effects = {}

    

    def kde(self, 
            points : np.ndarray, 
            sigmas, scale = 1.0, 
            opacity = 1.0, 
            colormap = "viridis", 
            custom_colormap : np.array = None):
        """Actor that displays the Kernel Density Estimation of a given set of points.
        
        Parameters
        ----------
        points : np.ndarray (N, 3)
            Array of points to be displayed.
        sigmas : np.ndarray (1, ) or (N, 1)
            Array of sigmas to be used in the KDE calculations. Must be one or one for each point.
        scale : float, optional
            Scale of the actor.
        opacity : float, optional
            Opacity of the actor.
        colormap : str, optional.
            Colormap matplotlib name for the KDE rendering. Default is "viridis".
        custom_colormap : np.ndarray (N, 4), optional
            Custom colormap for the KDE rendering. Default is none which means no 
            custom colormap is desired. If passed, will overwrite matplotlib colormap 
            chosen in the previous parameter.
            
        Returns
        -------
        textured_billboard : actor.Actor
            KDE rendering actor."""
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
            fragOutput0 = vec4(color, u_opacity);
        }
        """

        fs_dec = compose_shader([varying_dec, kde_dec])

        # Scales parameter will be defined by the empirical rule:
        # 1*sima radius = 68.27% of data inside the curve
        # 2*sigma radius = 95.45% of data inside the curve
        # 3*sigma radius = 99.73% of data inside the curve
        scales = 2*3.0*np.copy(sigmas)

        center_of_mass = np.average(points, axis = 0)
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

        if self._n_active_effects > 0:
            self.off_manager.scene.GetActors().GetLastActor().SetVisibility(False)
        self.off_manager.scene.add(bill)
        self.off_manager.render()

        # Render to second billboard for color map post-processing.
        textured_billboard = billboard(np.array([center_of_mass]), scales=scale, fs_dec=tex_dec, fs_impl=tex_impl)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniform2f("res", self.off_manager.size)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniformf("u_opacity", opacity)

        # Disables the texture warnings
        textured_billboard.GetProperty().GlobalWarningDisplayOff() 

        if custom_colormap == None:
            cmap = create_colormap(np.arange(0.0, 1.0, 1/256), colormap)
        else:
            cmap = custom_colormap

        colormap_to_texture(cmap, "colormapTexture", textured_billboard)

        def kde_callback(obj, event):
            cam_params = self.on_manager.scene.get_camera()
            self.off_manager.scene.set_camera(*cam_params)
            self.off_manager.scene.Modified()
            shader_apply_effects(window, bill, gl_disable_depth)
            shader_apply_effects(window, bill, gl_set_additive_blending)
            self.off_manager.render()

            window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate",
            d_type = "rgba")

        # Initialization
        window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate",
            d_type = "rgba")
        
        callback_id = self.on_manager.add_iren_callback(kde_callback, "RenderEvent")

        self._active_effects[textured_billboard] = callback_id
        self._n_active_effects += 1

        return textured_billboard
    
    def grayscale(self, actor, scale, opacity):


        tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 res_factor = vec2(res.y/res.x, 1.0);
        vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
        vec4 col = texture(screenTexture, renorm_tex);
        col.a = col.a*int(vec3(0.0) != col.rgb) + 0.0*int(vec3(0.0) == col.rgb);
        float bw = 0.2126*col.r + 0.7152*col.g + 0.0722*col.b;

        fragOutput0 = vec4(vec3(bw), u_opacity*col.a);
        """
        actor_pos = np.array([actor.GetPosition()])
        
        if self._n_active_effects > 0:
            self.off_manager.scene.GetActors().GetLastActor().SetVisibility(False)
        self.off_manager.scene.add(actor)
        self.off_manager.render()

        # Render to second billboard for color map post-processing.
        textured_billboard = billboard(actor_pos, scales=scale, fs_impl=tex_impl)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniform2f("res", self.off_manager.size)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniformf("u_opacity", opacity)

        # Disables the texture warnings
        textured_billboard.GetProperty().GlobalWarningDisplayOff() 

        def gray_callback(obj, event):
            actor.SetVisibility(True)
            pos, focal, vu = self.on_manager.scene.get_camera()
            self.off_manager.scene.set_camera(pos, focal, vu)
            self.off_manager.render()

            window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate",
            d_type = "rgba")

            actor.SetVisibility(False)
            actor.Modified()
            

        # Initialization
        window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate",
            d_type = "rgba")
        
        callback_id = self.on_manager.add_iren_callback(gray_callback, "RenderEvent")

        self._active_effects[textured_billboard] = callback_id
        self._n_active_effects += 1

        return textured_billboard
        
    def laplacian(self, actor, scale, opacity):


        laplacian_operator = """
        const float laplacian_mat[3*3] = {0.0, 1.0, 0.0,
                                          1.0,-4.0, 1.0,
                                          0.0, 1.0, 0.0};

        const float x_offsets[3*3] = {-1.0, 0.0, 1.0, 
                                      -1.0, 0.0, 1.0,
                                      -1.0, 0.0, 1.0};
        
        const float y_offsets[3*3] = {-1.0, -1.0, -1.0, 
                                       0.0,  0.0,  0.0,
                                       1.0,  1.0,  1.0};
        """

        lapl_dec = """
        vec4 laplacian_calculator(sampler2D screenTexture, vec2 tex_coords, vec2 res){
            vec4 value = vec4(0.0);
            vec4 col = vec4(0.0);
            for(int i = 0; i < 9; i++){
                col = texture(screenTexture, tex_coords + vec2(1/res.x, 1/res.y)*vec2(x_offsets[i], y_offsets[i]));
                col.a = col.a*int(vec3(0.0) != col.rgb) + 0.0*int(vec3(0.0) == col.rgb); 
                value += vec4(laplacian_mat[i])*col;
            }
            return value;
        }
        """

        tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 res_factor = vec2(res.y/res.x, 1.0);
        vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
        vec4 lapl_color = laplacian_calculator(screenTexture, renorm_tex, res);

        fragOutput0 = vec4(lapl_color.rgb, u_opacity*lapl_color.a);
        """
        tex_dec = compose_shader([laplacian_operator, lapl_dec])

        if self._n_active_effects > 0:
            self.off_manager.scene.GetActors().GetLastActor().SetVisibility(False)
        self.off_manager.scene.add(actor)
        self.off_manager.render()

        actor_pos = np.array([actor.GetPosition()])

        # Render to second billboard for color map post-processing.
        textured_billboard = billboard(actor_pos, scales=scale, fs_dec=tex_dec, fs_impl=tex_impl)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniform2f("res", self.off_manager.size)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniformf("u_opacity", opacity)

        # Disables the texture warnings
        textured_billboard.GetProperty().GlobalWarningDisplayOff() 

        def laplacian_callback(obj, event):
            actor.SetVisibility(True)
            pos, focal, vu = self.on_manager.scene.get_camera()
            self.off_manager.scene.set_camera(pos, focal, vu)
            self.off_manager.render()

            window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate",
            d_type = "rgba")

            actor.SetVisibility(False)
            actor.Modified()

        # Initialization
        window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate",
            d_type = "rgba")
        
        callback_id = self.on_manager.add_iren_callback(laplacian_callback, "RenderEvent")

        self._active_effects[textured_billboard] = callback_id
        self._n_active_effects += 1

        return textured_billboard
    

    def gaussian_blur(self, actor, scale, opacity):


        gaussian_kernel = """
        const float gauss_kernel[3*3] = {1/16.0, 1/8, 1/16.0,
                                          1/8.0, 1/4.0, 1/8.0,
                                          1/16.0, 1/8.0, 1/16.0};

        const float x_offsets[3*3] = {-1.0, 0.0, 1.0, 
                                      -1.0, 0.0, 1.0,
                                      -1.0, 0.0, 1.0};
        
        const float y_offsets[3*3] = {-1.0, -1.0, -1.0, 
                                       0.0,  0.0,  0.0,
                                       1.0,  1.0,  1.0};
        """

        gauss_dec = """
        vec4 kernel_calculator(sampler2D screenTexture, vec2 tex_coords, vec2 res){
            vec4 value = vec4(0.0);
            vec4 col = vec4(0.0);
            for(int i = 0; i < 9; i++){
                col = texture(screenTexture, tex_coords + vec2(1/res.x, 1/res.y)*vec2(x_offsets[i], y_offsets[i]));
                col.a = col.a*int(vec3(0.0) != col.rgb) + 0.0*int(vec3(0.0) == col.rgb); 
                value += gauss_kernel[i]*col;
            }
            return value;
        }
        """

        tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 res_factor = vec2(res.y/res.x, 1.0);
        vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
        vec4 kernel_color = kernel_calculator(screenTexture, renorm_tex, res);

        
        fragOutput0 = vec4(kernel_color.rgb, u_opacity*kernel_color.a);
        """
        tex_dec = compose_shader([gaussian_kernel, gauss_dec])

        if self._n_active_effects > 0:
            self.off_manager.scene.GetActors().GetLastActor().SetVisibility(False)
        self.off_manager.scene.add(actor)
        self.off_manager.render()

        actor_pos = np.array([actor.GetPosition()])

        # Render to second billboard for color map post-processing.
        textured_billboard = billboard(actor_pos, scales=scale, fs_dec=tex_dec, fs_impl=tex_impl)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniform2f("res", self.off_manager.size)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniformf("u_opacity", opacity)


        # Disables the texture warnings
        textured_billboard.GetProperty().GlobalWarningDisplayOff() 

        def kernel_callback(obj, event):
            actor.SetVisibility(True)
            pos, focal, vu = self.on_manager.scene.get_camera()
            self.off_manager.scene.set_camera(pos, focal, vu)
            self.off_manager.render()

            window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate",
            border_color=(0.0, 0.0, 0.0, 0.0),
            d_type = "rgba")

            actor.SetVisibility(False)
            actor.Modified()
            

        # Initialization
        window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            blending_mode="Interpolate",
            border_color=(0.0, 0.0, 0.0, 0.0),
            d_type = "rgba")
        
        
        callback_id = self.on_manager.add_iren_callback(kernel_callback, "RenderEvent")

        self._active_effects[textured_billboard] = callback_id
        self._n_active_effects += 1

        return textured_billboard
    

    def remove_effect(self, effect_actor):
        """Remove an existing effect from the effects manager. 
        Beware that the effect and the actor will be removed from the rendering pipeline 
        and shall not work after this action.

        Parameters
        ----------
        effect_actor : actor.Actor
            Actor of effect to be removed.
        """
        if self._n_active_effects > 0:
            self.on_manager.iren.RemoveObserver(self._active_effects[effect_actor])
            self.on_manager.scene.RemoveActor(effect_actor)
            self.off_manager.scene.RemoveActor(effect_actor)
            self._active_effects.pop(effect_actor)
            self._n_active_effects -= 1
        else:
            raise IndexError("Manager has no active effects.")

    