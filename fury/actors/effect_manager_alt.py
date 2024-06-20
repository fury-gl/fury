import os
import numpy as np
from fury.actor import Actor, billboard                                     
from fury.io import load_image           
from fury.lib import Texture, WindowToImageFilter, numpy_support
from fury.shaders import (compose_shader,
                          shader_custom_uniforms)
from fury.utils import rgb_to_vtk
from fury.window import (RenderWindow,
                         Scene,
                         ShowManager)


WRAP_MODE_DIC = {"clamptoedge" : Texture.ClampToEdge,
                 "repeat" : Texture.Repeat,
                 "mirroredrepeat" : Texture.MirroredRepeat,
                 "clamptoborder" : Texture.ClampToBorder}

BLENDING_MODE_DIC = {"none" : 0, "replace" : 1,
                     "modulate" : 2, "add" : 3,
                     "addsigned" : 4, "interpolate" : 5,
                     "subtract" : 6}

def window_to_texture(
        window : RenderWindow,
        texture_name : str,
        target_actor : Actor,
        blending_mode : str = "None",
        wrap_mode : str = "ClampToBorder",
        border_color : tuple = (
            0.0,
            0.0,
            0.0,
            1.0),
        interpolate : bool = True,
        d_type : str = "rgb"):
    """Capture a rendered window and pass it as a texture to the given actor.
    Parameters
    ----------
    window : window.RenderWindow
        Window to be captured.
    texture_name : str
        Name of the texture to be passed to the actor.
    target_actor : Actor
        Target actor to receive the texture.
    blending_mode : str, optional
        Texture blending mode. The options are:
        1. None
        2. Replace
        3. Modulate
        4. Add
        5. AddSigned
        6. Interpolate
        7. Subtract
    wrap_mode : str, optional
        Texture wrapping mode. The options are:
        1. ClampToEdge
        2. Repeat
        3. MirroredRepeat
        4. ClampToBorder
    border_color : tuple (4, ), optional
        Texture RGBA border color.
    interpolate : bool, optional
        Texture interpolation.
    d_type : str, optional
        Texture pixel type, "rgb" or "rgba". Default is "rgb" 
    """

    windowToImageFilter = WindowToImageFilter()
    windowToImageFilter.SetInput(window)
    type_dic = {"rgb" : windowToImageFilter.SetInputBufferTypeToRGB, "rgba" : windowToImageFilter.SetInputBufferTypeToRGBA,
                "zbuffer" : windowToImageFilter.SetInputBufferTypeToZBuffer}
    type_dic[d_type.lower()]()
    windowToImageFilter.Update()

    texture = Texture()
    texture.SetMipmap(True)
    texture.SetInputConnection(windowToImageFilter.GetOutputPort())
    texture.SetBorderColor(*border_color)
    texture.SetWrap(WRAP_MODE_DIC[wrap_mode.lower()])
    texture.SetInterpolate(interpolate)
    texture.MipmapOn()
    texture.SetBlendingMode(BLENDING_MODE_DIC[blending_mode.lower()])

    target_actor.GetProperty().SetTexture(texture_name, texture)

    img = numpy_support.vtk_to_numpy(texture.GetInput().GetPointData().GetScalars())

    return img



def texture_to_actor(
        path_to_texture : str,
        texture_name : str,
        target_actor : Actor,
        blending_mode : str = "None",
        wrap_mode : str = "ClampToBorder",
        border_color : tuple = (
            0.0,
            0.0,
            0.0,
            1.0),
        interpolate : bool = True):
    """Pass an imported texture to an actor.
    Parameters
    ----------
    path_to_texture : str
        Texture image path.
    texture_name : str
        Name of the texture to be passed to the actor.
    target_actor : Actor
        Target actor to receive the texture.
    blending_mode : str
        Texture blending mode. The options are:
        1. None
        2. Replace
        3. Modulate
        4. Add
        5. AddSigned
        6. Interpolate
        7. Subtract
    wrap_mode : str
        Texture wrapping mode. The options are:
        1. ClampToEdge
        2. Repeat
        3. MirroredRepeat
        4. ClampToBorder
    border_color : tuple (4, )
        Texture RGBA border color.
    interpolate : bool
        Texture interpolation."""
    
    texture = Texture()

    colormapArray = load_image(path_to_texture)
    colormapData = rgb_to_vtk(colormapArray)

    texture.SetInputDataObject(colormapData)
    texture.SetBorderColor(*border_color)
    texture.SetWrap(WRAP_MODE_DIC[wrap_mode.lower()])
    texture.SetInterpolate(interpolate)
    texture.MipmapOn()
    texture.SetBlendingMode(BLENDING_MODE_DIC[blending_mode.lower()])

    target_actor.GetProperty().SetTexture(texture_name, texture)


def colormap_to_texture(
        colormap : np.array,
        texture_name : str,
        target_actor : Actor,
        interpolate : bool = True):
    """Convert a colormap to a texture and pass it to an actor.
    Parameters
    ----------
    colormap : np.array (N, 4) or (1, N, 4)
        RGBA color map array. The array can be two dimensional, although a three dimensional one is preferred.
    texture_name : str
        Name of the color map texture to be passed to the actor.
    target_actor : Actor
        Target actor to receive the color map texture.
    interpolate : bool
        Color map texture interpolation."""

    if len(colormap.shape) == 2:
        colormap = np.array([colormap])

    texture = Texture()

    cmap = (255*colormap).astype(np.uint8)
    cmap = rgb_to_vtk(cmap)

    texture.SetInputDataObject(cmap)
    texture.SetWrap(Texture.ClampToEdge)
    texture.SetInterpolate(interpolate)
    texture.MipmapOn()
    texture.SetBlendingMode(0)

    target_actor.GetProperty().SetTexture(texture_name, texture)

def back_converter(h : np.ndarray):
    return ((h[:, 0] + h[:, 1]/255. + h[:, 2]/65025. + h[:, 3]/16581375.)/256.0).astype(np.float32)

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
    
    def grayscale(self, actor, opacity):


        tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 tex_coords = gl_FragCoord.xy/res;
        vec4 col = texture(screenTexture, tex_coords);
        float bw = 0.2126*col.r + 0.7152*col.g + 0.0722*col.b;

        fragOutput0 = vec4(vec3(bw), u_opacity*col.a);
        """
        
        if self._n_active_effects > 0:
            self.off_manager.scene.GetActors().GetLastActor().SetVisibility(False)
        self.off_manager.scene.add(actor)
        self.off_manager.render()

        actor_pos = np.array([actor.GetCenter()])
        actor_bounds = actor.GetBounds()

        actor_scales = np.array([actor_bounds[1] - actor_bounds[0], 
                                 actor_bounds[3] - actor_bounds[2],
                                 0.0])
        
        scale = np.array([[actor_scales.max(), 
                           actor_scales.max(),
                           0.0]])

        # Render to second billboard for color map post-processing.
        textured_billboard = billboard(actor_pos, scales=scale, fs_impl=tex_impl)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniform2f("res", self.on_manager.size)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniformf("u_opacity", opacity)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniform2f("u_scale", scale[0, :2])

        # Disables the texture warnings
        textured_billboard.GetProperty().GlobalWarningDisplayOff() 

        def gray_callback(obj = None, event = None):
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
        
    def laplacian(self, actor, opacity):


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
                value += vec4(laplacian_mat[i])*col;
            }
            return value;
        }
        """

        tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 tex_coords = gl_FragCoord.xy/res;
        vec4 lapl_color = laplacian_calculator(screenTexture, tex_coords, res);

        fragOutput0 = vec4(lapl_color.rgb, u_opacity*lapl_color.a);
        """
        tex_dec = compose_shader([laplacian_operator, lapl_dec])

        if self._n_active_effects > 0:
            self.off_manager.scene.GetActors().GetLastActor().SetVisibility(False)
        self.off_manager.scene.add(actor)
        self.off_manager.render()

        actor_pos = np.array([actor.GetCenter()])
        actor_bounds = actor.GetBounds()

        actor_scales = np.array([actor_bounds[1] - actor_bounds[0], 
                                 actor_bounds[3] - actor_bounds[2],
                                 0.0])
        
        scale = np.array([[actor_scales.max(), 
                           actor_scales.max(),
                           0.0]])
        
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
    

    def gaussian_blur(self, actor, opacity):


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
                value += gauss_kernel[i]*col;
            }
            return value;
        }
        """

        tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 res_factor = vec2(res.y/res.x, 1.0);
        vec2 tex_coords = gl_FragCoord.xy/res;
        vec4 kernel_color = kernel_calculator(screenTexture, tex_coords, res);

        fragOutput0 = vec4(kernel_color.rgb, u_opacity*kernel_color.a);
        """
        tex_dec = compose_shader([gaussian_kernel, gauss_dec])

        if self._n_active_effects > 0:
            self.off_manager.scene.GetActors().GetLastActor().SetVisibility(False)
        self.off_manager.scene.add(actor)
        self.off_manager.render()

        actor_pos = np.array([actor.GetCenter()])
        actor_bounds = actor.GetBounds()

        actor_scales = np.array([actor_bounds[1] - actor_bounds[0], 
                                 actor_bounds[3] - actor_bounds[2],
                                 0.0])
        
        scale = np.array([[actor_scales.max(), 
                           actor_scales.max(),
                           0.0]])
        
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

    