import os
import numpy as np
from fury.actor import Actor, billboard
from fury.colormap import create_colormap
from fury.io import load_image
from fury.lib import Texture, WindowToImageFilter, numpy_support
from fury.shaders import (attribute_to_actor,
                          compose_shader,
                          import_fury_shader,
                          shader_apply_effects,
                          shader_custom_uniforms)
from fury.utils import rgb_to_vtk
from fury.window import (gl_disable_depth,
                         gl_set_additive_blending,
                         RenderWindow,
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
    type_dic = {"rgb" : windowToImageFilter.SetInputBufferTypeToRGB, "rgba" : windowToImageFilter.SetInputBufferTypeToRGBA}
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

    def kde(self, 
            points : np.ndarray, 
            sigmas, 
            kernel : str = "gaussian",
            opacity : float = 1.0, 
            colormap : str = "viridis", 
            custom_colormap : np.array = None):
        """Actor that displays the Kernel Density Estimation of a given set of points.
        
        Parameters
        ----------
        points : np.ndarray (N, 3)
            Array of points to be displayed.
        sigmas : np.ndarray (1, ) or (N, 1)
            Array of sigmas to be used in the KDE calculations. Must be one or one for each point.
        kernel : str, optional
            Kernel to be used for the distribution calculation. The available options are:
            * "cosine"
            * "epanechnikov"
            * "exponential"
            * "gaussian"
            * "linear"
            * "tophat"

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
        if np.min(sigmas) <= 0:
            raise ValueError("sigmas can't have zero or negative values.")

        varying_dec = """
        varying float out_sigma;
        varying float out_scale;
        """


        # converter = """
        # vec4 float_to_rgba(float value){
        #     float ival = floor( value*4294967295.0 );
        #     float r =  floor( mod(ival, 256.0) );
        #     float g =  floor( ( mod(ival, 65536.0) - r ) / 256.0 );
        #     float b =  floor( ( mod(ival, 16777216.0) - g - r ) / 65536.0 );
        #     float a =  floor( ( mod(ival, 4294967296.0) - b - g - r ) / 16777216.0 );

        #     vec4 rgba = vec4(r, g, b, a)/255.0;
        #     return rgba;
        # }
        # """

        converter = """
        vec4 float_to_rgba(float value) {
            vec4 bitEnc = vec4(1.,255.,65025.,16581375.);
            vec4 enc = bitEnc * value;
            enc = fract(enc);
            enc -= enc.yzww * vec2(1./255., 0.).xxxy;
            return enc;
        }
        """

        kde_dec = import_fury_shader(os.path.join("utils", f"{kernel.lower()}_distribution.glsl"))

        kde_dec = compose_shader([kde_dec, converter])

        kde_impl = """
        float current_kde = kde(normalizedVertexMCVSOutput*out_scale, out_sigma)/n_points;
        // color = vec3(current_kde);
        vec4 final_color = float_to_rgba(current_kde);
        fragOutput0 = vec4(final_color);

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

        # de_converter = """
        # float rgba_to_float(vec4 value){
        #     return (255.0* (value.r*1.0 + value.g*256.0 + value.b*65536.0 + value.a*16777216.0) ) / 4294967295.0;
        # }
        # """

        # de_converter = """
        # float rgba_to_float(vec4 packedRGBA) {
        #     // Convert RGBA values from [0, 1] range to 8-bit integer range [0, 255]
        #     int r = int(packedRGBA.r * 255.0);
        #     int g = int(packedRGBA.g * 255.0);
        #     int b = int(packedRGBA.b * 255.0);
        #     int a = int(packedRGBA.a * 255.0);

        #     // Combine the four 8-bit integers into a 32-bit integer
        #     int intValue = (r << 24) | (g << 16) | (b << 8) | a;

        #     // Convert the 32-bit integer back to the original float value range [0, 1]
        #     float maxValue = 4294967295.0; // 2^16 - 1
        #     return float(intValue) / maxValue;
        # }
        # """

        de_converter = """
        float rgba_to_float(vec4 v) {
            vec4 bitEnc = vec4(1.,255.,65025.,16581375.);
            vec4 bitDec = 1./bitEnc;
            return dot(v, bitDec);
        }
        """

        gaussian_kernel = """
        const float gauss_kernel[81] = {
                        0.000123, 0.000365, 0.000839, 0.001504, 0.002179, 0.002429, 0.002179, 0.001504, 0.000839,
                        0.000365, 0.001093, 0.002503, 0.004494, 0.006515, 0.007273, 0.006515, 0.004494, 0.002503,
                        0.000839, 0.002503, 0.005737, 0.010263, 0.014888, 0.016590, 0.014888, 0.010263, 0.005737,
                        0.001504, 0.004494, 0.010263, 0.018428, 0.026753, 0.029880, 0.026753, 0.018428, 0.010263,
                        0.002179, 0.006515, 0.014888, 0.026753, 0.038898, 0.043441, 0.038898, 0.026753, 0.014888,
                        0.002429, 0.007273, 0.016590, 0.029880, 0.043441, 0.048489, 0.043441, 0.029880, 0.016590,
                        0.002179, 0.006515, 0.014888, 0.026753, 0.038898, 0.043441, 0.038898, 0.026753, 0.014888,
                        0.001504, 0.004494, 0.010263, 0.018428, 0.026753, 0.029880, 0.026753, 0.018428, 0.010263,
                        0.000839, 0.002503, 0.005737, 0.010263, 0.014888, 0.016590, 0.014888, 0.010263, 0.005737};

        const float x_offsets[81] = {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                                    -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                                    -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                                    -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                                    -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                                    -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                                    -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                                    -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                                    -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
        
        const float y_offsets[81] = {-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0,
                                    -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
                                    -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
                                    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                     1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                                     2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,
                                     3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,
                                     4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0};
        """

        gauss_dec = """
        vec4 kernel_calculator(sampler2D screenTexture, vec2 tex_coords, vec2 res){
            vec4 value = vec4(0.0);
            vec4 col = vec4(0.0);
            for(int i = 0; i < 81; i++){
                col = texture(screenTexture, tex_coords + vec2(1/res.x, 1/res.y)*vec2(x_offsets[i], y_offsets[i]));
                value += gauss_kernel[i]*col;
            }
            return value;
        }
        """

        avg_filter = """
        vec4 avg_calculator(sampler2D screenTexture, vec2 tex_coords, vec2 res){
            float x_median_offsets[5] = {-1.0, 0.0, 1.0, 
                                      0.0, 0.0};
        
            const float y_median_offsets[5] = {0.0, -1.0, 0.0, 
                                        1.0,  0.0};
            vec4 value = vec4(0.0);
            vec4 col = vec4(0.0);
            for(int i = 0; i < 5; i++){
                col = texture(screenTexture, tex_coords + vec2(1/res.x, 1/res.y)*vec2(x_median_offsets[i], y_median_offsets[i]));
                value += col;
            }
            return value/5.0;
        }
        """

        map_func = """
        float map(float value, float o_min, float o_max, float new_min, float new_max) {
            return new_min + (value - o_min) * (new_max - new_min) / (o_max - o_min);
        }
        """

        tex_dec = import_fury_shader(os.path.join("effects", "color_mapping.glsl"))

        tex_dec = compose_shader([tex_dec, de_converter, map_func, gaussian_kernel, gauss_dec, avg_filter])

        tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 res_factor = vec2(res.y/res.x, 1.0);
        vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
        //vec4 intensity = texture(screenTexture, renorm_tex);
        vec4 intensity = kernel_calculator(screenTexture, renorm_tex, res);
        //float intensity = texture(screenTexture, renorm_tex).r;

        //float fintensity = intensity.r;
        float fintensity = rgba_to_float(intensity);
        fintensity = map(fintensity, min_value, max_value, 0.0, 1.0);

        if(fintensity<=0.0){
            discard;
        }else{     
            vec4 final_color = color_mapping(fintensity, colormapTexture);
            fragOutput0 = vec4(final_color.rgb, u_opacity*final_color.a);
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
        shader_custom_uniforms(bill, "fragment").SetUniformf("n_points", points.shape[0])

        if self._n_active_effects > 0:
            self.off_manager.scene.GetActors().GetLastActor().SetVisibility(False)
        self.off_manager.scene.add(bill)

        bill_bounds = bill.GetBounds()
        max_sigma = 2*4.0*np.max(sigmas)

        actor_scales = np.array([[bill_bounds[1] - bill_bounds[0] + center_of_mass[0] + max_sigma, 
                                 bill_bounds[3] - bill_bounds[2] + center_of_mass[1] + max_sigma,
                                 0.0]])
        
        res = self.off_manager.size
        
        scale = actor_scales.max()*np.array([[res[0]/res[1], 
                                              1.0,
                                              0.0]])

        # Render to second billboard for color map post-processing.
        textured_billboard = billboard(np.array([center_of_mass]), scales=scale, fs_dec=tex_dec, fs_impl=tex_impl)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniform2f("res", res)
        shader_custom_uniforms(textured_billboard, "fragment").SetUniformf("u_opacity", opacity)
    
        # Disables the texture warnings
        textured_billboard.GetProperty().GlobalWarningDisplayOff() 

        if custom_colormap == None:
            cmap = create_colormap(np.arange(0.0, 1.0, (1/points.shape[0])), colormap)
        else:
            cmap = custom_colormap

        colormap_to_texture(cmap, "colormapTexture", textured_billboard)

        def kde_callback(obj = None, event = None):
            cam_params = self.on_manager.scene.get_camera()
            self.off_manager.scene.set_camera(*cam_params)
            self.off_manager.scene.Modified()
            shader_apply_effects(window, bill, gl_disable_depth)
            shader_apply_effects(window, bill, gl_set_additive_blending)
            self.off_manager.render()

            img = window_to_texture(
            self.off_manager.window,
            "screenTexture",
            textured_billboard,
            border_color = (0.0, 0.0, 0.0, 0.0),
            blending_mode="Interpolate",
            d_type = "rgba")

            converted_img = back_converter(img)
            converted_img = converted_img[converted_img != 0.0]
            
            avg = np.average(converted_img)
            min_value = np.min(converted_img)
            low_v = converted_img[converted_img <= avg].shape[0]
            high_v = converted_img[converted_img > avg].shape[0]
            max_value_2 = avg + (avg - min_value)*(high_v/low_v)
            # max_value = np.max(converted_img)
            # print(min_value, max_value, max_value_2)
            # print(converted_img[converted_img <= max_value_2].shape[0], converted_img[converted_img > max_value_2].shape[0])
            shader_custom_uniforms(textured_billboard, "fragment").SetUniformf("min_value", min_value)
            shader_custom_uniforms(textured_billboard, "fragment").SetUniformf("max_value", max_value_2)

        # Initialization
        kde_callback()
        
        callback_id = self.on_manager.add_iren_callback(kde_callback, "RenderEvent")

        self._active_effects[textured_billboard] = callback_id
        self._n_active_effects += 1

        return textured_billboard
    
    def grayscale(self, actor, opacity):


        tex_impl = """
        // Turning screen coordinates to texture coordinates
        vec2 res_factor = vec2(res.y/res.x, 1.0);
        vec2 scale_factor = vec2(u_scale);
        vec2 renorm_tex = scale_factor*res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
        vec4 col = texture(screenTexture, renorm_tex);
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
        vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
        vec4 kernel_color = kernel_calculator(screenTexture, renorm_tex, res);

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

    