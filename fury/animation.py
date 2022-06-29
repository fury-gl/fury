import math
import time
import numpy as np
from scipy import interpolate
from skimage import color
from fury.colormap import _rgb2lab, _lab2rgb
from fury import utils
from fury.shaders import shader_to_actor, add_shader_callback


class Interpolator(object):
    def __init__(self, keyframes):
        super(Interpolator, self).__init__()
        self.keyframes = keyframes
        self.timestamps = np.sort(np.array(list(keyframes)), axis=None)

    def update_timestamps(self):
        self.timestamps = np.sort(np.array(list(self.keyframes)), axis=None)

    def _get_nearest_smaller_timestamp(self, t):
        try:
            return self.timestamps[self.timestamps <= t].max()
        except:
            if self.timestamps is not []:
                return self.timestamps[0]
            else:
                return None

    def _get_nearest_larger_timestamp(self, t):
        try:
            return self.timestamps[self.timestamps > t].min()
        except:
            if self.timestamps is not []:
                return self.timestamps[-1]
            else:
                return None

    def _get_neighbour_timestamps(self, t):
        t1 = self._get_nearest_smaller_timestamp(t)
        t2 = self._get_nearest_larger_timestamp(t)
        return t1, t2


class StepInterpolator(Interpolator):
    """Step interpolator for keyframes.

    This is a simple step interpolator to be used for any shape of keyframes data.
    """

    def __init__(self, keyframes):
        super(StepInterpolator, self).__init__(keyframes)

    def interpolate(self, t):
        t_lower = self._get_nearest_smaller_timestamp(t)
        return self.keyframes[t_lower]


class LinearInterpolator(Interpolator):
    """Linear interpolator for keyframes.

    This is a general linear interpolator to be used for any shape of keyframes data.
    """

    def __init__(self, keyframes):
        super(LinearInterpolator, self).__init__(keyframes)

    def interpolate(self, t):
        t1 = self._get_nearest_smaller_timestamp(t)
        t2 = self._get_nearest_larger_timestamp(t)
        if t1 == t2:
            return self.keyframes[t1]
        p1 = self.keyframes[t1]
        p2 = self.keyframes[t2]
        d = p2 - p1
        dt = (t - t1) / (t2 - t1)
        return dt * d + p1


class SplineInterpolator(Interpolator):
    """N-th degree spline interpolator for keyframes.

    This is a general n-th degree spline interpolator to be used for any shape of keyframes data.
    """

    def __init__(self, keyframes, degree=3, smoothness=3):
        super(SplineInterpolator, self).__init__(keyframes)

        points = np.asarray(self.get_points())

        if len(points) < (degree + 1):
            raise ValueError(f"Minimum {degree + 1} keyframes must be set in order to use {degree}-degree spline")

        self.tck = interpolate.splprep(points.T, k=degree, full_output=1, s=smoothness)[0][0]
        self.linear_lengths = []
        for x, y in zip(points, points[1:]):
            self.linear_lengths.append(math.sqrt((x[1] - y[1]) * (x[1] - y[1]) + (x[0] - y[0]) * (x[0] - y[0])))

    def get_points(self):
        return [self.keyframes[i] for i in sorted(self.keyframes.keys())]

    def interpolate(self, t):
        t1 = self._get_nearest_smaller_timestamp(t)
        t2 = self._get_nearest_larger_timestamp(t)
        if t1 == t2:
            return self.keyframes[t1]

        dt = (t - t1) / (t2 - t1)
        mi_index = np.where(self.timestamps == t1)[0][0]

        sect = sum(self.linear_lengths[:mi_index])
        t = (sect + dt * (self.linear_lengths[mi_index])) / sum(self.linear_lengths)
        return np.array(interpolate.splev(t, self.tck))


class CubicSplineInterpolator(SplineInterpolator):
    def __init__(self, keyframes, smoothness=3):
        super(CubicSplineInterpolator, self).__init__(keyframes, degree=3, smoothness=smoothness)


class BezierInterpolator(Interpolator):
    def __init__(self, keyframes):
        super(BezierInterpolator, self).__init__(keyframes)

    def interpolate(self, t):
        ...


class LABInterpolator(Interpolator):
    """LAB interpolator for color keyframes """

    def __init__(self, keyframes):
        super().__init__(keyframes)
        self.lab_keyframes = self._initialize_lab_keyframes()

    def _initialize_lab_keyframes(self):
        lab_keyframes = {}
        for key, value in self.keyframes.items():
            lab_keyframes[key] = color.rgb2lab(value)

        return lab_keyframes

    def interpolate(self, t):

        t1, t2 = self._get_neighbour_timestamps(t)
        p1 = self.lab_keyframes[t1]
        p2 = self.lab_keyframes[t2]

        if t1 == t2:
            lab_val = p1

        else:
            d = p2 - p1
            dt = (t - t1) / (t2 - t1)
            lab_val = (dt * d + p1)

        return color.lab2rgb(lab_val)


class HSVInterpolator(Interpolator):
    """LAB interpolator for color keyframes """

    def __init__(self, keyframes):
        super().__init__(keyframes)
        self.hsv_keyframes = self._initialize_hsv_keyframes()

    def _initialize_hsv_keyframes(self):
        hsv_keyframes = {}
        for key, value in self.keyframes.items():
            hsv_keyframes[key] = color.rgb2hsv(value)

        return hsv_keyframes

    def interpolate(self, t):
        t1, t2 = self._get_neighbour_timestamps(t)

        p1 = self.hsv_keyframes[t1]
        p2 = self.hsv_keyframes[t2]

        if t1 == t2:
            hsv_val = p1

        else:
            d = p2 - p1
            dt = (t - t1) / (t2 - t1)
            hsv_val = (dt * d + p1)

        return color.hsv2rgb(hsv_val)


# Shaders for doing the animation
vertex_shader_code_decl = \
    """
    uniform float time;
    
    // calculated by CPU 
    uniform vec3 position;
    uniform vec3 scale;
    
    struct Keyframe {
        float t;
        vec3 data;  
    };
    
    struct Keyframes {
        Keyframe start;
        Keyframe end;
    };
    
    
    uniform Keyframes position_k;
    uniform Keyframes scale_k;
    uniform Keyframes color_k;

    
    out float t;
    

    
    mat4 transformation(vec3 position, vec3 scale)
    {
        return mat4(
            vec4(scale.x, 0.0, 0.0, 0.0),
            vec4(0.0, scale.y, 0.0, 0.0),
            vec4(0.0, 0.0, scale.z, 0.0),
            vec4(position, 1.0));
    }
    
    vec3 lerp(Keyframe k1, Keyframe k2){
        float t = (time - k1.t) / (k2.t - k1.t);
        return k1.data * (k2.data - k1.data) * t;
    }
    """

vertex_shader_code_impl = \
    """
    t = time;
    // vertexVCVSOutput = MCVCMatrix * vertexMC;
    vec4 a = vertexMC ;
    gl_Position = MCDCMatrix * transformation(position, scale) * a ;   
    """

fragment_shader_code_decl = \
    """
    varying float t;
    """


class Timeline:
    """Keyframe animation timeline class.

    This timeline is responsible for keyframe animations for a single or a group of models.
    It's used to handle multiple attributes and properties of Fury actors such as transformations, color, and scale.
    It also accepts custom data and interpolates them such as temperature.
    Linear interpolation is used by default to interpolate data between main keyframes.
    """

    def __init__(self, actors=None, use_shaders=False):
        self._keyframes = {}
        self._keyframes = {'position': {0: np.array([0, 0, 0])}, 'rotation': {0: np.array([0, 0, 0])},
                           'scale': {0: np.array([1, 1, 1])}, 'color': {0: np.array([0, 0, 0])}}
        self._interpolators = self._init_interpolators()

        # Handle actors while constructing the timeline.
        self._actors = []
        if actors is not None:
            if isinstance(actors, list):
                for a in actors:
                    a.vcolors = utils.colors_from_actor(a)
                    if use_shaders:
                        self._use_shader(a)
                    self._actors.append(a)
            else:
                actors.vcolors = utils.colors_from_actor(actors)
                if use_shaders:
                    self._use_shader(actors)
                self._actors = [actors]

        self.playing = False
        self.loop = False
        self.reversePlaying = False
        self._last_started_at = 0
        self._last_timestamp = 0
        self.speed = 1
        self._use_shaders = use_shaders

    def _use_shader(self, actor):
        # print(actor.GetShaderProperty().GetVertexShaderCode())

        shader_to_actor(actor, "vertex", impl_code=vertex_shader_code_impl)
        shader_to_actor(actor, "vertex", decl_code=vertex_shader_code_decl, block="prim_id")
        shader_to_actor(actor, "fragment", decl_code=fragment_shader_code_decl)

        # actor.GetShaderProperty().GetVertexCustomUniforms().SetUniformf('time', 0.0)
        # vcolors = utils.colors_from_actor(actor)
        def shader_callback(_caller, _event, calldata=None):
            program = calldata
            t = self.current_timestamp
            if program is not None:
                try:
                    program.SetUniformf("time", t)
                    program.SetUniform3f("position", self.get_position(t))
                    program.SetUniform3f("scale", self.get_scale(t))
                except ValueError:
                    print('Error')

        add_shader_callback(actor, shader_callback)

    def _init_interpolators(self):
        return {'position': LinearInterpolator(self._keyframes["position"]),
                'rotation': LinearInterpolator(self._keyframes["rotation"]),
                'scale': LinearInterpolator(self._keyframes["scale"]),
                'color': LinearInterpolator(self._keyframes["color"])}

    def play(self):
        """Play the animation"""
        if not self.playing:
            self._last_started_at = time.perf_counter() - self._last_timestamp
            self.playing = True

    def pause(self):
        """Pause the animation"""
        self._last_timestamp = self.current_timestamp
        self.playing = False

    def stop(self):
        """Stops the animation"""
        self._last_timestamp = 0
        self.playing = False

    @property
    def current_timestamp(self):
        """Get current timestamp of the animation"""
        return (time.perf_counter() - self._last_started_at) if self.playing else self._last_timestamp

    @property
    def last_timestamp(self):
        """Get the max timestamp of all keyframes"""
        return max(list(max(list(self._keyframes[i].keys()) for i in self._keyframes.keys())))

    def set_timestamp(self, t):
        """Set current timestamp of the animation"""
        if self.playing:
            self._last_started_at = time.perf_counter() - t
        else:
            self._last_timestamp = t

    def is_playing(self):
        """Get the playing status of the timeline"""
        return self.playing

    def is_stopped(self):
        """Get the stopped status of the timeline"""
        return not self.playing and not self._last_timestamp

    def is_paused(self):
        """Get the paused status of the timeline"""
        return not self.playing and self._last_timestamp

    def set_speed(self, speed):
        """Set the speed of the timeline"""
        self.speed = speed

    def get_speed(self):
        """Get the speed of the timeline"""
        return self.speed

    def translate(self, timestamp, position, control_point=None):
        self.set_custom_data('position', timestamp, position, control_point)

    def rotate(self, timestamp, quat):
        pass

    def scale(self, timestamp, scalar):
        self.set_custom_data('scale', timestamp, scalar)

    def set_color(self, timestamp, color):
        self.set_custom_data('color', timestamp, color)

    def set_keyframes(self, timestamp, keyframes):
        for key in keyframes:
            self.set_custom_data(key, timestamp, keyframes[key])

    def set_custom_data(self, attrib, timestamp, value, control_point=None):
        if attrib not in self._keyframes:
            self._keyframes[attrib] = {}
            self._interpolators[attrib] = LinearInterpolator({})

        self._keyframes[attrib][timestamp] = value

        if attrib not in self._interpolators:
            self._interpolators[attrib] = LinearInterpolator(self._keyframes[attrib])
        self._interpolators[attrib].update_timestamps()

    def get_custom_data(self, timestamp):
        pass

    def set_interpolator(self, attrib, interpolator):
        if attrib in self._keyframes:
            self._interpolators[attrib] = interpolator(self._keyframes[attrib])

    def set_position_interpolator(self, interpolator):
        self.set_interpolator('position', interpolator)

    def set_scale_interpolator(self, interpolator):
        self.set_interpolator('scale', interpolator)

    def set_color_interpolator(self, interpolator):
        self.set_interpolator('color', interpolator)

    def get_position(self, t=None):
        return self._interpolators['position'].interpolate(t)

    def get_quaternion(self, t=None):
        return self._interpolators['rotation'].interpolate(t)

    def get_scale(self, t=None):
        return self._interpolators['scale'].interpolate(t)

    def get_color(self, t=None):
        return self._interpolators['color'].interpolate(t or self.current_timestamp)

    def get_custom_attrib(self, attrib, t):
        return self._interpolators[attrib].interpolate(t)

    def add_actor(self, actor):
        if self._use_shaders:
            self._use_shader(actor)
        self._actors.append(actor)

    def get_actors(self):
        return self._actors

    def remove_actor(self, actor):
        self._actors.remove(actor)

    def update(self):
        if not self._use_shaders:
            t = self.current_timestamp
            position = self.get_position(t)
            scale = self.get_scale(t)
            col = np.clip(self.get_color(t), 0, 1) * 255
            for actor in self.get_actors():
                actor.SetPosition(*position)
                actor.SetScale(scale)

                #  heavy
                actor.vcolors[:] = col
                utils.update_actor(actor)
