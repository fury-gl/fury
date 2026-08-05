import numpy as np
from fury import window, actor, ui, io, utils
from fury.data.fetcher import fetch_viz_models, read_viz_models
from fury.shaders import shader_to_actor
import vtk

fetch_viz_models()
dragon = read_viz_models('dragon.obj')
dragon = io.load_polydata(dragon)
dragon = utils.get_polymapper_from_polydata(dragon)
dragon = utils.get_actor_from_polymapper(dragon)

fragment_dec = \
    """
    uniform float intensity;
    uniform float ambient;
    uniform float diffuse;

    vec3 CoolColor = vec3(0, 0, 1.0);
    float DiffuseCool = 0.25;
    float DiffuseWarm = 0.25;
    vec3 LightPosition = vec3(0, 10, 4);
    vec3 SurfaceColor = vec3(1, 0.75, 0.75);
    vec3 WarmColor = vec3(1.0, 1.0, 0);
    """
fragment_impl = \
    """
    vec3 kcool    = min(CoolColor + DiffuseCool * SurfaceColor, 1.0);
    vec3 kwarm    = min(WarmColor + DiffuseWarm * SurfaceColor, 1.0);
    vec3 kfinal   = mix(kcool, kwarm, diffuse);

    fragOutput0 = vec4(min(kfinal + specular, 1.0), 1.0);
    """

shader_to_actor(dragon, "fragment", block="light", impl_code=fragment_impl,
                decl_code=fragment_dec)

scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
scene.add(dragon)

window.show(scene)
