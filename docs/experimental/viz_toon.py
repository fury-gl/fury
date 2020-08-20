import numpy as np
from fury import window, actor, ui, io, utils
from fury.data.fetcher import fetch_viz_models, read_viz_models
import vtk

fetch_viz_models()
dragon = read_viz_models('dragon.obj')
dragon = io.load_polydata(dragon)
dragon = utils.get_polymapper_from_polydata(dragon)
dragon = utils.get_actor_from_polymapper(dragon)

dragon.GetProperty().SetDiffuse(0.7)

mapper = dragon.GetMapper()

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Light::Impl',
    True,
    """
    //VTK::Light::Impl

    vec4 color;

    float ll = length(diffuse);
    if (ll > 0.95)
        color = vec4(1.0,0.5,0.5,1.0);
    else if (ll > 0.5)
        color = vec4(0.6,0.3,0.3,1.0);
    else if (ll > 0.25)
        color = vec4(0.4,0.2,0.2,1.0);
    else
        color = vec4(0.2,0.1,0.1,1.0);

    fragOutput0 = color;
    """,
    False
)

scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
scene.add(dragon)

window.show(scene)
