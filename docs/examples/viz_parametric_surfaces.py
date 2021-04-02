"""
Parametric surfaces
================================

This is a simple demonstration of how users can create and render parametric
surfaces by using the line actor and add colormaps to them too.

Parametric surfaces visualized in this tutorial -
Möbius strip
Klein bottle
Roman surface
Boy's surface
"""

###############################################################################
# Importing necessary modules

from fury import window, actor, ui, colormap
import numpy as np

###############################################################################
# Function that returns the parametric surface specified by the user.


def parametric_func(name, colors=(1, 1, 1), coords=[0, 0, 0],
                    opacity=1, scale=1.0, cmap=False, cmap_name=None):
    """
    Parameters
    ----------
    name : string - "mobius_strip" or "kleins_bottle" or "roman_surface"
        or "boys_surface". Determines the parametric surface to be returned.
    colors : ndarray (3) or tuple (3)
        Determines the color of the surface. (default : white)
    coords : ndarray (3)
        Determines the coordinates where the parametric surface will be
        rendered. (default : [0, 0, 0])
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque). (default : 1)
    scale : float, optional
        Determines the size of the parametric surface returned. (default : 1)
    cmap : bool, optional
        Set it to true if colormap is being mapped to the surface.
    cmap_name : string, optional
        Stores the name of the colormap specified by the user, cmap should be
        set to True for the colormap to be rendered.

    Returns
    -------
    vtkActor

    """
    def sin(x):
        return np.sin(x)

    def cos(x):
        return np.cos(x)

    npoints = 400

    if (name == "mobius_strip"):
        u = np.linspace(0, 2*np.pi, npoints)
        v = np.linspace(-1, 1, npoints)
        u, v = np.meshgrid(u, v)
        u = u.reshape(-1)
        v = v.reshape(-1)

        x = coords[0] + scale*(1 + v/2 * cos(u/2)) * cos(u)
        y = coords[1] + scale*(1 + v/2 * cos(u/2)) * sin(u)
        z = coords[2] + scale*v/2 * sin(u/2)

        xyz = np.array([(a, b, c) for (a, b, c) in zip(x, y, z)])
        if cmap:
            v = np.copy(z)
            v /= np.max(np.abs(v), axis=0)
            colors = colormap.create_colormap(v, name=cmap_name)
        return(actor.line([xyz], colors=colors, opacity=opacity, linewidth=3))

    elif (name == "klein_bottle"):
        u = np.linspace(0, np.pi, npoints)
        v = np.linspace(0, 2*np.pi, npoints)
        u, v = np.meshgrid(u, v)
        u = u.reshape(-1)
        v = v.reshape(-1)

        x = coords[0] + scale*(-2/15*cos(u)*(3*cos(v) - 30*sin(u) +
                               90*cos(u)**4*sin(u) - 60*cos(u)**6*sin(u) +
                               5*cos(u)*cos(v)*sin(u)))
        y = coords[1] + scale*(-1/15*sin(u)*(3*cos(v) - 3*cos(u)**2*cos(v)
                               - 48*cos(u)**4*cos(v) + 48*cos(u)**6*cos(v)
                               - 60*sin(u) + 5*cos(u)*cos(v)*sin(u) -
                               5*cos(u)**3*cos(v)*sin(u) -
                               80*cos(u)**5*cos(v)*sin(u) +
                               80*cos(u)**7*cos(v)*sin(u)))
        z = coords[2] + scale*(2/15*(3 + 5*cos(u)*sin(u))*sin(v))
        if cmap:
            v = np.copy(y)
            v /= np.max(np.abs(v), axis=0)
            colors = colormap.create_colormap(v, name=cmap_name)
        xyz = np.array([(a, b, c) for (a, b, c) in zip(x, y, z)])
        return(actor.line([xyz], colors=colors, opacity=opacity, linewidth=3))

    elif (name == 'roman_surface'):
        u = np.linspace(0, np.pi/2, npoints)
        v = np.linspace(0, 2*np.pi, npoints)
        u, v = np.meshgrid(u, v)
        u = u.reshape(-1)
        v = v.reshape(-1)

        a = 1
        x = coords[0] + scale*a*cos(u)*sin(u)*sin(v)
        y = coords[1] + scale*a*cos(u)*sin(u)*cos(v)
        z = coords[2] + scale*a*cos(u)**2*cos(v)*sin(v)

        if cmap:
            v = np.copy(z)
            v /= np.max(np.abs(v), axis=0)
            colors = colormap.create_colormap(v, name=cmap_name)
        xyz = np.array([(a, b, c) for (a, b, c) in zip(x, y, z)])
        return(actor.line([xyz], colors=colors, opacity=opacity, linewidth=3))

    elif (name == 'boys_surface'):
        u = np.linspace(-np.pi/2, np.pi/2, npoints)
        v = np.linspace(0, np.pi, npoints)
        u, v = np.meshgrid(u, v)
        u = u.reshape(-1)
        v = v.reshape(-1)

        x = coords[0] + scale*((2**0.5*cos(v)**2*cos(2*u) +
                               cos(u)*sin(2*v))/(2 -
                               2**0.5*sin(3*u)*sin(2*v)))
        y = coords[1] + scale*((2**0.5*cos(v)**2*sin(2*u) -
                               sin(u)*sin(2*v))/(2 -
                               2**0.5*sin(3*u)*sin(2*v)))
        z = coords[2] + scale*(3*cos(v)**2/(2 - 2**0.5*sin(3*u)*sin(2*v)))

        if cmap:
            v = np.copy(z)
            v /= np.max(np.abs(v), axis=0)
            colors = colormap.create_colormap(v, name=cmap_name)
        xyz = np.array([(a, b, c) for (a, b, c) in zip(x, y, z)])
        return(actor.line([xyz], colors=colors, opacity=opacity, linewidth=3))

###############################################################################
# Creating a scene object and configuring the camera's position


scene = window.Scene()
scene.zoom(5.5)
scene.set_camera(position=(-10, 10, -150), focal_point=(0.0, 0.0, 0.0),
                 view_up=(0.0, 0.0, 0.0))

showm = window.ShowManager(scene,
                           size=(600, 600), reset_camera=True,
                           order_transparent=True)


###############################################################################
# Generating the surfaces and adding them to the scene
klein = parametric_func("klein_bottle", coords=[4.5, -2, 0], opacity=0.6,
                        cmap=True, cmap_name='winter')
scene.add(klein)

roman = parametric_func("roman_surface", coords=[1.5, 0, 0], scale=1.1,
                        colors=(1, 0, 0))
scene.add(roman)

mobius = parametric_func("boys_surface", coords=[-1.5, 0, -2], cmap=True,
                         cmap_name='spring')
scene.add(mobius)

boys = parametric_func("mobius_strip", coords=[-4.5, 0, 0], scale=0.7)
scene.add(boys)

tb5 = ui.TextBlock2D(bold=True, font_size=20, position=(200, 470))
tb5.message = "Some Parametric Objects"
scene.add(tb5)

interactive = False
if interactive:
    window.show(scene, size=(600, 600))
window.record(showm.scene, size=(600, 600),
              out_path="viz_parametric_surfaces.png")
