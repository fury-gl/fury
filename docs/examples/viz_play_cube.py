"""
=======================================================
Play a video in the 3D world
=======================================================

The goal of this demo is to show how to visualize a video
on a cube by updating a texture.
"""

import vtk
from fury import window
from fury.lib import Texture, PolyDataMapper, Actor, JPEGReader
# vtkPlaneSource to be added here


class TexturedCube:
    """
    A class to represent a textured cube.
    """

    def __init__(
            self,
            negx: str,
            negy: str,
            negz: str,
            posx: str,
            posy: str,
            posz: str
    ):
        """
        Initializes a TexturedCube object.

        Args:
            negx (str): Path to the negative X-axis texture file.
            negy (str): Path to the negative Y-axis texture file.
            negz (str): Path to the negative Z-axis texture file.
            posx (str): Path to the positive X-axis texture file.
            posy (str): Path to the positive Y-axis texture file.
            posz (str): Path to the positive Z-axis texture file.
        """

        self.negx = negx
        self.negy = negy
        self.negz = negz
        self.posx = posx
        self.posy = posy
        self.posz = posz

        self.planes = [vtk.vtkPlaneSource() for _ in range(6)]

        self.plane_centers = [
            (0, 0.5, 0),
            (0, 0, 0.5),
            (0, 1, 0.5),
            (0, 0.5, 1),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
        ]

        self.plane_normals = [
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 0),
            (1, 0, 0),
        ]

        for plane, center, normal in zip(
            self.planes,
            self.plane_centers,
            self.plane_normals
        ):
            plane.SetCenter(*center)
            plane.SetNormal(*normal)

        self.texture_filenames = [negx, negy, negz, posx, posy, posz]

        self.textures = [Texture() for _ in self.texture_filenames]
        self.texture_readers = [JPEGReader() for _ in self.texture_filenames]

        for filename, reader, texture in zip(
            self.texture_filenames,
            self.texture_readers,
            self.textures
        ):
            reader.SetFileName(filename)
            reader.Update()
            texture.SetInputConnection(reader.GetOutputPort())

        self.mappers = [PolyDataMapper() for _ in self.planes]
        self.actors = [Actor() for _ in self.planes]

        for mapper, actor, plane, texture in zip(
            self.mappers,
            self.actors,
            self.planes,
            self.textures
        ):
            mapper.SetInputConnection(plane.GetOutputPort())
            actor.SetMapper(mapper)
            actor.SetTexture(texture)

    def visualize(self):
        """
        Visualizes the textured cube using Fury.
        """

        scene = window.Scene()
        for actor in self.actors:
            scene.add(actor)

        show_manager = window.ShowManager(
            scene,
            size=(1280, 720),
            reset_camera=False
        )
        show_manager.start()


# Example usage
cube = TexturedCube(
    "negx.jpg", "negy.jpg", "negz.jpg", "posx.jpg", "posy.jpg", "posz.jpg"
)
cube.visualize()
