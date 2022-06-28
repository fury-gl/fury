Week 1 - A Basic glTF Importer
==============================

.. post:: June 20 2022
   :author: Shivam Anand
   :tags: google
   :category: gsoc


What did you do during the Community Bonding Period?
----------------------------------------------------

In the community bonding period I met with my mentor Serge Koudoro, along with other mentors in FURY and gsoc contributors.
We discussed about my project and set up project timeline on github projects section. As my project (glTF Integration)
and Keyframe animations were related so we discussed on that too.

I read documentation of various glTF loading libraries (``pygltflib``, ``panda3D-gltf`` and ``pyrender``) in python and tried to understand how do they work.
I started creating a basic glTF loader, it was using dataclasses to store json data. I created polydata for each primitive in a mesh, each polydata will contain the vertices, triangles, normals.
To apply textures properly, I had to apply texture coordinates to polydata and flip the texture along X axis by 90 degrees.


What did you do this week?
--------------------------

After discussing on pros and cons of various glTF libraries we decided to use ``pygltflib`` to handle json to python dataclass conversion.
This week I reshaped PR `#600 <https://github.com/fury-gl/fury/pull/600/>`_ to use pygltflib. I also modified the code to handle multiple base textures.
While experimenting with textures, I accidentally applied normals to the polydata and discovered that it makes the surface look much smoother, however in few models it results in a darker model which reflects almost no light. So I made it optional for now to apply normals in a model.

I also created a basic fetcher (PR `#602 <https://github.com/fury-gl/fury/pull/602/>`_) to get glTF models from Khronos group's glTF sample repository.
I also made this function asynchronous, as Serge suggested me to use asyncio and aiohttp to make the API callbacks asynchronous and it decreased the downloading time of multiple models.

I am also working on exporting scene to a glTF file. Using `pygltflib` I am converting the python dataclass to a json like structure.


What is coming up next week?
----------------------------

Create a PR for the fetcher function, add tests, fix bugs and merge it by the end of the week.
Fix the colors issue in the glTF exporter.
Add texture and camera in glTF exporter and create a PR for it.


Did you get stuck anywhere?
---------------------------

* ``var: list[int]`` was causing the error ``TypeError: 'type' object is not subscriptable`` since the ability to use the [] operator on types like list was added in 3.9+. I had to modify the dataclasses and use ``Typing.List`` instead.
* Texture in actors weren't applied correctly. I tried flipping the texture image by 90 degrees using gimp and the texture worked nicely. The reason for this issue was the coordinate system of glTF texture format which has origin (0, 0) at top-left and (1, 1) in the bottom-right corner, Where as in vtkTexture we want coordinate (0, 0) and (1, 1) at bottom-left and top-right corner respectively. Here's an example of the issue:

.. image:: https://raw.githubusercontent.com/xtanion/Blog-Images/main/-1_orig.jpg
   :width: 500
   :align: center

* Praneeth told me that some models with multiple nodes and multiple meshes weren't loading correctly. The reason for it was the way SCALAR data was extracted from the binary blob, I had to change some variables in `get_accessor_data` method and everything started to work just fine. 
