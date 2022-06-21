First week of coding!
=====================

.. post:: June 20 2022
   :author: Shivam Anand
   :tags: google
   :category: gsoc


What did you do during the Community Bonding Period?
----------------------------------------------------

In the community bonding period I met with my mentor Serge Koudoro, along with other mentors in FURY and gsoc contributers.
We discussed about my project and set up project timeline on github projects section. As my project (glTF Integration)
and Keyframe animations were related so we discussed on that too.

I read documentation of various glTF loading libraries (`pygltflib`, `panda3D-gltf` and `pyrender`) in python and tried to understand how do they work.
I started creating a basic glTF loader, it was using dataclasses to store json data. I created polydata for each primitive in a mesh, each polydata will contain the vertices, traingles, normals.
To apply textures properly, I had to apply texture coordinates to polydata and flip the baseTexture along X axis by 90 degrees.


What did you do this week?
--------------------------

After discussing on pros and cons of various gltf libraries we decided to use `pygltflib` to handle json to python dataclass conversion.
This week I reshaped PR `#600 <https://github.com/fury-gl/fury/pull/600/>`_ to use pygltflib. I also modified the code to handle multiple baseTextures.
While experimenting with textures, I accidently applied normals to the polydata and discovered that it makes the surface look much smoother, However in few models It results in a darker model which reflects almost no light. So I made it optional for now to apply normals in a model.

I also created a basic fetcher (PR `#602 <https://github.com/fury-gl/fury/pull/602/>`_) to get glTF models from Khronos group's gltff sample repository.
I also made this function asynchronous, as Serge suggested me to use asyncio and aiohttp to make the api callbacks asynchronous and it decreased the downloading time of multiple models.

I am also working on exporting scene to a gltf file. Using `pygltflib` I am converting the python dataclass to a json like structue.


What is coming up next week?
----------------------------

Completing exporting gltf from a scene.
TBD


Did you get stuck anywhere?
---------------------------

* `var: list[int]` was causing the error `TypeError: 'type' object is not subscriptable` since the ability to use the [] operator on types like list was added in 3.9+. I had to modify the dataclasses and use `Typing.List` insted.
* Texture in actors weren't applied correctly. I tried flipping the texture image by 90 degrees using gimp and the texture worked nicely. The reson of this issue can be seen `here <https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/images/testTexture.png>`_ .
* Praneeth told me that some models with multiple nodes and multiple meshes weren't loading correctly. The reason of it was the way SCALAR data was extracted from the binary blob, I had to change some variables in `get_accessor_data` method and everything started to work just fine. 