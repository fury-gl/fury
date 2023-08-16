The Beginning of Everything - Week 0
====================================

.. post:: May 29, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc

So it begins...
---------------

Hello everyone, welcome to the beginning of my journey through GSoC 2023! I would like to thank everyone involved for the opportunity provided, it is an honour to be working side by side with professionals and so many experienced people from around the world.

The Community Bonding Period
----------------------------

During my community bonding period, I had the opportunity to meet my mentors and some people from the FURY team. It was a great time to learn about community guidelines and everything I will need to work with them during this summer.

The Project's Goal
------------------

Briefly explaining this project, I plan to implement a real-time Kernel Density Estimation shader inside FURY library, based on `Filipi Nascimento's WebGL implementation <https://github.com/filipinascimento/PACSExplorer/blob/782e52334a635528ec3ab4c7a4409cc88958d3ba/lib/density-gl.js>`_. KDE, or Kernel Density Estimation, is a visualization technique that provides a good macro visualization of large and complex data sets, like point clouds, well summarizing their spatial distribution in smooth areas. I really think FURY will benefit from this as a scientific library, knowing it is a computer graphics library that originated in 2018 based on the Visualization Toolkit API (VTK), and has been improving since then.

This Week's Goal
----------------

For all of this to work, the project needs one component working: the **KDE framebuffer**. As this `Khronos wiki page well explains <https://www.khronos.org/opengl/wiki/Framebuffer>`_:

"A Framebuffer is a collection of buffers that can be used as the destination for rendering. OpenGL has two kinds of framebuffers: the `Default Framebuffer <https://www.khronos.org/opengl/wiki/Default_Framebuffer>`_,
which is provided by the `OpenGL Context <https://www.khronos.org/opengl/wiki/OpenGL_Context>`_; and user-created framebuffers called `Framebuffer Objects <https://www.khronos.org/opengl/wiki/Framebuffer_Object>`_ (FBOs).
The buffers for default framebuffers are part of the context and usually represent a window or display device. The buffers for FBOs reference images from either `Textures <https://www.khronos.org/opengl/wiki/Texture>`_ or `Renderbuffers <https://www.khronos.org/opengl/wiki/Renderbuffer_Object>`_; they are never directly visible."

Which means that a framebuffer is an object that stores data related to a frame. So the goal for this week is to investigate whether VTK, the API which FURY is written on, has a framebuffer object interface, and if it has, to understand how it works and how to use it for the project.

Let's get to work!
