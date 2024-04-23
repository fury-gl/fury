The FBO Saga - Week 1
=====================

.. post:: June 5, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc


This Past Week
--------------

As mentioned in the last week's blogpost, the goal for that week was to investigate VTK's Framebuffer Object framework.
An update on that is that indeed, VTK has one more low-level working `FBO class <https://vtk.org/doc/nightly/html/classvtkOpenGLFramebufferObject.html>`_ that can be used inside FURY, however,
they come with some issues that I will explain further below.


My Current Problems
-------------------

The problems I am having with these FBO implementations are first something related to how a FBO works, and second related to how VTK works.
In OpenGL, a custom user's FBO needs some things to be complete (usable):

1. At least one buffer should be attached. This buffer can be the color, depth or stencil buffer.
2. If no color buffer will be attached then OpenGL needs to be warned no draw or read operations will be done to that buffer. Otherwise, there should be at least one color attachment.
3. All attachments should have their memory allocated.
4. Each buffer should have the same number of samples.

My first problem relies on the third requirement. VTK's implementation of FBO requires a `vtkTextureObject <https://vtk.org/doc/nightly/html/classvtkTextureObject.html>`_
as a texture attachment. I figured out how to work with this class, however, I cannot allocate memory for it, as its methods for it, `Allocate2D <https://vtk.org/doc/nightly/html/classvtkTextureObject.html#abc91bbf9a3414bded7a132d366ca4951>`_, `Create2D <https://vtk.org/doc/nightly/html/classvtkTextureObject.html#a7e9dd67f377b7f91abd9df71e75a5f67>`_ and `Create2DFromRaw <https://vtk.org/doc/nightly/html/classvtkTextureObject.html#a0e56fe426cb0e6749cc6f2f8dbf53ed7>`_
does not seem to work. Every time I try to use them, my program stops with no error message nor nothing.
For anyone interested in what is happening exactly, below is how I my tests are implemented:

::

| color_texture = vtk.vtkTextureObject() # color texture declaration
| color_texture.Bind() # binding of the texture for operations
| color_texture.SetDataType(vtk.VTK_UNSIGNED_CHAR) # setting the datatype for unsigned char
| color_texture.SetInternalFormat(vtk.VTK_RGBA) # setting the format as RGBA
| color_texture.SetFormat(vtk.VTK_RGBA)
| color_texture.SetMinificationFilter(0) # setting the minfilter as linear
| color_texture.SetMagnificationFilter(0) # setting the magfilter as linear
|
| color_texture.Allocate2D(width, height, 4, vtk.VTK_UNSIGNED_CHAR) # here is where the code stops

In contrast, for some reason, the methods for 3D textures, `Allocate3D <https://vtk.org/doc/nightly/html/classvtkTextureObject.html#aaeefa46bd3a24bf62126512a276819d0>`_ works just fine.
I could use it as a workaround, but I do not wish to, as this just does not make sense.

My second problem relies on VTK. As VTK is a library that encapsulates some OpenGL functions in more palatable forms, it comes with some costs.
Working with FBOs is a more low-level work, that requires strict control of some OpenGL states and specific functions that would be simpler if it was the main API here.
However, some of this states and functions are all spread and implicit through VTK's complex classes and methods, which doubles the time expended to make some otherwise simple instructions,
as I first need to dig in lines and lines of VTK's documentation, and worse, the code itself.


What About Next Week?
---------------------

For this next week, I plan to investigate further on why the first problem is happening. If that is accomplished, then things will be more simple, as it will be a lot easier for my project to move forward as I will finally be able
to implement the more pythonic functions needed to finally render some kernel distributions onto my screen.

Wish me luck!
