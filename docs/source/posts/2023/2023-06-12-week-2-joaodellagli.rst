Week 2: The Importance of (good) Documentation
==============================================

.. post:: June 12, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc


Hello everybody, welcome to the week 2 of this project! I must admit I thought this would be simpler than it is currently being, but I forgot that when it comes to dealing with computer graphics' applications, things never are. Below, some updates on what I have been up to for this past week. 

This Last Week's Effort
-----------------------

Last week, I was facing some issues with a VTK feature essential so I could move forward with my project: Framebuffer Objects.
As described in my :doc:`last blogpost <2023-06-05-week-1-joaodellagli>`, for some reason the 2D allocation methods for it weren't working.
In a meeting with my mentors, while we were discussing and searching through VTK's FramebufferObject and TextureObject documentation, and the code itself for the problem,
one TextureObject method caught my attention: `vtkTextureObject.SetContext() <https://vtk.org/doc/nightly/html/classvtkTextureObject.html#a0988fa2a30b640c93392c2188030537e>`_.

Where the Problem Was
---------------------
My last week's code was:

::

   color_texture = vtk.vtkTextureObject() # color texture declaration
   color_texture.Bind() # binding of the texture for operations

   color_texture.SetDataType(vtk.VTK_UNSIGNED_CHAR) # setting the datatype for unsigned char
   color_texture.SetInternalFormat(vtk.VTK_RGBA) # setting the format as RGBA
   color_texture.SetFormat(vtk.VTK_RGBA)
   color_texture.SetMinificationFilter(0) # setting the minfilter as linear
   color_texture.SetMagnificationFilter(0) # setting the magfilter as linear

   color_texture.Allocate2D(width, height, 4, vtk.VTK_UNSIGNED_CHAR) # here is where the code stops

But it turns out that to allocate the FBO's textures, of type vtkTextureObject, you need to also set the context where the texture object
will be present, so it lacked a line, that should be added after ``Bind()``:

::

   color_texture = vtk.vtkTextureObject()
   color_texture.Bind()

   color_texture.SetContext(manager.window) # set the context where the texture object will be present

   color_texture.SetDataType(vtk.VTK_UNSIGNED_CHAR)
   color_texture.SetInternalFormat(vtk.VTK_RGB)
   color_texture.SetFormat(vtk.VTK_RGB)
   color_texture.SetMinificationFilter(0)
   color_texture.SetMagnificationFilter(0)

The code worked fine. But as my last blogpost showed, ``Allocate3D()`` method worked just fine without a (visible) problem, why is that?
Well, in fact, it **didn't work**. If we check the code for the ``Allocate2D()`` and ``Allocate3D()``, one difference can be spotted:



.. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/allocate-2d-3d.png
   :align: center
   :alt: Image comparing Allocate2D and Allocate3D methods



While in ``Allocate2D()`` there is an ``assert(this->Context);``, in ``Allocate3D()`` the assertion is translated into:

::

   if(this->Context==nullptr)
   {
     vtkErrorMacro("No context specified. Cannot create texture.");
     return false;
   }

This slight difference is significant: while in ``Allocate2D()`` the program immediately fails, in ``Allocate3D()`` the function is simply returned
**false**, with its error pushed to vtkErrorMacro. I could have realised that earlier if I were using vtkErrorMacro, but this difference in their
implementation made it harder for me and my mentors to realise what was happening.


This Week's Goals
-----------------
After making that work, this week's goal is to render something to the Framebuffer Object, now that is working. To do that,
first I will need to do some offscreen rendering to it, and afterwards render what it was drawn to its color attachment, the Texture Object I
was struggling to make work, into the screen, drawing its texture to a billboard. Also, I plan to start using vtkErrorMacro, as it seems like
the main error interface when working with VTK, and that may make my life easier.

See you next week!
