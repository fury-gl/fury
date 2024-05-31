Week 4: Nothing is Ever Lost
============================

.. post:: June 26, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc


Welcome again to another weekly blogpost! Today, let's talk about the importance of guidance throughout a project.

Last Week's Effort
-----------------------
So, last week my project was struggling with some supposedly simple in concept, yet intricate in execution issues. If you recall from
my :doc:`last blogpost <2023-06-19-week-3-joaodellagli>`, I could not manage to make the Framebuffer Object setup work, as its method,
``SetContext()``, wasn't being able to generate the FBO inside OpenGL. Well, after some (more) research about that as I also dived in my
plan B, that involved studying numba as a way to accelerate a data structure I implemented on my PR `#783 <https://github.com/fury-gl/fury/pull/783>`_,
me and one of my mentors decided we needed a pair programming session, that finally happened on thursday. After that session,
we could finally understand what was going on.

Where the Problem Was
---------------------
Apparently, for the FBO generation to work, it is first needed to initialize the context interactor:

::

   FBO = vtk.vtkOpenGLFramebufferObject()

   manager.window.SetOffScreenRendering(True) # so the window doesn't show up, but important for later as well
   manager.initialize() # missing part that made everything work

   FBO.SetContext(manager.window) # Sets the context for the FBO. Finally, it works
   FBO.PopulateFramebuffer(width, height, True, 1, vtk.VTK_UNSIGNED_CHAR, False, 24, 0) # And now I could populate the FBO with textures


This simple missing line of code was responsible for ending weeks of suffer, as after that, I called:
::
   print("FBO of index:", FBO.GetFBOIndex())
   print("Number of color attachments:", FBO.GetNumberOfColorAttachments())

That outputted:
::
   FBO of index: 4
   Number of color attachments: 1

That means the FBO generation was successful! One explanation that seems reasonable to me on why was that happening is that, as it was
not initialized, the context was being passed ``null`` to the ``SetContext()`` method, that returned without any warning of what was happening.

Here, I would like to point out how my mentor was **essential** to this solution to come: I had struggled for some time with that, and could
not find a way out, but a single session of synchronous pair programming where I could expose clearly my problem and talk to someone
way more experienced than I, someone designated for that, was my way out of this torment, so value your mentors! Thanks Bruno!


This Week's Goals
-----------------
Now, with the FBO working, I plan to finally *render* something to it. For this week, I plan to come back to my original plan and
experiment with simple shaders just as a proof of concept that the FBO will be really useful for this project. I hope the road is less
bumpier by now and I don't step on any other complicated problem.

Wish me luck!
