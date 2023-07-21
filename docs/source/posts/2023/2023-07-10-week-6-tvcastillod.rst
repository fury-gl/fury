Week 6: First draft of the Ellipsoid tutorial
=============================================

.. post:: July 10, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

`#PR 818: Tutorial on using ellipsoid actor to visualize tensor ellipsoids for DTI <https://github.com/fury-gl/fury/pull/818>`_

I created the PR for the tutorial that will show the use that can be made of the *ellipsoid* actor in the visualization of diffusion tensor ellipsoids. It is still in its most basic stage, but the structure that I have thought of for now consists of: displaying a slice using *tensor_slicer* with spheres of 100, 200, and 724 vertices, and using *ellipsoid* actor, and show a comparison of the visual quality of the tensor ellipsoids. Then, display a ROI using both actors and a whole brain using the *ellipsoid* actor, to show that this new actor gives the possibility to display more data.

I also submitted the `uncertainty PR <https://github.com/fury-gl/fury/pull/810>`_ for review, in order to start making the necessary corrections.

What is coming up next?
-----------------------

I need `#PR 791 <https://github.com/fury-gl/fury/pull/791>`_ to be merged first, but meanwhile, I will start working on the explanation of the tutorial, since I already have the code structure and the idea of what I want to illustrate. I will discuss further work with my mentors at the upcoming meeting, so I can organize myself better and plan how I'm going to address the pending parts of my project.

Did I get stuck anywhere?
-------------------------

I found no major difficulties this week.
