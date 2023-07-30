Week 5: Preparing the data for the Ellipsoid tutorial
=====================================================

.. post:: July 03, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

During the weekly meeting with my mentors, there was a small discussion over the naming of the actor and its usage. On the one hand, although the purpose of the actor is to visualize diffusion tensor ellipsoids, the idea is that it can also be used for any other type of visualization that requires the use of ellipsoids, so in the end, we decided to keep the name *ellipsoid* as it is more generic. On the other hand, as there is already an actor made for the purpose of tensor visualization, namely `tensor_slicer <https://github.com/fury-gl/fury/blob/e595bad0246899d58d24121dcc291eb050721f9f/fury/actor.py#L1172>`_, it might not be obvious how and why one would use this new ellipsoid actor for this purpose, thus it was proposed to make a tutorial that can clarify this. The main difference between both actors relies on the quality and the amount of data that can be displayed, so the idea is to show the difference between both alternatives so the user can choose which one to use depending on their needs. To prepare the tutorial the first step was to `add the data <https://github.com/fury-gl/fury-data/pull/12>`_ I will use on `fury-data <https://github.com/fury-gl/fury-data>`_ so I can then fetch and load the datasets I need to work on the tutorial.

What is coming up next?
-----------------------

I need `#PR 791 <https://github.com/fury-gl/fury/pull/791>`_ to be reviewed by my GSoC fellows at FURY, so I will address their comments, and additionally make adjustments on `#PR 810 <https://github.com/fury-gl/fury/pull/810>`_ based on the feedback I receive. I will also start working on the tutorial, the idea is to show the use that can be made of the ellipsoid actor in the visualization of diffusion tensor ellipsoids, compared to the *tensor_slicer* actor. I plan to create a WIP PR to start getting feedback on the general structure of the tutorial and the way everything will be explained.

Did I get stuck anywhere?
-------------------------

I did not encounter any obstacles this week.
