Week 9: Tutorial done and polishing DTI uncertainty
===================================================

.. post:: July 31, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

I addressed the comments from the tutorial of `PR #818 <https://github.com/fury-gl/fury/pull/818>`_  related to how to display specific visualizations I wanted to make. I was suggested to use *ShowManager* to handle the zoom of the scene and also to use *GridUI* to display several actors at the same time for a visual quality comparison of the tensors. Below are some images generated for the tutorial that is almost done.

.. image:: https://user-images.githubusercontent.com/31288525/260906510-d422e7b4-3ba3-4de6-bfd0-09c04bec8876.png
    :width: 600
    :align: center

What is coming up next?
-----------------------

There are some issues with the tests of the uncertainty implementation, specifically a segmentation problem that has to be with the shaders, so I expect to correct the problem by next week.

Did I get stuck anywhere?
-------------------------

I'm still thinking about how to approach the implementation of the spherical harmonics for ODF glyphs. Most of the implementations I have found are static so my task would be to try to parametrize the existing functions, so I can pass data from Python to the shaders properly so that I can obtain the same result as the current *odf_slicer*.
