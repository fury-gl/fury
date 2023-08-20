Week 8: Working on Ellipsoid Tutorial and exploring SH
======================================================

.. post:: July 25, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

I mainly worked on the ellipsoid actor tutorial, as `PR #791 <https://github.com/fury-gl/fury/pull/791>`_ is finally merged, so I was able to complete the tutorial by adding my implementation. In addition, during the weekly meeting, I received a good overview of the next issue I will be working on, which is using raymarching SDFs to display spherical harmonics (SH) for visualizing ODF glyphs for DTI. I got several ideas and resources which I can start experimenting with, such as `Shadertoy <https://www.shadertoy.com/results?query=Spherical+Harmonics>`_ and some base implementations from other FURY contributors. The main drawback when creating these objects is the amount of data required to create them, because depending on the SH order, the number of parameters that the function receives may vary, also unlike the tensors, which are represented only with a 3x3 matrix, here we could have more than 9 values associated with a single glyph, so passing the information from python to the shaders is not so trivial, besides requiring more resources as there is more information that needs to be processed. Some ideas I received were using matrixes instead of vectors, using templating, or even using texture to pass the data. I started to explore these options further, as well as to review in more detail the existing implementations of SH with raymarching, in order to understand them better.

What is coming up next?
-----------------------

I currently have two PRs under review, so I will address the comments I receive and update them accordingly. I also will continue to explore and start working on the implementation of these objects so that I can start making adjustments and further discuss possible improvements to the implementation I will make.

Did I get stuck anywhere?
-------------------------

Fortunately, I did not encounter any drawbacks this week.
