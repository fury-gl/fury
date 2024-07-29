Week 11 : Adjusting ODF implementation and looking for solutions on issues found
================================================================================

.. post:: August 16, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

I continued to experiment with the ODF glyph implementation. Thanks to one of my mentors I figured out how to get the missing data corresponding to the SH coefficients :math:`a^l_m` part of the function :math:`f(\theta, \phi)` described `here <https://dipy.org/documentation/1.7.0/theory/sh_basis/>`_. I also was told to make sure to implement the correct SH basis since there are different definitions from the literature, I have to focus now in the one proposed by Descoteaux, described in `this paper <https://onlinelibrary.wiley.com/doi/10.1002/mrm.21277>`_, which is labeled in *DIPY* as *descoteaux07*. To do this I had to make a small adjustment to the base implementation that I took as a reference, from which I obtained a first result using SH of order 4.

.. image:: https://user-images.githubusercontent.com/31288525/260909561-fd90033c-018a-465b-bd16-3586bb31ca36.png
    :width: 600
    :align: center

It appears that the results on the shape are about the same, except for the direction, but there is still work to be done.

What is coming up next?
-----------------------

For now, there are 3 things I will continue to work on:

- The color and lighting. As these objects present curvatures with quite a bit of detail in some cases, this is something that requires more specific lighting work, in addition to having now not only one color but a color map.
- The scaling. This is something I still don't know how to deal with. I had to adjust it manually for now, but the idea is to find a relationship between the coefficients and the final object size so it can be automatically scaled, or maybe there is a proper way to pre-process this data before passing it to the shaders to get the right result at once.
- How to pass the information of the coefficients efficiently. Right now I'm creating one actor per glyph since I'm using a uniform array to pass the coefficients, but the idea is to pass all the data at once. I found several ideas `ideas here <https://stackoverflow.com/questions/7954927/passing-a-list-of-values-to-fragment-shader>`_ of how to pass a list of values to the fragment shader directly, I just need to explore deeper how this can be done on **FURY**, and see which option is most suitable.

Did I get stuck anywhere?
-------------------------

All the points mentioned above are things that I tried to fix, however, it is something that I need to look at in much more detail and that I know is going to take me some time to understand and test before I get to the expected result. I hope to get some ideas from my mentors and fellow GSoC contributors on how I can proceed to deal with each of the problems.
