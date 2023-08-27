Week 10 : Start of SH implementation experiments
================================================

.. post:: August 08, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

I started formally working on SH implementation. I was told to start first doing a simple program where I can modify in real-time the order :math:`l` and degree :math:`m`, parameters corresponding to the `Spherical Harmonics function <https://dipy.org/documentation/1.7.0/theory/sh_basis/>`_ :math:`Y^m_l(\theta,\phi)=`, based on `previous work <https://github.com/lenixlobo/fury/commit/2b7ce7a71fd422dc5a250d7b49e1eea2db9d3bce>`_. That is just one part of the final ODF calculation, but here is what a first experimental script looks like.

.. image:: https://user-images.githubusercontent.com/31288525/260910073-10b0edd4-40e3-495c-85ad-79993aef3b19.png
    :width: 600
    :align: center

I did it in order to make sure it was visually correct and also to understand better how those 2 parameters are related and need to be incorporated into the final calculation. There is one issue at first sight that needs to be addressed, and that is the scaling, since for SH with a degree near 0, the object gets out of bounds.

What is coming up next?
-----------------------

I will keep polishing details from my current open PRs, hopefully, I will get another PR merged before the last GSoC week.

Did I get stuck anywhere?
-------------------------

Not sure about how to use the current implementation I have to get similar visualizations made with *odf_slicer*, since the parameters that the function receive are different, so I need to take a deeper look and see where it might be the connection or if I should make some adjustments on the parameters.
