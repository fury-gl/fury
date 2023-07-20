Week 4: First draft of the DTI uncertainty visualization
========================================================

.. post:: June 27, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

`#PR 810: DTI uncertainty visualization <https://github.com/fury-gl/fury/pull/810>`_

I made a second PR with the implementation of DTI uncertainty calculation and visualization. Below is an image of diffusion tensor ellipsoids and their associated uncertainty cones.

.. image:: https://user-images.githubusercontent.com/31288525/254747296-09a8674e-bfc0-4b3f-820f-8a1b1ad8c5c9.png
    :width: 530
    :align: center

I had to use some dipy functions, specifically: `estimate_sigma <https://dipy.org/documentation/1.4.1./reference/dipy.denoise/#estimate-sigma>`_ for the noise variance calculation, `design_matrix <https://dipy.org/documentation/1.4.0./reference/dipy.reconst/#design-matrix>`_ to get the b-matrix, and `tensor_prediction <https://dipy.org/documentation/1.4.0./reference/dipy.reconst/#tensor-prediction>`_ for the signal estimation. The details of this calculations can be found `here <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21111>`_.

What is coming up next?
-----------------------

I will continue working on the uncertainty PR which is still in its early stage, I'm going to make a couple of adjustments to the description of the parameters and the actor, and keep working on based on the feedback I receive. There are also minor details to be discussed with my mentors about the first PR, which I hope to finish refining.

Did I get stuck anywhere?
-------------------------

It took me a while to make the PR because I had some problems with the uncertainty function definition. I tried to use the least amount of parameters for the function, since with data, bvals and bvecs it is possible to obtain the rest of the parameters needed to generate the cones, which led me to readjust some calculations from the base implementation I had, to keep everything working correctly.
