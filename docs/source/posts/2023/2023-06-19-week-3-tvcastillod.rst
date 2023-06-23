Week 3: Working on uncertainty and details of the first PR
==========================================================

.. post:: June 19, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

I made some adjustments to the ellipsoid actor definition, now called *tensor*. This was something discussed in the weekly meeting as the coming changes are related to this actor, the idea now is to have a tensor actor that allows choosing between displaying the tensor ellipsoids or the uncertainty cones (later on). I also worked on the uncertainty calculation, and the cone SDF for the visualization, so I plan to do a WIP PR next to start getting feedback on this new addition.

What is coming up next?
-----------------------

As for the uncertainty calculation, other data is needed such as the noise variance and the design matrix (check `this <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21111>`_ article for more details), I need to identify which should be the parameters for the function definition. I also have to work on the documentation, so the function and its purpose are clear. I plan to make some final adjustments related to the uncertainty so that the next PR is ready for submission this week. I also expect to make final changes to the first PR so that it can be merged soon.

Did I get stuck anywhere?
-------------------------

Not this week, I will wait for feedback to see if there is anything I need to review or correct.
