Week 2: Making adjustments to the Ellipsoid Actor
=================================================

.. post:: June 12, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

I made some minor adjustments to the last PR I submit. Last time it was a draft since I was waiting for the weekly meeting to know how to proceed, but now is ready. I am waiting for the review so I can make the necessary corrections and adjustments to merge this first PR soon.

What is coming up next?
-----------------------

As I receive feedback, I will continue to work on the `PR #791 <https://github.com/fury-gl/fury/pull/791>`_ and make adjustments and changes as needed. That said, I will start working on another part of the project, which is the visualization of uncertainty. Without going into details (for now) what I have to do is:

- Create a double_cone or dti_uncertainty actor. I'm going to work on the double cone made also with raymarching and SDF, since the implementation is pretty much the same as the ellipsoid I already have.
- Make a function that returns the level of the uncertainty given by the angle of the uncertainty cone we want to visualize. For this I need to double-check the maths behind the uncertainty calculation to make sure I'm getting the right results.

Did I get stuck anywhere?
-------------------------

Not exactly, but one of the things that were mentioned in the last meeting is that we should try to simplify the shader code as much as we can, that is, to break down the entire implementation into simple and easy-to-understand lines of code, which also allows the definition of functions that can be reused later on. I need to keep working on this, so I can make my code even more readable and fit the new shader structure.
