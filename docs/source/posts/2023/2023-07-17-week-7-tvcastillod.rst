Week 7: Adjustments on the Uncertainty Cones visualization
==========================================================

.. post:: July 17, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

I was told to refactor some parts of the uncertainty PR, since I was relying too much on **dipy** functions which is not good because it makes maintenance more difficult as **dipy** requires **FURY** for some functionalities. So I did some adjustments on the uncertainty function parameters and the corresponding tests, hopefully I managed to get with the most appropriate definition but I need to receive a first feedback to see how much I have to adjust the implementation. As I had to delete some relevant code lines inside the uncertainty calculation which consisted of preprocessing the data in order to define the necessary variables for the uncertainty formula, I was also suggested to make a tutorial of this new feature, so I can explain in detail how to obtain and adjust the necessary information, before passing it to the actor, and in general how and what is the purpose of this new function.

I also continued working on the ellipsoid tutorial, which I hope to finish this week so that I can ask for a first revision.

What is coming up next?
-----------------------

I will finish defining some details of the tutorial so that it is ready for review, and now I will start working on the tutorial related to the uncertainty, while I receive feedback on the other PRs. Also, as preparation for the next step I will start exploring on how to address visualization of spherical harmonics for ODF glyphs visualization, I found that a previous GSoC participant at FURY started working on that and also did several work with raymarching and SDF (:doc:`here is a summary of the work <../2020/2020-08-24-final-work-lenix>`), so I will take a deeper look on that to see if I can get something useful I can start with.

Did I get stuck anywhere?
-------------------------

Not this week, but I foresee some problems with the uncertainty PR, we will see how it goes.
