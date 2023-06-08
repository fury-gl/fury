Week 1: Ellipsoid actor implemented with SDF
============================================

.. post:: June 5, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

`PR #791: Ellipsoid actor implemented with SDF <https://github.com/fury-gl/fury/pull/791>`_

I made a first PR with the implementation of the ellipsoid actor defined with an SDF using raymarching. The current `sdf <https://github.com/fury-gl/fury/blob/master/fury/actor.py#L3537>`_ actor allows the creation of ellipsoids, but it lacks control over their shape, and the displayed direction does not match the intended orientation. For this reason, a new actor just focused on ellipsoids was made, this one is defined by its axes (3x3 orthogonal matrix) and their corresponding lengths (3x1 vector), along with other attributes like color, opacity, and scale. The goal is to make an implementation that allows displaying a large amount of data, with good visual quality, and without compromising performance. I'm still working on this but here is a first glance of how it looks like:

.. image:: https://user-images.githubusercontent.com/31288525/244503195-a626718f-4a13-4275-a2b7-6773823e553c.png
    :width: 376
    :align: center

This will be used later to create the tensor ellipsoids used on `tensor_slicer <https://github.com/fury-gl/fury/blob/master/fury/actor.py#L1172>`_.

What is coming up next?
-----------------------

I need to talk to my mentors first but the idea is to start making improvements on the SDF definition and raymarching algorithm, I have already started looking for information about how I can do it, and if I get good ideas, I will compare if there is an improvement in performance respect to the implementation I have right now. I also need to keep working on tests, the most common way of doing it is to check the number of objects and colors displayed, but I would like to test other things related to performance.

Did I get stuck anywhere?
-------------------------

Not yet, I need to get feedback first to see if there is anything I need to review or correct.
