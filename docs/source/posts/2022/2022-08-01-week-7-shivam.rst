Week 7 - Fixing bugs in animations
==================================

.. post:: August 01 2022
   :author: Shivam Anand
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- This week I started with implementing scaling to the animation example.

- I had a meeting with Mohamed in which we discussed the rotation issue, and we found out that glTF uses ``Slerp`` interpolator instead of the ``LinearInterpolator``. Using ``Slerp`` interpolator for rotation fixed the rotation issue.

- Another issue we faced was with the multi-actor system. Some actors weren't rotating or translating as intended. This was caused because the transformations were applied to the ``polydata`` before creating an actor; Mohamed suggested applying transformation after the actor is created from polydata (keeping the center to origin).


  .. raw:: html

        <iframe id="player" type="text/html"   width="640" height="360" src="https://user-images.githubusercontent.com/74976752/182166133-585d06ef-55ff-48db-8ce7-98b377ebf8ec.mp4" frameborder="0"></iframe>

Expected animation: `Interpolation Test Sample <https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/InterpolationTest>`_ 

- Created functions to return a list of animation timelines and apply them to the main timeline for keyframe animations.

- ``CubicSpline`` has not been implemented to glTF animation yet since it works differently than other Interpolators (takes tangent input to smoothen the curve) but it'll be done before our next meeting. 


What is coming up next week?
----------------------------

- Adding skinning animations support


Did you get stuck anywhere?
---------------------------

I still need to figure out how to apply the transformation matrix to the actor and not to the polydata.
