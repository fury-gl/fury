Week 12: Adding new tutorials
=============================

.. post:: September  7 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Restructured tutorials to be more readable and more focused on a specific topic.

- Replaced the old animation introductory tutorial with a lot simpler one, and added tutorial to explain keyframes and interpolators.

- Simplified setting lighting uniforms for the geometry-based-billboard actor by getting the ``Scene`` from the actor using ``actor.GetConsumer(scene_idx)``.


What is coming up next week?
----------------------------

- Allow the ``Timeline`` to take the ``ShowManager`` as an argument to reduce the amount of code the user has to write every time using FURY animations.

- Fix some typos in the tutorials and write some info about ``Slerp``.

- Find a way to fix the shader-callback problem of not being executed when the actor is out of the camera's frustum.


Did you get stuck anywhere?
---------------------------

I didn't get stuck this week.