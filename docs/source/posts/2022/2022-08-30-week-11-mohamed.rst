Week 11: Improving tutorials a little
=====================================

.. post:: August 30 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Fixed some issues in the hierarchical order animation support `PR`_ that we discussed during last week's meeting (mostly naming issues).

- Explained the introductory tutorial a little. But it is not suitable for beginners. So, I will spend time improving tutorials this week.

- Added extrusion to `vector_text`_ to allow the z-scaling to be functional.

- Fixed ``lightColor0`` being `hard-set`_ to ``(1, 1, 1)``. Instead, it's now using the ``Scene`` to set the lighting uniforms.


What is coming up next week?
----------------------------


- Improve tutorials.

- Find out how to get the ``Scene`` from the actor instead of manually assigning it.

- If I have time, I will try to implement recording animation as GIF or as a video.


Did you get stuck anywhere?
---------------------------

I didn't get stuck this week.


.. _`PR`: https://github.com/fury-gl/fury/pull/665
.. _`vector_text`: https://github.com/fury-gl/fury/pull/661
.. _`hard-set`: https://github.com/fury-gl/fury/blob/464b3dd3f5be5159f5f9617a2c7b6f7bd65c0c80/fury/actor.py#L2395