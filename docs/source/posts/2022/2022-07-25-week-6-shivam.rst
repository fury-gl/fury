Week 6 - Extracting the animation data
======================================

.. post:: July 25 2022
   :author: Shivam Anand
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- This week, it was all about reading docs and extracting the animation data from buffers.

- Currently, the glTF class can extract simple node transformation and morphing data, as they are stored in the ``Animations`` of the glTF file.

- Skinning (Skeletal Animation) data is stored in ``Nodes`` inside the skin parameter. We shall be able to load that before our next meeting on Wednesday.

- Created a `tutorial <https://github.com/xtanion/fury/blob/gltf-anim-merge-kf/docs/tutorials/01_introductory/viz_simple_gltf_animation.py>`_ using keyframe animations (`#626`_) and adding multiple ``timelines`` into a main ``timeline`` as suggested by Mohamed.

  .. raw:: html

        <iframe id="player" type="text/html"   width="550" height="400" src="https://user-images.githubusercontent.com/74976752/180841262-7356204e-2097-4ea4-ab71-39f04a1e7a07.mp4" frameborder="0"></iframe>



What is coming up next week?
----------------------------

As of now, we've decided the following:

- Create a custom Interpolator (for simple node transformations, morph and Skeletal animations).


Did you get stuck anywhere?
---------------------------

No, I didn't get stuck this week.


.. _`#626`: https://github.com/fury-gl/fury/pull/626
