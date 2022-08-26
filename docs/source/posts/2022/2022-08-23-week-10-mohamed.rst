Week 10: Supporting hierarchical animating
==========================================

.. post:: August 23 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Implemented hierarchical order support for animations using matrices in this `PR`_.

- Improved the API of a `PartialActor`_ by adding some methods to control the scales, positions and colors.

- Added a new example of using the new hierarchical feature to animate an arm robot.

    .. raw:: html

        <iframe id="player" type="text/html"   width="440" height="390" src="https://user-images.githubusercontent.com/63170874/185803285-9184c561-a787-4ad0-ac1a-0b22854da889.mp4" frameborder="0"></iframe>

- Improved `vector_text`_ a little by adding options to control its direction.


What is coming up next week?
----------------------------

- Finish the hierarchical order animation support `PR`_.

- Explain tutorials in more detail. See this `issue`_.

- Fix issues discovered by Serge in this `review`_.

- Fix ``lightColor0`` being `hard-set`_ to ``(1, 1, 1)``. Instead, find a way to get this property the same way other polygon-based actor gets it.

- Try to get PRs and issues mentioned above merged, closed or ready for a final review.


Did you get stuck anywhere?
---------------------------

I didn't get stuck this week.


.. _`PR`: https://github.com/fury-gl/fury/pull/665
.. _`PartialActor`: https://github.com/fury-gl/fury/pull/660
.. _`vector_text`: https://github.com/fury-gl/fury/pull/661
.. _`review`: https://github.com/fury-gl/fury/pull/647#pullrequestreview-1061261078
.. _`issue`: https://github.com/fury-gl/fury/issues/664
.. _`hard-set`: https://github.com/fury-gl/fury/blob/464b3dd3f5be5159f5f9617a2c7b6f7bd65c0c80/fury/actor.py#L2395