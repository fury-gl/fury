======================================================
Week 12 - Fixing translating issues and updating tests
======================================================

.. post:: August 24 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
I started with updating the tests for PR `#623 <https://github.com/fury-gl/fury/pull/623>`_ as some of the tests weren't covering all the aspects in the code.
Previously I was just creating the ``DrawShape`` and adding it to the scene but now I had to analyze the scene to see whether they were properly added or not.

Then I moved on to PR `#653 <https://github.com/fury-gl/fury/pull/653>`_ to resolve the clamping issue. As you can see below, the shapes overlappes when they move nearer to the panel border.

.. image:: https://user-images.githubusercontent.com/64432063/187023615-d7c69904-4925-41b1-945d-b5993c20c862.gif
    :width: 400
    :align: center

To solve this, I am thinking of calculating the center of the group and clipping it according to the panel, which may give us the required result. I tried doing the same, and it worked partially.

.. image:: https://user-images.githubusercontent.com/64432063/187023880-84e13dab-313b-49e4-9b06-df5465c9d762.gif
    :width: 400
    :align: center

As we can see above, the shapes are kind of clamping but there's some issue with positioning. It would be good to go once this is fixed.

Along this, I tried to integrate shaders with the ``Rectangle2D`` but there's something which I am missing. When I tried executing that program, it executed successfully but I am not getting the desired output. I tried debugging the code as well using the `debug` flag on the `shader_to_actor` function but still, it doesn't throw any error.

Did you get stuck anywhere?
---------------------------
While updating the tests for PR `#623 <https://github.com/fury-gl/fury/pull/623>`_ the quad shape wasn't getting analyzed by the `analyse_snapshot` method. I tried various approaches to fix it, but it didn't work.

What is coming up next?
-----------------------
Merging the PRs `#623 <https://github.com/fury-gl/fury/pull/623>`_, `#653 <https://github.com/fury-gl/fury/pull/653>`_ and working on the freehand drawing.