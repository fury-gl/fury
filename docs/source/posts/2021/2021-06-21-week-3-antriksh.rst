Week #3: Adapting GridLayout to work with UI
============================================

.. post:: June 21 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

What did I do this week?
------------------------
This week my tasks revolved around layout and UI elements. The primary goal for this week was to adapt the GridLayout to work with different UI elements. Currently, GridLayout just supports vtk actors and not UI elements, my task was to modify the class to support UI elements. The other tasks for this week are described below in detail:

* `Adapt GridLayout to support UI elements <https://github.com/fury-gl/fury/pull/443>`_ : This was the main task for the week and the aim for this was to actually modify GridLayout to support UI elements. This was not possible before because GridLayout only supported vtk actors (because of certain methods only being provided by vtk actors). I modified the main class itself along with some utility functions. The problem that I faced during this was circular imports. Currently, the structure of FURY doesn't allow certain modules to be imported into other modules because of circular imports. A way to get around this was to actually import the modules inside the methods but this is not ideal always. This will be fixed in the future PRs where the UI module will be redesigned. I also added support for grid position offsetting, which basically means that the position of the UI elements that are placed in the Grid can be offset by a global offset passed in the constructor of GridLayout class. Below is an example showing the current state of GridLayout with different UI elements. I also created a brief example to demonstrate how to use GridLayout of different cellshapes with UI elements link to which is `here <https://github.com/fury-gl/fury/pull/443/files#diff-853d17c3134e7d22de88523bb787dc05d52ec798dc2111aa0419dfd5d634350a>`_.

    .. image:: https://i.imgur.com/EX2cN1i.png
        :width: 200
        :height: 200
* `Reviewed the FileDialog2D PR <https://github.com/fury-gl/fury/pull/294>`_ : This PR added support for FileDialog2D in the UI module. The PR needed to be reviewed in order to merge it as soon as other required PRs were merged. One of the mentors already reviewed the PR briefly my job was to review the PR for any remaining bugs.
* `Study #422 PR to understand contours around the drawn markers <https://github.com/fury-gl/fury/pull/422>`_ : In my previous week's tasks I created a PR to add support for borders in Panel2D. The borders were individually customizable just like in CSS which meant 4 Rectangle2D objects were needed to represent border in each direction. This is not ideal for a scenario where a lot of Panel2D are present in the scene as it can be performance taxing. A possible solution for this was to actually look how this was implemented in the #422. This PR allowed drawing millions of markers in one call that too from the GPU. Interestingly, each marker had a contour surrounding it which is exactly what we needed for Panel2D. This is something that can be considered in the future for border implementation in other complex UI elements.
* I also continued my work on the watcher class that I mentioned in the previous week's blog. The work for this is almost done and just needs some tests implemented, which should be done soon.

Did I get stuck anywhere?
-------------------------
Fortunately, I did not get stuck this week.

What is coming up next?
-----------------------
Next week I would probably continue to work on GridLayout and possibly other layouts as well, other tasks will be decided in the next meeting.

**See you guys next week!**

