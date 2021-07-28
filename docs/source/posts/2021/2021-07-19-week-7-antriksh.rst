Week #7: Finalizing the stalling PRs, finishing up Tree2D UI.
==============================================================

.. post:: July 19 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

What did I do this week?
------------------------
This week I had limited tasks to do, mostly tasks related to existing PRs. Other than some minor fixes I had to implement some more things in Tree2D which included some minor UI fixes, some changes in tutorial, adding tests. Below is the detailed description of what I worked on this week:

* `Tests, tutorial changes, UI fixes for Tree2D <https://github.com/fury-gl/fury/pull/460>`_ : The Tree2D lacked some things like proper UI resizing, relative indentation, tests for the UI class. These were added with this PR. Currently, the indentation, resizing needs some improvement, which will be fixed after feedback from this week's meeting. Also, tests for Tree2D, TreeNode2D were added as well.
* `Updating Panel2D tests, re-recording the events <https://github.com/fury-gl/fury/pull/446>`_ : This PR is almost done with just some tests blocking the PR. The tests were added this week, but tests for some callbacks that are associated with window event are still not added. This is because there is no way to count the WindowResizeEvent without actually using the API of the window provided by the OS. This can become very complicated very soon so, these tests may be added in the future.
* `Fixing the failing CI's for #443 <https://github.com/fury-gl/fury/pull/443>`_ : The CI was failing on this PR and needed some fixing which was done this week. This PR still needs some refactoring before the all CI's pass. This will hopefully be fixed before this week's meeting.
* `Addressing all comments regarding #442 <https://github.com/fury-gl/fury/pull/442>`_ : Previously, it was pointed out that the some code can be extracted into a function and can be reused in other methods. So, this week the extracted method was updated to reuse even more code and now almost no code is repeated.
* `Adding has_border flag in Panel2D <https://github.com/fury-gl/fury/pull/441>`_ : Adding a has_border flag in Panel2D: Previously, to create the borders 4 Rectangle2D's were used and they were created everytime even when border_width was set to 0. This would take a lot of wasted system resources. To fix this, a flag is added in the the constructor which is by default set to False. If false, the borders are not initialized and the resources are saved.

Did I get stuck anywhere?
-------------------------
Fortunately, this week I didn't get stuck anywhere.

**See you guys next week!**