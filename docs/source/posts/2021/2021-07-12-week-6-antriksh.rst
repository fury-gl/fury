Week #6: Bug fixes, Working on Tree2D UI
========================================

.. post:: July 12 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

What did I do this week?
------------------------
This week I had relatively few tasks and most of them were to fix some bugs/design flaws that were discussed in last week's meeting. Other than that, I had to implement a few things in the Tree2D UI element that will be discussed in detail below. I also had to update some existing PRs in order to make things work well. Below are the list of things I worked on this week:

* `Extracted Button2D class from elements to core <https://github.com/fury-gl/fury/pull/459>`_ : Button2D was placed in elements during the UI restructuring. The problem with that was that Button2D was needed in other UI elements outside UI elements present in elements in Panel2D. So, it was decided to create a rule that only the UI elements that do not depend on any other UI element are allowed to be placed in core UI elements. Button2D does not depend on any other UI element so it was extracted from elements to core.

* `Adapting GridLayout to work with UI elements <https://github.com/fury-gl/fury/pull/443>`_ : This was a PR that aimed to add support for UI elements to be placed in a grid fashion. the problem was that there still some circular imports even after UI restructuring, primarily because UI was being imported in the layout module that in turn indirectly imported some UI elements making them circularly dependent. To remove the circular imports, it was decided to determine the UI element by checking for a add_to_scene method attribute in the instance. I updated the existing PR to implement the same.

* `Continuing my work on Tree2D <https://github.com/fury-gl/fury/pull/460>`_: The Tree2D lacked some important things related to design and visual aspect of it. Before, if the children of any node exceeded its height they would just overflow. To prevent this I came up with a few solutions two of which were to either add a scrollbar on the overflowing node or to simply auto resize the parent node. Currently, there is no global API for the scrollbar and it has to be manually setup in a UI element, this will be hopefully implemented in the near future probably using layout management. Till then the auto resizing has been implemented for the nodes. In future, an option for scrollbar will be added.

Did I get stuck anywhere?
-------------------------
I am still stuck with adding tests for panel resizing PR. As it needs windows events to be recorded as well. I tried to propagate the event to the interactor first but that just lead to that particular event being registered multiple times. I will try to find a workaround for it.

What is coming up next?
-----------------------
If the Tree2D gets merged by this week then I'll probably work on other UI elements. Other tasks will be decided in the next meeting.

**See you guys next week!**