Week #8: Code Cleanup, Finishing up open PRs, Continuing work on Tree2D
========================================================================

.. post:: July 26 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

What did I do this week?
------------------------
This week I had to work on the open PRs specifically work on the bugs that were pointed out in last week's meeting. Along side the bugs I had to continue the work on Tree2D UI element. Below is the detailed description of what I worked on this week:

* `Added new tutorial, code clean-up, bug fixes <https://github.com/fury-gl/fury/pull/460>`_ : The Tree2D had some issues with its resizing of child nodes. The size for the nodes was calculated automatically based on the vertical size occupied by its children but this could be problematic when working with sliders or UI elements that take up a lot of vertical size. To avoid this the children sizes are calculated relative to each other and the vertical size is calculated such that all children fit in perfectly. Besides this, a multiselection flag has been added that decides whether multiple child nodes can be selected or not.
* `Adding tests for corner resize callback <https://github.com/fury-gl/fury/pull/446>`_ : This PR is almost done as it was decided that WindowsResizeEvent will be ignored for now. Which leaves us with corner resizing, the callback for corner resizing didn't have any tests so the recording was redone and tests for the corner resize callback was added.
* `Fixing the failing CI's for #443 <https://github.com/fury-gl/fury/pull/443>`_ : The solution that ended up working was creating custom objects for testing of is_ui method. With this update there will be no circular dependencies and no failing CI's.
* `Addressing all comments regarding #442 <https://github.com/fury-gl/fury/pull/442>`_ : In the last meeting, a bug was pointed out wherein the text wouldn't wrap as expected. The reason for this was some minor calculation mistakes. The whole wrap_overflow method was redone and now everything works as expected. Hopefully, no bugs pop up during this week's meeting.
* `Addressing some minor comments <https://github.com/fury-gl/fury/pull/441>`_ : This PR is almost done too, there were some minor changes that were required to be addressed before this could be merged. So, these comments were fixed and hopefully this will be merged soon.
* Using different fonts using FreeType python API: A major thing that FURY needs right now is using different fonts on the fly. This is more complicated than it seems, in case of browser environment this is not a problem as browsers can support and render all fonts using various techniques. In case of a desktop environment, we need to generate the bitmap for the fonts and then use them in form of textures. For now I have created a small example that generates these bitmaps from a python API called freetype-py, the fonts are fetched from google fonts and then they are displayed as textures.
* **Starting working on Vertical Layout**: As majority of PRs are almost done, I started working on Vertical Layout. This will be hihgly inspired from the Grid Layout with obvious differences. The same techniques are being used in the Tree2D so this shouldn't be difficult to implement.

Did I get stuck anywhere?
-------------------------
The failing CI's for Grid Layout Pr needed some custom objects to remove circular dependencies. I couldn't figure out where should these custom objects go but fortunaltely the mentors showed me a quick example of where it should go.

What is coming up next week?
----------------------------
Next week I will continue my work on some other UI's and the remaining Layouts.

**See you guys next week!**