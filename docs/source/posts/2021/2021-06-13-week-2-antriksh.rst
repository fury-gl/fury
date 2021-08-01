Week #2: Feature additions in UI and IO modules
===============================================

.. post:: June 13 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

What did I do this week?
------------------------
This week I had to work on 3 PRs as well as some documentation. I really enjoyed this week's work as the tasks were really interesting. The aim for these PRs were to actually add a couple of features in the UI as well as the IO module, which includes, adding support for border in Panel2D, adding support for network/URL images in load_image method in IO module, adding resizing Panel2D from bottom right corner, completing the document with layout solutions provided by Unity/Unreal engine. Below are the PRs that I worked on:

* `Added support for URL image in load_image <https://github.com/fury-gl/fury/pull/440>`_ : The load_image of IO module didn't support network /URL images, so I made this PR to add support for the same.
* `Added support for border in Panel2D <https://github.com/fury-gl/fury/pull/441>`_ : This PR was made in association with the Card2D PR. This PR adds support for border in Panel2D. The borders are individually customizable just like in CSS. This PR needs a little tweaking in terms of getters/setters. The same support needs to be added in Rectangle2D.
* `Complete the document with layout solutions provided by Unity/Unreal engine <https://docs.google.com/document/d/1zo981_cyXZUgMDA9QdkVQKAHTuMmKaixDRudkQi4zlc/edit?usp=sharing>`_ : Completed the document with layout solutions provided by Unity/Unreal Engine.
* Behind the scenes I also worked on a Watcher class for the UI elements. The purpose of the watcher would be to monitor the UI elements for any changes after they have been added to the scene. A PR should be up by 2-3 days.

Did I get stuck anywhere?
-------------------------
I had a minor issue with the tests for the **IO** module. When running the tests for IO module using **pytest 5.0.0** resulted in Window fatal error, this was a sideeffect of pytest 5.0.0 wherein support for **faulthandler** was added. This error was suppressed by using certain flags while running the tests.

What is coming up next?
-----------------------
Next week I would probably work on adapting the **GridLayout** with UI elements, some other tasks that will be decided in the next meeting.

**See you guys next week!**