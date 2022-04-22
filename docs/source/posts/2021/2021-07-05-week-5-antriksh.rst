Week #5: Rebasing all PRs w.r.t the UI restructuring, Tree2D, Bug Fixes
========================================================================

.. post:: July 05 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

What did I do this week?
------------------------
The UI restructuring was finally merged this week. This means UI is now a submodule in itself and provides different UI elements based on their types which are, core, elements, containers and some helper methods/classes. So, this week my main tasks were to rebase and fix merge conflicts of my open PR's. Other than that, I had to continue my work on Tree2D UI element and finish the remaining aspects of it. Also, I had to create an example demonstrating how to use the newly added UI element. Many use cases were discussed in the open meeting like, an image gallery which displays preview image on the title and when expanded reveals the full scale image. I am thinking of adding multiple examples for the Tree2D to brainstorm on its certain features. Also, I had this annoying bug in Panel2D which didn't allow it to be resized from the bottom right corner. It was resizing from the top right corner. I had to address this bug as well. Below are the tasks in detail:

* `Rebasing all PRs w.r.t the UI restructuring <https://github.com/fury-gl/fury/pulls/antrikshmisri>`_: As discussed in the earlier blogs, due to circular imports and large size of the UI module, a bit of restructuring was required. This week the PR that converts the UI into a sub module was finally merged. This meant I had to fix all the merge conflicts and rebase all UI related PR's. So, I rebased all the UI related PR's and fixed the merge conflicts. Currently, there are still some PR's that need some attention as still some of the imports are circular in nature. This means if the issue is correct then some more restructuring is required, which will be hopefully done in the near future.
* `Continuing the work on Tree2D <https://github.com/antrikshmisri/fury/blob/86b16ba3f74c3bdcf9aab58f546b37b919254cd1/fury/ui/elements.py#L3278>`_ : This week I continued my work on Tree2D, TreeNode2D. I had to fix/add multiple features on both the classes but my priority was to fix the looks of the UI element as well as make it easier for the user to manipulate the UI element. The first thing that I fixed was node offsetting, when a node is collapsed and expanded the nodes below the current node should also offset. Previously, only the child nodes within the same parents were offset and not the nodes/parent beyond that. With some minor adjusting, now the nodes are offset recursively and all child/parent nodes below the current nodes are offset. Secondly, before only a node could be added to a node which meant it wasn't any easy way to add any other UI element to a node but with some updates/fixes any UI element can be added to a node. Also, the Tree2D lacked some properties/methods to easily manipulate it. So, i added some properties/methods that allow to easily/efficiently manipulate individual node inside the Tree2D. Below is the current state of the Tree2D. In the below tree, two panels are added to a child node after the tree has been initialized. Also, the coordinated of the child elements are totally fluid i.e they can be placed anywhere inside the content panel by passing normalized or absolute coordinates.

    .. image:: https://i.imgur.com/rQQvLqs.png
        :width: 200
        :height: 200

* Fixed Panel2D bottom corner resizing: Previously, the panel would not resize from the bottom left corner but it would resize from top right corner. I didn't understand what was going wrong and was stuck on this for a long time. But I finally figured out the problem, I was calculating the Y-offset wrong as well as the panel resized from the top side instead of bottom. With some minor tweaking the bug was gone and the panel resizes correctly from the bottom right corner.

Did I get stuck anywhere?
-------------------------
I got stuck on recording events for the updated panel UI element. The panel updates w.r.t the window size but I couldn't figure out how to record the events invoked by the window. Unfortunately, I still haven't figured out how this will be done. My guess is that I have to propagate the event first to the interactor and then to the UI element.

What is coming up next?
-----------------------
I will probably finish up the GridLayout, Tree2D UI along side some other UI's. This will be decided in the next meeting.

**See you guys next week!**