Week #4: Adding Tree UI to the UI module
========================================

.. post:: June 28 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

What did I do this week?
------------------------
This week I had very few tasks to do, almost all of them revolved around UI elements and adding stuff to the UI module. Earlier, it was pointed out that due to some design issues, importing certain modules into others caused circular imports which led to importing the specific modules inside a class/method which is not the best approach. This will be resolved as soon as the PR that fixes this issue is reviewed/merged in the codebase. In the meantime, all the PR's related to UI will be on hold, which is why this I had few tasks this week. The tasks are described below in detail:

* `Addition of watcher class in UI <https://github.com/fury-gl/fury/pull/448>`_ :This is finally done, as described in the previous blogs this was something that was on hold for a long time. Primarily, due to other tasks I couldn't work on this but this week due to less tasks I was able to complete the watcher class and create a PR. This PR adds support for a watcher class in the UI elements. The purpose of this class is to monitor a particular attribute from the UI element after it has been added to the scene. If the attribute changes in the real time, a user defined callback is triggered and the scene is force rendered. Currently, if any attribute of the UI element changes after it is added to the scene it does not get updated accordingly. The only way to update the UI element would be to add a custom user hook that will be triggered when a particular event that can change the attribute is invoked. This is highly ambiguous as some unmonitored event can easily change many attributes of the UI element. Also it would be really hard to add user hooks for so many events. The watcher class does this automatically, it monitors the attribute for changes and if the attribute changes, a user defined callback is triggered. If this is something that is required in the UI module, then in the future a good addition would be to monitor the UI element instance as a whole instead of a single attribute .
* `Addition of Tree UI in the UI module <https://github.com/antrikshmisri/fury/blob/bb45d1c5b6fc0495dfe4d7814fab9aefbf9f7727/fury/ui.py#L5249>`_ : Another task for this week was to work on either Tree UI or the Accordion UI. I chose to work on Tree UI as it is very interesting to implement and the logic for Tree is almost similar to that of an Accordion. So far, I have managed to implement TreeNode2D. The Tree UI contains several nodes and each node can have its own sub-nodes/children. Also, each node has an expand/collapse button which can be used to chow/hide the underlying children. The Tree UI would take some sort of data structure that contains nodes/sub-nodes and convert each node to TreeNode2D and add all the processed node to the main Panel. So far this the result I have achieved: 

    .. image:: https://i.imgur.com/WIMWsrp.png
        :width: 200
        :height: 200

    .. image:: https://i.imgur.com/u33D7Qi.png
        :width: 200
        :height: 200
* `Resize Panel2D on window resizing <https://github.com/fury-gl/fury/pull/446>`_ : This PR adds support for resizing Panel2D on WindowResizeEvent. This means that the Panle2D resizes itself with respect to the changed window size. It also retains its maximum possible size and does not overflow. Also, this PR adds support for resizing the Panel2D for the bottom right corner. A placeholder button is placed at the bottom right corner of the Panel2D and when it is dragged by the placeholder the Panel2D resize accordingly. Below is an example:

    .. image:: https://i.imgur.com/87PN7TQ.gif
        :width: 200
        :height: 200
* Also, I did some testing of GridLayout when placed inside a resizable Panel2D. This will need to be worked on before advancing any further. Currently the elements present in the Panel2D do not resize properly w.r.t the changed panel size. Hopefully, this will be fixed in the future PRs.

Did I get stuck anywhere?
-------------------------
Fortunately, I did not get stuck this week.

What is coming up next?
-----------------------
The tasks for the next week will be decided in this weeks meeting.

**See you guys next week!**