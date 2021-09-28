Weekly Check-In #5
===================

.. post:: July 05 2021
   :author: Bruno Messias
   :tags: google
   :category: gsoc

What did you do this week?
--------------------------

`fury-gl/fury PR#437: WebRTC streaming system for FURY`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Before the `8c670c2`_ commit, for some versions of MacOs the
   streaming system was falling in a silent bug. I’ve spent a lot of
   time researching to found a cause for this. Fortunately, I could found
   the cause and the solution. This troublesome MacOs was falling in a
   silent bug because the SharedMemory Object was creating a memory
   resource with at least 4086 bytes indepedent if I've requested less
   than that. If we look into the MultiDimensionalBuffer Object
   (stream/tools.py) before the 8c670c2 commit we can see that Object
   has max_size parameter which needs to be updated if the SharedMemory
   was created with a "wrong" size.

`fury-gl/helios PR 1: Network Layout and SuperActors`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the past week I've made a lot of improvements in this PR, from
performance improvements to visual effects. Bellow are the list of the
tasks related with this PR:

-  - Code refactoring.
-  - Visual improvements: Using the UniformTools from my pull request
   `#424`_ now is possible to control all the visual characteristics at
   runtime.
-  - 2D Layout: Meanwhile 3d network representations are very usefully
   for exploring a dataset is hard to convice a group of network
   scientists to use a visualization system which dosen't allow 2d
   representations. Because of that I started to coding the 2d behavior
   in the network visualization system.
-  - Minimum Distortion Embeddings examples: I've created some examples
   which shows how integrate pymde (Python Minimum Distortion
   Embeddings) with fury/helios. The image below shows the result of
   this integration: a "perfect" graph embedding

.. image:: https://user-images.githubusercontent.com/6979335/124524052-da937e00-ddcf-11eb-83ca-9b58ca692c2e.png

What is coming up next week?
----------------------------

I'll probably focus on the `heliosPR#1`_. Specifically, writing tests
and improving the minimum distortion embedding layout.

Did you get stuck anywhere?
---------------------------

I did not get stuck this week.

.. _`fury-gl/fury PR#437: WebRTC streaming system for FURY`: https://github.com/fury-gl/fury/pull/427
.. _8c670c2: https://github.com/fury-gl/fury/pull/437/commits/8c670c284368029cdb5b54c178a792ec615e4d4d
.. _`fury-gl/helios PR 1: Network Layout and SuperActors`: https://github.com/fury-gl/helios/pull/1
.. _#424: https://github.com/fury-gl/fury/pull/424
.. _heliosPR#1: 