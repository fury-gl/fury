Weekly Check-In #7
===================

.. post:: July 19 2021
   :author: Bruno Messias
   :tags: google
   :category: gsoc


What did I do this week?
------------------------

-  `PR fury-gl/helios#16
   (merged): <https://github.com/fury-gl/helios/pull/16>`__ Helios IPC
   network layout support for MacOs

-  `PR fury-gl/helios#17
   (merged): <https://github.com/fury-gl/helios/pull/17>`__ Smooth
   animations for IPC network layout algorithms

   Before this commit was not possible to record the positions to have a
   smooth animations with IPCLayout approach. See the animation below

   |image1|

   After this PR now it's possible to tell Helios to store the evolution
   of the network positions using the record_positions parameter. This
   parameter should be passed on the start method. Notice in the image
   below how this gives to us a better visualization

   |image2|

-  `PR fury-gl/helios#13
   (merged) <https://github.com/fury-gl/helios/pull/13>`__ Merged the
   forceatlas2 cugraph layout algorithm

Did I get stuck anywhere?
-------------------------

I did not get stuck this week.

What is coming up next?
-----------------------

Probably, I'll work more on Helios. Specifically I want to improve the
memory management system. It seems that some shared memory resources are
not been released when using the IPCLayout approach.

.. |image1| image:: https://user-images.githubusercontent.com/6979335/126175596-e6e2b415-bd79-4d99-82e7-53e10548be8c.gif
.. |image2| image:: https://user-images.githubusercontent.com/6979335/126175583-c7d85f0a-3d0c-400e-bbdd-4cbcd2a36fed.gif
