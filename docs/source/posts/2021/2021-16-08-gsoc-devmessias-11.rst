Week #11: Removing the flickering effect
========================================

.. post:: August 16 2021
   :author: Bruno Messias
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

FURY
^^^^

-  `PR fury-gl/fury#489: <https://github.com/fury-gl/fury/pull/489>`__

  This PR give to FURY three
  pre-built texture maps using different fonts. However, is quite easy
  to create new fonts to be used in a visualization.
| It’s was quite hard to develop the shader code and find the correct
  positions of the texture maps to be used in the shader. Because we
  used the freetype-py to generate the texture and packing the glyps.
  However, the lib has some examples with bugs. But fortunelly, now
  everthing is woking on FURY. I’ve also created two different examples
  to show how this PR works.

   The first example, viz_huge_amount_of_labels.py, shows that the user can 
   draw hundreds of thounsands of characters.


   |image2|

   The second example, viz_billboad_labels.py, shows the different behaviors of the label actor. In addition, presents 
   to the user how to create a new texture atlas font to be used across different visualizations.

-  `PR fury-gl/fury#437: <https://github.com/fury-gl/fury/pull/437>`__

   -  Fix: avoid multiple OpenGl context on windows using asyncio
         The streaming system must be generic, but opengl and vtk behaves in uniques ways in each Operating System. Thus, can be tricky 
         to have the same behavior acrros different OS. One hard stuff that we founded is that was not possible to use my 
         TimeIntervals objects (implemented with threading module) with vtk. The reason for this impossibility is because we can't use 
         vtk in windows in different threads. But fortunely, moving from the threading (multithreading) to the asyncio approcach (concurrency) 
         have fixed this issue and now the streaming system is ready to be used anywhere.

   -  Flickering:
  
         Finally, I could found the cause of the flickering effect on the streaming system. 
         This flickering was appearing only when the streaming was created using the Widget object. 
         The cause seems to be a bug or a strange behavior from vtk. 
         Calling   iren.MouseWheelForwardEvent() or iren.MouseWheelBackwardEvent() 
         inside of a thread without invoking the
         Start method from a vtk instance produces a memory corruption.
         Fortunately, I could fix this behavior and now the streaming system is
         working without this glitch effect.


FURY/Helios
^^^^^^^^^^^

-  `PR fury-gl/helios#24
   : <https://github.com/fury-gl/helios/pull/24>`__

This uses the
`PRfury-gl/fury#489: <https://github.com/fury-gl/fury/pull/489>`__ to
give the network label feature to helios. Is possible to draw node
labels, update the colors, change the positions at runtime. In addition,
when a network layout algorithm is running this will automatically
update the node labels positions to follow the nodes across the screen.

|image1|

-  `PR fury-gl/helios#23:
   Merged. <https://github.com/fury-gl/helios/pull/23>`__

This PR granted compatibility between IPC Layouts and Windows. Besides
that , now is quite easier to create new network layouts using inter
process communication

Did I get stuck anywhere?
-------------------------

I did not get stuck this week.

.. |image1| image:: https://user-images.githubusercontent.com/6979335/129642582-fc6785d8-0e4f-4fdd-81f4-b2552e1ff7c7.png
.. |image2| image:: https://user-images.githubusercontent.com/6979335/129643743-6cb12c06-3415-4a02-ba43-ccc97003b02d.png
