Weekly Check-In #3
==================

.. post:: June 21 2021
   :author: Bruno Messias
   :tags: google
   :category: gsoc



What did you do this week?
--------------------------

-  `PR fury-gl/fury#422
   (merged): <https://github.com/fury-gl/fury/pull/422/commits/8a0012b66b95987bafdb71367a64897b25c89368>`__
   Integrated the 3d impostor spheres with the marker actor.
-  `PR fury-gl/fury#422
   (merged): <https://github.com/fury-gl/fury/pull/422>`__ Fixed some
   issues with my maker PR which now it's merged on fury.
-  `PR fury-gl/fury#432 <https://github.com/fury-gl/fury/pull/432>`__
   I've made some improvements in my PR which can be used to fine tune 
   the opengl state on VTK.
-  `PR fury-gl/fury#437 <https://github.com/fury-gl/fury/pull/437>`__
   I've made several improvements in my streamer proposal for FURY related to memory management.


-  `PR fury-gl/helios#1 <https://github.com/fury-gl/helios/pull/1>`__
   First version of async network layout using force-directed.

Did I get stuck anywhere?
-------------------------

A python-core issue
~~~~~~~~~~~~~~~~~~~

I've spent some hours trying to discover this issue. But now it's solved
through the commit
`devmessias/fury/commit/071dab85 <https://github.com/devmessias/fury/commit/071dab85a86ec4f97eba36721b247ca9233fd59e>`__

The `SharedMemory <https://docs.python.org/3/library/multiprocessing.shared_memory.html>`__
from python>=3.8 offers a new a way to share memory resources between
unrelated process. One of the advantages of using the SharedMemory
instead of the RawArray from multiprocessing is that the SharedMemory
allows to share memory blocks without those processes be related with a
fork or spawm method. The SharedMemory behavior allowed to achieve our
jupyter integration and `simplifies the use of the streaming
system <https://github.com/fury-gl/fury/pull/437/files#diff-7680a28c3a88a93b8dae7b777c5db5805e1157365805eeaf2e58fd12a00df046>`__.
However, I saw a issue in the shared memory implementation.

Let’s see the following scenario:

::

   1-Process A creates a shared memory X
   2-Process A creates a subprocess B using popen (shell=False)
   3-Process B reads X
   4-Process B closes X
   5-Process A kills B
   4-Process A closes  X
   5-Process A unlink() the shared memory resource X

The above scenario should work flawless. Calling unlink() in X is the right way as
discussed in the python official documentation. However, there is a open
issue  related the unlink method

-  `Issue:
   https://bugs.python.org/issue38119 <https://bugs.python.org/issue38119>`__
-  `PR
   python/cpython/pull/21516 <https://github.com/python/cpython/pull/21516>`__

Fortunately, I could use a
`monkey-patching <https://bugs.python.org/msg388287>`__ solution to fix
that meanwhile we wait to the python-core team to fix the
resource_tracker (38119) issue.

What is coming up next?
-----------------------

I'm planning to work in the
`fury-gl/fury#432 <https://github.com/fury-gl/fury/pull/432>`__ and
`fury-gl/helios#1 <https://github.com/fury-gl/helios/pull/1>`__.
