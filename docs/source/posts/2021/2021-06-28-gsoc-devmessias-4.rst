SOLID, monkey patching  a python issue and  network visualization through WebRTC
================================================================================

.. post:: July 05 2021
   :author: Bruno Messias
   :tags: google
   :category: gsoc



These past two weeks I’ve spent most of my time in the `Streaming System
PR <https://github.com/fury-gl/fury/pull/437>`__ and the `Network Layout
PR <https://github.com/fury-gl/helios/pull/1/>`__ . In this post I’ll
focus on the most relevant things I’ve made for those PRs.

Streaming System
----------------

**Pull
request** : \ `fury-gl/fury/pull/437 <https://github.com/fury-gl/fury/pull/437/>`__.

Code Refactoring
~~~~~~~~~~~~~~~~

Abstract class and SOLID
^^^^^^^^^^^^^^^^^^^^^^^^

The past weeks I've spent some time refactoring the code to see what
I’ve done let’ s take a look into this
`fury/blob/b1e985.../fury/stream/client.py#L20 <https://github.com/devmessias/fury/blob/b1e985bd6a0088acb4a116684577c4733395c9b3/fury/stream/client.py#L20>`__,
the FuryStreamClient Object before the refactoring.

The code is a mess. To see why this code is not good according to SOLID
principles let’s just list all the responsibilities of FuryStreamClient:

-  Creates a RawArray or SharedMemory to store the n-buffers
-  Creates a RawArray or SharedMemory to store the information about
   each buffer
-  Cleanup the shared memory resources if the SharedMemory was used
-  Write the vtk buffer into the shared memory resource
-  Creates the vtk callbacks to update the vtk-buffer

That’s a lot and those responsibilities are not even related to each
other. How can we be more SOLID[1]? An obvious solution is to create a
specific object to deal with the shared memory resources. But it's not
good enough because we still have a poor generalization since this new
object still needs to deal with different memory management systems:
rawarray or shared memory (maybe sockets in the future). Fortunately, we
can use the python Abstract Classes[2] to organize the code.

To use the ABC from python I first listed all the behaviors that should
be mandatory in the new abstract class. If we are using SharedMemory or
RawArrays we need first to create the memory resource in a proper way.
Therefore, the GenericImageBufferManager must have a abstract method
create_mem_resource. Now take a look into the ImageBufferManager inside
of
`stream/server/server.py <https://github.com/devmessias/fury/blob/c196cf43c0135dada4e2c5d59d68bcc009542a6c/fury/stream/server/server.py#L40>`__,
sometimes it is necessary to load the memory resource in a proper way.
Because of that, the GenericImageBufferManager needs to have a
load_mem_resource abstract method. Finally, each type of
ImageBufferManager should have a different cleanup method. The code
below presents the sketch of the abstract class


.. code-block:: python

   from abc import ABC, abstractmethod

   GenericImageBufferManager(ABC):
       def __init__(
               self, max_window_size=None, num_buffers=2, use_shared_mem=False):
            ...
       @abstractmethod
       def load_mem_resource(self):
           pass
       @abstractmethod
       def create_mem_resource(self):
           pass
       @abstractmethod
       def cleanup(self):
           pass

Now we can look for those behaviors inside of FuryStreamClient.py and
ImageBufferManger.py that does not depend if we are using the
SharedMemory or RawArrays. These behaviors should be methods inside of
the new GenericImageBufferManager.



.. code-block:: python

   # code at: https://github.com/devmessias/fury/blob/440a39d427822096679ba384c7d1d9a362dab061/fury/stream/tools.py#L491

   class GenericImageBufferManager(ABC):
       def __init__(
               self, max_window_size=None, num_buffers=2, use_shared_mem=False)
           self.max_window_size = max_window_size
           self.num_buffers = num_buffers
           self.info_buffer_size = num_buffers*2 + 2
           self._use_shared_mem = use_shared_mem
            # omitted code
       @property
       def next_buffer_index(self):
           index = int((self.info_buffer_repr[1]+1) % self.num_buffers)
           return index
       @property
       def buffer_index(self):
           index = int(self.info_buffer_repr[1])
           return index
       def write_into(self, w, h, np_arr):
           buffer_size = buffer_size = int(h*w)
           next_buffer_index = self.next_buffer_index
            # omitted code

       def get_current_frame(self):
           if not self._use_shared_mem:
           # omitted code
           return self.width, self.height, self.image_buffer_repr

       def get_jpeg(self):
           width, height, image = self.get_current_frame()
           if self._use_shared_mem:
           # omitted code
           return image_encoded.tobytes()

       async def async_get_jpeg(self, ms=33):
          # omitted code
       @abstractmethod
       def load_mem_resource(self):
           pass

       @abstractmethod
       def create_mem_resource(self):
           pass

       @abstractmethod
       def cleanup(self):
           Pass

With the
`GenericImageBufferManager <https://github.com/devmessias/fury/blob/440a39d427822096679ba384c7d1d9a362dab061/fury/stream/tools.py#L491>`__
the
`RawArrayImageBufferManager <https://github.com/devmessias/fury/blob/440a39d427822096679ba384c7d1d9a362dab061/fury/stream/tools.py#L609>`__
and
`SharedMemImageBufferManager <https://github.com/devmessias/fury/blob/440a39d427822096679ba384c7d1d9a362dab061/fury/stream/tools.py#L681>`__
is now implemented with less duplication of code (DRY principle). This
makes the code more readable and easier to find bugs. In addition, later
we can implement other memory management systems in the streaming system
without modifying the behavior of FuryStreamClient or the code inside of
server.py.

I’ve also applied the same SOLID principles to improve the CircularQueue
object. Although the CircularQueue and FuryStreamInteraction were not
violating the S from SOLID, the head-tail buffer from the CircularQueue
must have a way to lock the write/read if the memory resource is busy.
Meanwhile the
`multiprocessing.Arrays <https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Array>`__
already has a context which allows lock (.get_lock()) SharedMemory
doesn’t[2]. The use of abstract class allowed me to deal with those
peculiarities. `commit
358402e <https://github.com/fury-gl/fury/pull/437/commits/358402ea2f06833f66f45f3818ccc3448b2da9cd>`__

Using namedtuples to grant immutability and to avoid silent bugs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The circular queue and the user interaction are implemented in the
streaming system using numbers to identify the type of event (mouse
click, mouse weel, ...) and where to store the specific values
associated with the event , for example if the ctrl key is pressed or
not. Therefore, those numbers appear in different files and locations:
tests/test_stream.py, stream/client.py, steam/server/app_async.py. This
can be problematic because a typo can create a silent bug. One
possibility to mitigate this is to use a python dictionary to store the
constant values, for example

.. code-block:: python

   EVENT_IDS = {
        "mouse_move" : 2, "mouse_weel": 1, #...
   }

But this solution has another issue, anywhere in the code we can change
the values of EVENT_IDS and this will produce a new silent bug. To avoid
this I chose to use
`namedtuples <https://docs.python.org/3/library/collections.html#collections.namedtuple>`__
to create an immutable object which holds all the constant values
associated with the user interactions.
`stream/constants.py <https://github.com/devmessias/fury/blob/b1e985bd6a0088acb4a116684577c4733395c9b3/fury/stream/constants.py#L59>`__

The namedtuple has several advantages when compared to dictionaries for
this specific situation. In addition, it has a better performance. A
good tutorial about namedtuples it’s available here
https://realpython.com/python-namedtuple/

Testing
~~~~~~~

My mentors asked me to write tests for this PR. Therefore, this past
week I’ve implemented the most important tests for the streaming system:
`/fury/tests/test_stream.py <https://github.com/devmessias/fury/blob/440a39d427822096679ba384c7d1d9a362dab061/fury/tests/test_stream.py>`__

Most relevant bugs
~~~~~~~~~~~~~~~~~~

As I discussed in my `third
week <https://blogs.python-gsoc.org/en/demvessiass-blog/weekly-check-in-3-15/>`__
check-in there is an open issue related to SharedMemory in python.
This"bug" happens in the streaming system through the following scenario

.. code-block:: bash

   1-Process A creates a shared memory X
   2-Process A creates a subprocess B using popen (shell=False)
   3-Process B reads X
   4-Process B closes X
   5-Process A kills B
   4-Process A closes  X
   5-Process A unlink() the shared memory resource

In python, this scenario translates to

.. code-block:: python

   from multiprocessing import shared_memory as sh
   import time
   import subprocess
   import sys

   shm_a = sh.SharedMemory(create=True, size=10000)
   command_string = f"from multiprocessing import shared_memory as sh;import time;shm_b = sh.SharedMemory('{shm_a.name}');shm_b.close();"
   time.sleep(2)
   p = subprocess.Popen(
       [sys.executable, '-c', command_string],
       stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
   p.wait()
   print("\nSTDOUT")
   print("=======\n")
   print(p.stdout.read())
   print("\nSTDERR")
   print("=======\n")
   print(p.stderr.read())
   print("========\n")
   time.sleep(2)
   shm_a.close()
   shm_a.unlink()

Fortunately, I could use a monkey-patching[3] solution to fix that;
meanwhile we're waiting for the python-core team to fix the
resource_tracker (38119) issue [4].

Network Layout (Helios-FURY)
----------------------------

**Pull
request**\ `fury-gl/helios/pull/1 <https://github.com/fury-gl/helios/pull/1/>`__

Finally, the first version of FURY network layout is working as you can
see in the video below.

In addition, this already can be used with the streaming system allowing
user interactions across the internet with WebRTC protocol.

One of the issues that I had to solve to achieve the result presented in
the video above was to find a way to update the positions of the vtk
objects without blocking the main thread and at the same time allowing
the vtk events calls. My solution was to define an interval timer using
the python threading module:
`/fury/stream/tools.py#L776 <https://github.com/devmessias/fury/blob/440a39d427822096679ba384c7d1d9a362dab061/fury/stream/tools.py#L776>`__,
`/fury/stream/client.py#L112 <https://github.com/devmessias/fury/blob/440a39d427822096679ba384c7d1d9a362dab061/fury/stream/client.py#L112>`__
`/fury/stream/client.py#L296 <https://github.com/devmessias/fury/blob/440a39d427822096679ba384c7d1d9a362dab061/fury/stream/client.py#L296>`__

Refs:
-----

-  [1] A. Souly,"5 Principles to write SOLID Code (examples in Python),"
   Medium, Apr. 26, 2021.
   https://towardsdatascience.com/5-principles-to-write-solid-code-examples-in-python-9062272e6bdc
   (accessed Jun. 28, 2021).
-  [2]"[Python-ideas] Re: How to prevent shared memory from being
   corrupted ?"
   https://www.mail-archive.com/python-ideas@python.org/msg22935.html
   (accessed Jun. 28, 2021).
-  [3]“Message 388287 - Python tracker."
   https://bugs.python.org/msg388287 (accessed Jun. 28, 2021).
-  [4]“bpo-38119: Fix shmem resource tracking by vinay0410 · Pull
   Request #21516 · python/cpython," GitHub.
   https://github.com/python/cpython/pull/21516 (accessed Jun. 28,
   2021).
