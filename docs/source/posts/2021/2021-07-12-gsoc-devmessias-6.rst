Network layout algorithms using IPC
===================================

.. post:: July 12 2021
   :author: Bruno Messias
   :tags: google
   :category: gsoc

Hi all. In the past weeks, I’ve been focusing on developing Helios; the
network visualization library for FURY. I improved the visual aspects of
the network rendering as well as implemented the most relevant network
layout methods.

In this post I will discuss the most challenging task that I faced to
implement those new network layout methods and how I solved it.

The problem: network layout algorithm implementations with a blocking behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Case 1:** Suppose that you need to monitor a hashtag and build a
social graph. You want to interact with the graph and at the same time
get insights about the structure of the user interactions. To get those
insights you can perform a node embedding using any kind of network
layout algorithm, such as force-directed or minimum distortion
embeddings.

**Case 2:** Suppose that you are modelling a network dynamic such as an
epidemic spreading or a Kuramoto model. In some of those network
dynamics a node can change the state and the edges related to the node
must be deleted. For example, in an epidemic model a node can represent
a person who died due to a disease. Consequently, the layout of the
network must be recomputed to give better insights.

In described cases if we want a better (UX) and at the same time a more
practical and insightful application of Helios layouts algorithms
shouldn’t block any kind of computation in the main thread.

In Helios we already have a lib written in C (with a python wrapper)
which performs the force-directed layout algorithm using separated
threads avoiding the GIL problem and consequently avoiding the blocking.
But and the other open-source network layout libs available on the
internet? Unfortunately, most of those libs have not been implemented
like Helios force-directed methods and consequently, if we want to
update the network layout the python interpreter will block the
computation and user interaction in your network visualization. How to
solve this problem?

Why is using the python threading is not a good solution?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One solution to remove the blocking behavior of the network layout libs
like PyMDE is to use the threading module from python. However, remember
the GIL problem: only one thread can execute python code at once.
Therefore, this solution will be unfeasible for networks with more than
some hundreds of nodes or even less! Ok, then how to solve it well?

IPC using python
~~~~~~~~~~~~~~~~

As I said in my previous posts I’ve created a streaming system for data
visualization for FURY using webrtc. The streaming system is already
working and an important piece in this system was implemented using the
python SharedMemory from multiprocessing. We can get the same ideas from
the streaming system to remove the blocking behavior of the network
layout libs.

My solution to have PyMDE and CuGraph-ForceAtlas without blocking was to
break the network layout method into two different types of processes: A
and B. The list below describes the most important behaviors and
responsibilities for each process

**Process A:**

-  Where the visualization (NetworkDraw) will happen
-  Create the shared memory resources: edges, weights, positions, info..
-  Check if the process B has updated the shared memory resource which
   stores the positions using the timestamp stored in the info_buffer
-  Update the positions inside of NetworkDraw instance

**Process B:**

-  Read the network information stored in the shared memory resources:
   edges , weights, positions
-  Execute the network layout algorithm
-  Update the positions values inside of the shared memory resource
-  Update the timestamp inside of the shared memory resource

I used the timestamp information to avoid unnecessary updates in the
FURY/VTK window instance, which can consume a lot of computational
resources.

How have I implemented the code for A and B?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because we need to deal with a lot of different data and share them
between different processes I’ve created a set of tools to deal with
that, take a look for example in the `ShmManagerMultiArrays
Object <https://github.com/fury-gl/helios/blob/main/helios/layouts/ipc_tools.py#L111>`__
, which makes the memory management less painful.

I'm breaking the layout method into two different processes. Thus I’ve
created two abstract objects to deal with any kind of network layout
algorithm which must be performed using inter-process-communication
(IPC). Those objects are:
`NetworkLayoutIPCServerCalc <https://github.com/devmessias/helios/blob/a0a24525697ec932a398db6413899495fb5633dd/helios/layouts/base.py#L65>`__
; used by processes of type B and
`NetworkLayoutIPCRender <https://github.com/devmessias/helios/blob/a0a24525697ec932a398db6413899495fb5633dd/helios/layouts/base.py#L135>`__
; which should be used by processes of type A.

I’ll not bore you with the details of the implementation. But let’s take
a look into some important points. As I’ve said saving the timestamp
after each step of the network layout algorithm. Take a look into the
method \_check_and_sync from NetworkLayoutIPCRender
`here <https://github.com/fury-gl/helios/blob/a0a24525697ec932a398db6413899495fb5633dd/helios/layouts/base.py#L266>`__.
Notice that the update happens only if the stored timestamp has been
changed. Also, look at this line
`helios/layouts/mde.py#L180 <https://github.com/fury-gl/helios/blob/a0a24525697ec932a398db6413899495fb5633dd/helios/layouts/mde.py#L180>`__,
the IPC-PyMDE implementation This line writes a value 1 into the second
element of the info_buffer. This value is used to inform the process A
that everything worked well. I used that info for example in the tests
for the network layout method, see the link
`helios/tests/test_mde_layouts.py#L43 <https://github.com/fury-gl/helios/blob/a0a24525697ec932a398db6413899495fb5633dd/helios/tests/test_mde_layouts.py#L43>`__

Results
~~~~~~~

Until now Helios has three network layout methods implemented: Force
Directed , Minimum Distortion Embeddings and Force Atlas 2. Here
`docs/examples/viz_helios_mde.ipynb <https://github.com/fury-gl/helios/blob/a0a24525697ec932a398db6413899495fb5633dd/docs/examples/viz_helios_mde.ipynb>`__
you can get a jupyter notebook that I’ve a created showing how to use
MDE with IPC in Helios.

In the animation below we can see the result of the Helios-MDE
application into a network with a set of anchored nodes.

|image1|

Next steps
~~~~~~~~~~

I’ll probably focus on the Helios network visualization system.
Improving the documentation and testing the ForceAtlas2 in a computer
with cuda installed. See the list of opened
`issues <https://github.com/fury-gl/helios/issues>`__

Summary of most important pull-requests:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  IPC tools for network layout methods (helios issue #7)
   `fury-gl/helios/pull/6 <https://github.com/fury-gl/helios/pull/6>`__
-  New network layout methods for fury (helios issue #7)
   `fury-gl/helios/pull/9 <https://github.com/fury-gl/helios/pull/9>`__
   `fury-gl/helios/pull/14 <https://github.com/fury-gl/helios/pull/14>`__
   `fury-gl/helios/pull/13 <https://github.com/fury-gl/helios/pull/13>`__
-  Improved the visual aspects and configurations of the network
   rendering(helios issue #12)
   https://github.com/devmessias/helios/tree/fury_network_actors_improvements
-  Tests, examples and documentation for Helios (helios issues #3 and
   #4)
   `fury-gl/helios/pull/5 <https://github.com/fury-gl/helios/pull/5>`__
-  Reduced the flickering effect on the FURY/Helios streaming system
   `fury-gl/helios/pull/10 <https://github.com/fury-gl/helios/pull/10>`__
   `fury-gl/fury/pull/437/commits/a94e22dbc2854ec87b8c934f6cabdf48931dc279 <https://github.com/fury-gl/fury/pull/437/commits/a94e22dbc2854ec87b8c934f6cabdf48931dc279>`__

.. |image1| image:: https://user-images.githubusercontent.com/6979335/125310065-a3a9f480-e308-11eb-98d9-0ff5406a0e96.gif


