A Stadia-like system for data visualization
===========================================

.. post:: June 13 2021
   :author: Bruno Messias
   :tags: google
   :category: gsoc

Hi all! In this post I'll talk about the PR
`#437 <https://github.com/fury-gl/fury/pull/437>`__.

There are several reasons to have a streaming system for data
visualization. Because I’m doing a PhD in a developing country I always
need to think of the cheapest way to use the computational resources
available. For example, with the GPUs prices increasing, it’s necessary
to share a machine with a GPU with different users in different
locations. Therefore, to convince my Brazilian friends to use FURY I
need to code thinking inside of the (a) low-budget scenario.

To construct the streaming system for my project I’m thinking about the
following properties and behaviors:

#. I want to avoid blocking the code execution in the main thread (where
   the vtk/fury instance resides).
#. The streaming should work inside of a low bandwidth environment.
#. I need an easy way to share the rendering result. For example, using
   the free version of ngrok.

To achieve the property **1.** we need to circumvent the GIL problem.
Using the threading module alone it’s not good enough because we can’t
use the python-threading for parallel CPU computation. In addition, to
achieve a better organization it’s better to define the server system as
an uncoupled module. Therefore, I believe that multiprocessing-lib in
python will fit very well for our proposes.

For the streaming system to work smoothly in a low-bandwidth scenario we
need to choose the protocol wisely. In the recent years the WebRTC
protocol has been used in a myriad of applications like google hangouts
and Google Stadia aiming low latency behavior. Therefore, I choose the
webrtc as my first protocol to be available in the streaming system
proposal.

To achieve the third property, we must be economical in adding
requirements and dependencies.

Currently, the system has some issues, but it's already working. You can
see some tutorials about how to use this streaming system
`here <https://github.com/devmessias/fury/tree/feature_fury_stream_client/docs/tutorials/04_stream>`__.
After running one of these examples you can easily share the results and
interact with other users. For example, using the ngrok For example,
using the ngrok

::

     ./ngrok http 8000  
    

| 

How does it works?
------------------

The image below it's a simple representation of the streaming system.

|image1|

As you can see, the streaming system is made up of different processes
that share some memory blocks with each other. One of the hardest part
of this PR was to code this sharing between different objects like VTK,
numpy and the webserver. I'll discuss next some of technical issues that
I had to learn/circumvent.

Sharing data between process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We want to avoid any kind of unnecessary duplication of data or
expensive copy/write actions. We can achieve this economy of
computational resources using the multiprocessing module from python.

multiprocessing RawArray
^^^^^^^^^^^^^^^^^^^^^^^^

| The
  `RawArray <https://docs.python.org/3/library/multiprocessing.html#multiprocessing.sharedctypes.RawArray>`__
  from multiprocessing allows to share resources between different
  processes. However, there are some tricks to get a better performance
  when we are dealing with RawArray's. For example, `take a look at my
  PR in a older
  stage. <https://github.com/devmessias/fury/tree/6ae82fd239dbde6a577f9cccaa001275dcb58229>`__
  In this older stage my streaming system was working well. However, one
  of my mentors (Filipi Nascimento) saw a huge latency for
  high-resolutions examples. My first thought was that latency was
  caused by the GPU-CPU copy from the opengl context. However, I
  discovered that I've been using RawArray's wrong in my entire life!
| See for example this line of code
  `fury/stream/client.py#L101 <https://github.com/devmessias/fury/blob/6ae82fd239dbde6a577f9cccaa001275dcb58229/fury/stream/client.py#L101>`__
  The code below shows how I've been updating the raw arrays

::

   raw_arr_buffer[:] = new_data

This works fine for small and medium sized arrays, but for large ones it
takes a large amount of time, more than GPU-CPU copy. The explanation
for this bad performance is available here : `Demystifying sharedctypes
performance. <https://stackoverflow.com/questions/33853543/demystifying-sharedctypes-performance>`__
The solution which gives a stupendous performance improvement is quite
simple. RawArrays implements the buffer protocol. Therefore, we just
need to use the memoryview:

::

   memview(arr_buffer)[:] = new_data

The memview is really good, but there it's a litle issue when we are
dealing with uint8 RawArrays. The following code will cause an exception:

::

   memview(arr_buffer_uint8)[:] = new_data_uint8

There is a solution for uint8 rawarrays using just memview and cast
methods. However, numpy comes to rescue and offers a simple and a 
generic solution. You just need to convert the rawarray to a np
representation in the following way:

::

   arr_uint8_repr = np.ctypeslib.as_array(arr_buffer_uint8)
   arr_uint8_repr[:] = new_data_uint8

You can navigate to my repository in this specific `commit
position <https://github.com/devmessias/fury/commit/b1b0caf30db762cc018fc99dd4e77ba0390b2f9e>`__
and test the streaming examples to see how this little modification
improves the performance.

Multiprocessing inside of different Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Serge Koudoro, who is one of my mentors, has pointed out an issue of the
streaming system running in MacOs. I don't know many things about MacOs,
and as pointed out by Filipi the way that MacOs deals with
multiprocessing is very different than the Linux approach. Although we
solved the issue discovered by Serge, I need to be more careful to
assume that different operating systems will behave in the same way. If
you want to know more,I recommend that you read this post `Python:
Forking vs
Spawm <https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/>`__.
And it's also important to read the official documentation from python.
It can save you a lot of time. Take a look what the
official python documentation says about the multiprocessing method

|image2| Source:\ https://docs.python.org/3/library/multiprocessing.html

.. |image1| image:: https://user-images.githubusercontent.com/6979335/121934889-33ff1480-cd1e-11eb-89a4-562fbb953ba4.png
.. |image2| image:: https://user-images.githubusercontent.com/6979335/121958121-b0ebb780-cd39-11eb-862a-37244f7f635b.png
