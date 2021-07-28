Weekly Check-In #1
==================

.. post:: June 08 2021
   :author: Bruno Messias
   :tags: google
   :category: gsoc

Hi everyone! My name is Bruno Messias currently I'm a Ph.D student at
USP/Brazil. In this summer I'll develop new tools and features for
FURY-GL. Specifically, I'll focus into developing a system for
collaborative visualization of large network layouts using FURY and VTK.

What did I do this week?
------------------------

In my first meeting the mentors explained the rules and the code of
conduct inside the FURY organization. We also made some modifications in
the timeline and discussed the next steps of my project. I started
coding during the community bonding period. The next paragraph shows my
contributions in the past weeks

-  `A FURY/VTK webrtc stream system proposal:`_ to the second part of my
   GSoC project I need to have a efficiently and easy to use streaming
   system to send the graph visualizations across the Internet. In
   addition, I also need this to my Ph.D. Therefore, I’ve been working a
   lot in this PR. This PR it’s also help me to achieve the first part
   of my project. Because I don’t have a computer with good specs in my
   house and I need to access a external computer to test the examples
   for large graphs.
-  Minor improvements into the `shader markers PR`_ and `fine tunning
   open-gl state PR`_.

Did I get stuck anywhere?
-------------------------

I’ve been stuck into a performance issue (copying the opengl framebuffer
to a python rawarray) which caused a lot of lag in the webrtc streamer.
Fortunately, I discovered that I’ve been using rawarrays in the wrong
way. My `commit`_ solved this performance issue.

What is coming up next?
-----------------------

In this week I'll focus on finish the #432 and #422 pull-requests.

.. _`A FURY/VTK webrtc stream system proposal:`: https://github.com/fury-gl/fury/pull/437
.. _shader markers PR: https://github.com/fury-gl/fury/pull/422
.. _fine tunning open-gl state PR: https://github.com/fury-gl/fury/pull/432/
.. _commit: https://github.com/fury-gl/fury/pull/437/commits/b1b0caf30db762cc018fc99dd4e77ba0390b2f9e%20
