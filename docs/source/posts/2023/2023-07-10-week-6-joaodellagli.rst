Week 6: Things are Starting to Build Up
=======================================

.. post:: July 10, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc


Hello everyone, time for a other weekly blogpost! Today, I will show you my current progress on my project and latest activities.

What I did Last Week
--------------------
Last week I had the goal to implement KDE rendering to the screen (if you want to understand what this is, check my :doc:`last blogpost <2023-07-03-week-5-joaodellagli>`_).
After some days diving into the code, I finally managed to do it:

.. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/buffer_compose.png
   :align: center
   :alt: KDE render to a billboard

This render may seem clean and working, but the code isn't exactly like that. For this to work, some tricks and work arounds needed to
be done, as I will describe in the section below.

Also, I reviewed the shader part of Tania's PR `#791 <https://github.com/fury-gl/fury/pull/791>`_, that implement ellipsoid actors inside
FURY. It was my first review of a PR that isn't a blogpost, so it was an interesting experience and I hope I can get better at it.

It is important as well to point out that I had to dedicate myself to finishing my graduation capstone project's presentation that I will attend
to this week, so I had limited time to polish my code, which I plan to do better this week.

Where the Problem Was
---------------------
The KDE render basically works rendering the KDE of a point to a texture and summing that texture to the next render. For this to work,
the texture, rendered to a billboard, needs to be the same size of the screen, otherwise the captured texture will include the black background.
The problem I faced with that is that the billboard scaling isn't exactly well defined, so I had to guess for a fixed screen size
(in this example, I worked with *600x600*) what scaling value made the billboard fit exactly inside the screen (it's *3.4*). That is far from ideal as I
will need to modularize this behavior inside a function that needs to work for every case, so I will need to figure out a way to fix that
for every screen size. For that, I have two options:

1. Find the scaling factor function that makes the billboard fit into any screen size.
2. Figure out how the scaling works inside the billboard actor to understand if it needs to be refactored.

The first seems ok to do, but it is kind of a work around as well. The second one is a good general solution, but it is a more delicate one,
as it deals with how the billboard works and already existing applications of it may suffer problems if the scaling is changed.
I will see what is better talking with my mentors.

Another problem I faced (that is already fixed) relied on shaders. I didn't fully understood how shaders work inside FURY so I was
using my own fragment shader implementation, replacing the already existing one completely. That was working, but I was having an issue
with the texture coordinates of the rendering texture. As I completely replaced the fragment shader, I had to pass custom texture coordinates
to it, resulting in distorted textures that ruined the calculations. Those issues motivated me to learn the shaders API, which allowed me
to use the right texture coordinates and finally render the results you see above.


This Week's Goals
-----------------
For this week, I plan to try a different approach Filipi, one of my mentors, told me to do. This approach was supposed to be the original
one, but a communication failure lead to this path I am currently in. This approach renders each KDE calculation into its own billboard,
and those are rendered together with additive blending. After this first pass, this render is captured into a texture and then rendered to
another big billboard.

Also, I plan to refactor my draft PR `#804 <https://github.com/fury-gl/fury/pull/804>`_ to make it more understandable, as its description still dates back to the time I was using the
flawed Framebuffer implementation, and my fellow GSoC contributors will eventually review it, and to do so, they will need to understand it.

Wish me luck!
