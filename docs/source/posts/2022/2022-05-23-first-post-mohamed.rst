My journey till getting accepted into GSoC22
============================================


.. post:: May 23 2022
   :author: Mohamed Abouagour 
   :tags: google
   :category: gsoc

A Little About Myself
~~~~~~~~~~~~~~~~~~~~~

My name is Mohamed and I'm from Egypt. I am pursuing a Bachelor of Engineering in Computer Engineering and Automatic Control (expected: 2023), Tanta University, Egypt. 

I've been around computers since 2008 when I had my first PC with 128MB RAM and ran on Windows XP (that's almost all I could remember about it). 

Around 2013, I had some questions about how games are made and how to make them myself!
There was no one to answer these questions for me.

My english wasn't any good and the game development arabic community was very hard to find on the internet. But eventually, I came across a forum with some people speaking about some stuff such as game engines, 3D models, textures, animations, and a lot of fun stuff. That made sense back then and answered some of my questions. Then It was time to get involved with this amazing community. I was lucky enough to start with 3ds Max 2009 Edition, with that little view cube begging me to touch it. It was just love at first sight.

I learned enough to model a basic house with no interior or a low poly human face.

I was very curious about these game engines. Back then, the top three game engines available for the public were `CryEngine 3 <https://www.cryengine.com/>`_, `UDK <https://en.wikipedia.org/wiki/Unreal_Engine#Unreal_Development_Kit>`_, and `Unity <https://en.wikipedia.org/wiki/Unity_%28game_engine%29#Unity_3.0_%282010%29>`_.

I was advised to use CryEngine 3 since it required no coding experience to use it. I started designing and texturing terrains and adding 3D models (I remember adding a pyramid and had a hard time scaling it down).


My first coding experience
~~~~~~~~~~~~~~~~~~~~~~~~~~
It started with C programming language, which I came across while taking Harvard's CS50. I then used it in competitive programming as my primary language.

When I heard about OpenGL API for the first time from a senior and that it is being used to develop game engines, I started learning it along with GLSL.

And it finally hit me: It is all math!

I then started practicing pipelines, lighting models such as Blinn and Blinn-Phong models, cameras, texturing, fog, skybox, shadows, etc...


Developing a 3D game
~~~~~~~~~~~~~~~~~~~~
In the Graphics course, OpenGL was being taught to us using python and there was a course project!

I started preparing my notes and the shaders I wrote during that summer, only for the Professor to restrict us to use OpenGL v1.0 and only use PyOpenGL and Pygame.

So, no Numpy, no PyGLM, fixed pipelines, no custom shaders, and each vertex had to be sent individually to the GPU.

At first, I got disappointed, but while developing the game, I had a lot of optimization challenges that made me have fun figuring out how to still make the 3D game I've always wanted to make!

I searched on GitHub for similar projects to the one I'm trying to make and did not find any 3D FPS game using this version of OpenGL (at least not listed publicly).

I ended up implementing almost everything from scratch.

Real-world physics inspired the physics of the game. This means that I have collected data about walking speed, running speed, initial jumping velocity, and gravity. Even the sound Sound intensity is inversely proportional to the square of the distance between the source and the player.

I used Interpolation (bilinear interpolation) to interpolate the terrain's height (y) at any given (x-z) coordinates so that entities can move on 64*64 pixels height map based terrain smoothly to reduce the number of triangles. I also implemented for this game an animation system that used obj sequence animations, then optimized it to load faster, and consume 95% less disk space compared to the non-optimized form.

Source code of the game:  `MummyIsland <https://github.com/m-agour/MummyIsland>`_





My journey to GSoC22
~~~~~~~~~~~~~~~~~~~~
I first knew about GSoC from a friend last year. I then went through the organization's list. There were a lot of familiar names that I recognized such as Anki, Godot, Blender.

Then I glanced at the Python software foundation. I had to go in and scroll through the sub-organizations only to come across FURY. The luminous horse got my attention. I felt the connection right away.

I went through the docs of both FURY and VTK.

I tried to contribute back then by adding the option to use the FXAA anti-aliasing algorithm for the window (I didn't notice it has already been implemented as a method for the window).

So instead,  I implemented the first mentioned shader in the GSoC 21's project idea “**Add new shader effects**” (`the fire shader <https://github.com/m-agour/Simple-Animation-System/tree/main/additional%20files/GLSL%20GSoC21%20test/shaders>`_) in GLSL using Gaussian noise.

The next step was to implement the shader to make it work with FURY actors but then I knew I'm having a mandatory training required by the University last summer. I was disappointed, and I decided to postpone until next year's GSoC.

This year when the accepted organizations were announced, I didn't bother to go through the list since I knew exactly which organization I'm going to apply for. So I just went directly to read the Ideas list for FURY.


The keyframe animation system was the perfect match! I have been around keyframes for half of my life to even guess the required features before I read them! I then started to contribute and one contribution led to the other one.

**Code contributions:**

1. https://github.com/fury-gl/fury/pull/552
2. https://github.com/fury-gl/fury/pull/555


The day I got accepted
~~~~~~~~~~~~~~~~~~~~~~
I made it my mission for the day to keep watching the email for any changes. When It got 8:00 pm in the Cairo timezone, I didn't see any changes. So I started searching for any news whatsoever and I couldn't find any. 
The GSoC dashboard doesn't say anything (story of my life. I didn't even get any promotions or spam mails that day). That's when I gave up. But, something doesn't feel right! Right? I went again to check the dashboard to find that my proposal got accepted. I couldn't even believe it. I would better not jump to conclusions and wait for the official email before celebrating. Shortly after, I received the official email. It was a whole new feeling I had never experienced ever before! I reached a whole new level of happiness.