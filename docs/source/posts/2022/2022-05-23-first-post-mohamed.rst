First Post
================================================================================

  
.. post:: May 23 2022
  :author: Mohamed Abouagour
  :tags: google
  :category: gsoc

A Little About Myself
~~~~~~~~~~~~~~~~~~~~~

I am Mohamed from Egypt. , I am pursuing a Bachelor of Engineering in Computer Engineering and Automatic Control (expected: 2023), Tanta University, Egypt. 
I've been around computers since 2008 when I had my first PC with 128MB RAM and ran on Windows XP (that's almost all I could remember about it). 
Around 2013 I started to have some questions about how games are made and how to make them myself!. 
There was no one to answer these questions for me.
My English wasn't any good and the game development Arabic community was very hard to find on the internet but eventually, I came across a forum with some people speaking about some stuff such as game engines, 3D models, textures, animations, and a lot of fun stuff that made sense back then and answered some of my questions. Then It was time to get involved with this amazing community, I was lucky enough to start with 3ds Max 2009 Edition, with that little view cube begging me to touch it. It was just love at first sight)
I learned enough to model a basic house with no interior. 
I was very curious about these game engines. Back then the top three game engines available for the public were `CryEngine 3 <https://www.cryengine.com/>`_, `UDK <https://en.wikipedia.org/wiki/Unreal_Engine#Unreal_Development_Kit>`_ and `Unity <https://en.wikipedia.org/wiki/Unity_%28game_engine%29#Unity_3.0_%282010%29>`_. I  was advised to use CryEngine 3 since it required no coding experience to use it. I started designing and texturing terrains and adding 3D models (I remember adding a pyramid and had a hard time scaling it down).


My first coding experience
~~~~~~~~~~~~~~~~~~~~~~~~~~

It started with C programming language which I came across while taking Harvard's CS50. I then used it in competitive programming as my primary language. 
When I heard about OpenGL API for the first time from a senior and that it is being used to develop game engines, I started learning it along with GLSL. 
And it finally hit me: It's all math! 
I then started practicing pipelines, lighting models such as Blinn and Blinn-Phong models, cameras, texturing, fog, skybox, shadows, etc...



Develpoing 3D game
~~~~~~~~~~~~~~~~

In the Graphics course, OpenGL was being taught to us using python and there was a course Project! 
I started preparing my pipelines only for the Professor to tell us that we are restricted to using no more than OpenGL V1.0 and only use PyOpenGL and Pygame. 
So, no Numpy, no PyGLM, fixed pipelines, no custom shaders, and each vertex had to send individually to the GPU.
At first, I was disappointed but I had a lot of optimization challenges that made me have fun figuring out how to still make the 3D game I've always to make!
I searched on GitHub and did not find any 3D FPS game using this version of OpenGL (at least not listed publicly). 
I ended up implementing almost everything from scratch. 
I used Interpolation (bilinear interpolation) to interpolate the terrain’s height (y) at any given (x-z) coordinates so that entities can move on 64*64 pixels heightmap-based terrain smoothly to reduce the number of triangles. I also implemented for this game an animation system that used obj sequence animations, then optimized it to load faster, and consume 95% less disk space compared to the unoptimized form.

 - The source code of the game:  [https://github.com/m-agour/MummyIsland]



My journey to GSoC22
~~~~~~~~~~~~~~~~~~~~
I first knew about GSoC from a friend last year. I then went through the organization's list, I saw a lot of familiar names such as Anki, Godot, 
. . 
Then I glanced at the Python software foundation. I had to go in and scroll through the sub-organizations only to come across FURY. The luminous horse got my attention I felt the connection right away.
I went through the docs of both FURY and VTK.  
I tried to contribute back then by adding the option to use FXAA anti-aliasing algorithm for the window (I didn’t notice it has already been implemented as a method for the window). 
So instead,  I implemented the first mentioned shader in the GSoC 21’s project idea “**Add new shader effects**” (the fire shader) in GLSL using Gaussian noise. 
The result is shown in the videos: 
[https://youtu.be/tj1aysPqJpY]
[https://youtu.be/CQY9c3oGJ3E]). 

The next step was to implement the shader to make it work with FURY actors but then I knew I'm having a mandatory training required by the University last summer. I was disappointed and I decided to postpone till next year's GSoC.
This year when the accepted organizations were pronounced I didn't bother to go through the list since I knew exactly which organization I'm going to apply for. So I just went directly to read the Ideas list for FURY.
The Keyframe animation system was the perfect fit! I've been arount keyframes for half of my life to even participate the required features before I read them! I then started to contribute and one contribution led to the other one.

Code contributions:

1. [https://github.com/fury-gl/fury/pull/552]
2. [https://github.com/fury-gl/fury/pull/555]




The day I got accepted
~~~~~~~~~~~~~~~~~~~~~~
I made it my mission for the day to keep watching the email for any changes. When It got 8:00 pm in the Cairo timezone I didn't see any changes. So I started searching for any news whatsoever and I couldn't find any. 
The GSoC dashboard doesn't say anything (story of my life. I didn't even get any promotions or spam mails that day). That's when I gave up. but, something doesn't feel right! right? I went again to check the dashboard to find that my proposal is accepted. but I can't even believe it. I'd better not jump to conclusions and wait for the official email before celebrating. shortly after, I received the official email It was a whole new feeling I never experienced ever before! I reached a whole new level of happiness.