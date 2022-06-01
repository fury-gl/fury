My Journey to GSoC 2022
=======================

.. post:: May 24 2022
    :author: Shivam Sahu
    :tags: google
    :category: gsoc


About Myself
~~~~~~~~~~~~

Hi! I'm Shivam, currently pursuing my bachelor's (expected 2024) in Production and Industrial engineering from the Indian Institute of Technology (IIT) Roorkee.

I was introduced to the C programming language through the Introduction to programming course in my first year of college. I always liked computers, and I was overwhelmed by printing "hello world" in the terminal. The course continued to teach us data structures and algorithms. It was a lot of fun competing with friends over a programming question. I learned python on my own and did a lot of projects. After learning basic programming, I participated in many hackathons and won some, but lost a lot. Coding was enjoyable, and I knew this was what I wanted to do for the rest of my life.
In my second semester, I learned about open-source, git, and GitHub. I came across some computer graphics enthusiasts in my college; they told me that they created a 2D game engine on their own using OpenGL. This gave me enough motivation to learn OpenGL and basic computer graphics. 

Intro to Open-Source and GSoC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In October 2021, I participated in Hactoberfest and completed it successfully. I learned a lot about Free and Open Source Software during this time. I heard about GSoC around this time from one of my seniors and asked him about the program.
 
I went through the previous year's accepted organizations list and shortlisted 2-3 organizations. I started contributing to a few projects based on android and kotlin, but I lost interest after some time.
It was about December, and I hadn't chosen any organization yet.

I heard about FURY from one of my seniors. I started looking at its docs. Picked up some books from the library to brush up on my concepts of computer graphics. The documentation of FURY is remarkable. I created my `first pull request <https://github.com/fury-gl/fury/pull/520>`_ by improving one of the tutorials, and It got merged in a few days. 

I started looking at the source code of FURY and tried to understand it again. I read the `API reference` part of FURY docs this time along with the documentation of `VTK`. I also tried to solve some of the open issues. I created a few pull requests for the same. My first contribution was in the fury `primitive` class, I created a sphere primitive (inspired by the sphere in Blender).

I started looking at bugs and improvements that can be made to the source code and created PRs for the same. During this time I was also learning how computer graphics work from the `UCSC lectures <https://www.youtube.com/channel/UCSynd9Z5RdIpKfvTCITV_8A/videos>`_  & `OpenGL by Victor Gordon <https://youtube.com/playlist?list=PLPaoO-vpZnumdcb4tZc4x5Q-v7CkrQ6M->`_.

After the accepted organizations were announced, I was so happy that FURY got selected this year for GSoC and went straight to the Ideas list. The first project idea was `glTF Integration`. I heard about the `glTF` file format before but didn't know how it works. So I went straight through the reference materials provided on the `wiki <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2022-(GSOC2022)>`_. I read a lot of documentation by the Khronos group. I also tried to implement a basic glTF loader during this time using VTK's built-in `glTFReader`.

I also liked the Improve UI drawing project idea (I had a basic understanding of line drawing algorithms and rasterizations at that time) and thought I'll make a proposal for that too. After completing my proposal for glTF integration I started my research on this one too.

I started writing the proposal early so that I could get my proposal reviewed at least once with the project mentors. I almost forgot the fact that I had my end-term examinations at hand (Somehow I managed to pass all of my courses), I kept contributing to the project till the end of April 2022.

Code contributions:

1. [https://github.com/fury-gl/fury/pull/520]
2. [https://github.com/fury-gl/fury/pull/525]
3. [https://github.com/fury-gl/fury/pull/533]
4. [https://github.com/fury-gl/fury/pull/547]
5. [https://github.com/fury-gl/fury/pull/556]
6. [https://github.com/fury-gl/fury/pull/559]

The Day
~~~~~~~
May 18: I was a bit anxious since on May 18th the GSoC website was showing my proposal for glTF integration had been rejected. (p.s. maybe it was a bug of some sort that they fixed later on).

May 20: I woke up very early in the morning and started checking the contributor's profile. I kept checking for new emails but didn't receive any emails that day. "The result will be announced at 11:30 PM, isn't it? " My dad said, It was 8:00 AM IST. I called my friends in the night to join me on discord, I was talking to them and refreshing the GSoC site at the same time. One of my friends shouted that he got selected in NumFocus. I refreshed the page again, my proposal for glTF Integration was accepted. I can't express what I felt at that moment. I told my friends and my parents, they were happy for me and I got a lot of blessings :). I received an official email the next day.
