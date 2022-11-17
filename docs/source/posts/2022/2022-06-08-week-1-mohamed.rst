Week 1: Implementing a basic Keyframe animation API
===================================================

.. post:: June 19 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do during the Community Bonding Period?
----------------------------------------------------

During the community bonding period, we had weekly meetings in which I got to know the mentors better and bonded with Shivam and Praneeth, my fellow contributors.
We talked about the importance of open-source contribution and discussed scientific visualization, how important it is and how it differs from game engines despite having some similarities.
We also discussed how important the code review is and how to do a proper code review.


What did you do this week?
--------------------------

I continued implementing the `Timeline` class. Added some functionalities according to what we discussed in the weekly meeting, such as handling Fury Actors, adding, removing, translating, scaling them according to the keyframes. Then made two tutorials for it with simple UI to control the playback of the animation.

  .. raw:: html

        <iframe id=""player" type="text/html" width="640" height="390" src="https://user-images.githubusercontent.com/63170874/174503916-7ce0554b-9943-43e3-9d5c-c97c9ce48eaf.mp4" frameborder="0"></iframe>


Reviewed Bruno's PR `#424`_ as Filipi advised to help me figure out how to change uniforms' values during runtime. I found it straightforward. I also found a similar method already used in some tutorials and experiments to set the time uniform, which is based on the same idea. They both are using shaders callback functions.
Going through the VTK's documentations, I found a more efficient way to do this. It's newly implemented in VTK 9, does not require callbacks, and is easy to use on individual actors. I had to verify first that Fury does not support an older version of VTK and it does not. I tested it and ran into some issues so As Filip instructed I don't waste a lot of time with uniforms so I postponed it for the next week.

Also, when I was attending Shivam's meeting, he showed us a smooth glTF model which differed from the last time I saw it. When I asked him he said he applied the normal. Him saying that reminded me of how we can use normals to smooth shade a shapes. I applied this trick to the sphere actor in this PR `#604`_ and worked as expected.


What is coming up next week?
----------------------------

If the implemented `Timeline` is good to go, I will write the associated tests, finish up any missing methods, and document it properly.
I will probably continue implementing the other interpolation methods as well.

Did you get stuck anywhere?
---------------------------

I got stuck changing the color of some actors.

.. _`#424`: https://github.com/fury-gl/fury/pull/424
.. _`#604`: https://github.com/fury-gl/fury/pull/604
