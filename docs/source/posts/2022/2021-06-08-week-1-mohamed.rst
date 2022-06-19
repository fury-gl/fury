First week of coding!
=====================

.. post:: June 19 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do during the Community Bonding Period?
----------------------------------------------------

During the community bonding period, I got to know the mentors better and bonded with Shivam and Praneeth, my fellow contributors.
We talked about the importance of open-source contributing and discussed scientific visualization, how important it is and how it differs from game engines despite having some similarities.
We also discussed how important the code review is and how to do a proper code review.

|
What did you do this week?
--------------------------

I continued implementing the Timeline class. Added some functionalities according to what we discussed in the weekly meeting, such as handling Fury Actors, adding, removing, translating, scaling them according to the keyframes. Then made two tutorials for it with simple UI to control the playback of the animation.

Reviewed Bruno’s PR `#424`_ as Filip advised to help me figure out hot to change uniforms' values during runtime. I found it straightforward. I also found a similar method already used in some tutorials and experiments to set the time uniform, which is based on the same idea. They both are using shaders callback functions.

Going through the VTk's documentations, I found a more efficient way to do this. It's newly implemented in VTK 9, does not require callbacks, and is easy to use on individual actors. I had to verify first that Fury does not support an older version of VTK and it does not. I tested it and ran into some issues so As Filip instructed I don't waste a lot of time with uniforms so I postponed it for the next week.

Also, when I was attending Shivam’s meeting, he showed us a smooth glTF model which differed from the last time I saw it. When I asked him he said he applied the normal. Him saying that reminded me of how we can use normals to smooth shade a model. I applied this trick to the sphere actor in this PR `#604`_.

|
What is coming up next week?
----------------------------

If the implemented timeline is good to go, I will write the associated tests, finish up any missing methods, and document it properly.

I will probably continue implementing the other interpolation methods.

|
Did you get stuck anywhere?
---------------------------

I got stuck changing the color of some actors.

.. _`#424`: https://github.com/fury-gl/fury/pull/424
.. _`#604`: https://github.com/fury-gl/fury/pull/604

.. |normals| image:: https://user-images.githubusercontent.com/63170874/173938868-8336c2c9-37a0-4eb1-a0f2-f76c7275e767.png