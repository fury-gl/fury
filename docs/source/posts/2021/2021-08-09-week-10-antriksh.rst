Week#10: Accordion UI, Support for sprite sheet animations
======================

.. post:: August 09 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

What did I do this week?
------------------------
Below are the tasks that I worked on:

* `Added Accordion2D to UI sub-module <https://github.com/fury-gl/fury/pull/487>`_ : This PR adds the Accordion UI to the UI sub-module. This UI inherits from the Tree2D UI and can only be merged once the Tree2D UI is in. Here's a screenshot for reference:

    .. image:: https://i.imgur.com/klI4Tb5.png
        :width: 200
        :height: 200

* `Adding X, Y, Z Layouts <https://github.com/fury-gl/fury/pull/486>`_ :  It was pointed out in last week's meeting that in 3D space horizontal/vertical means nothing. Instead X, Y, Z are used, so, these three layouts were added on top of horizontal/vertical layouts. They also have functionality of changing the direction i.e. reverse the stacking order.
* `Added support of sprite sheet animation in Card2D <https://github.com/fury-gl/fury/pull/398>`_ : The image in Card2D was static in nature and wasn't very interesting. So, to make things a bit interesting support for animated images were added. These animations are played from a sprite sheet or a texture atlas. A buffer of processed sprite chunks is maintained and with the help of a timer callback the image in the card is updated after a certain delay which is dependent of the frame rate. Below is the demonstration:

    .. image:: https://i.imgur.com/DliSpf0.gif
        :width: 200
        :height: 200

* **Researching more about Freetype/Freetype-GL**: Apart from coding stuff, i did some more research on custom font using freetype and freetype-gl. I found some examples that used the python bindings of the c++ library and displayed custom fonts that were transformable i.e. can be rotated by some angle. Hopefully I can create a working example by this weeks meeting.

Did I get stuck anywhere?
-------------------------
No, I did not get stuck anywhere.

What is coming up next week?
----------------------------
Next week I will finish up my remaining work. Which includes addressing all PR reviews and adding some more features.

**See you guys next week!**