Week #11: Finalizing open Pull Requests
=======================================

.. post:: August 16 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

What did I do this week?
------------------------
Below are the tasks that I worked on:

* `Created PR for sprite sheet animation <https://github.com/fury-gl/fury/pull/491>`_ : This PR adds support for playing animations from a sprite sheet. This feature will be used in Card2D to create a tutorial in which the card will show the animation in the image box. Previously, the utility functions for this were added directly inside the tutorial but now they are refactored to go in their respective modules.
* `Finalized the x, y, z layouts <https://github.com/fury-gl/fury/pull/486>`_ : The PR that adds these layouts needed some updates for it to work as intended. These changes were added and this PR is ready to go.
* `Resolved all conflicts in the GridLayout PR <https://github.com/fury-gl/fury/pull/443>`_ : As the Horizontal and Vertical layouts were merged this week the GridLayout PR had got some conflicts. These conflicts were resolved and the PR is almost ready.
* **Continuing the work on custom font rendering** : In the last meeting, a few points were brought up. Firstly, to position each glyph to their respective location in the atlas a seperate module is used which is freetype-gl. The python bindings for this module are not available which means either we have to write the bindings ourselves or the freetype team will be emailed about this and they will add bindings for that. On the other hand, I looked how latex is rendered in matplotlib. `This <https://github.com/matplotlib/matplotlib/blob/3a4fdea8d23207d67431973fe5df1811605c4132/lib/matplotlib/text.py#L106>`_ is the Text class that is used to represent the string that is to be drawn and `This <https://github.com/matplotlib/matplotlib/blob/3a4fdea8d23207d67431973fe5df1811605c4132/lib/matplotlib/artist.py#L94>`_ is the class that it inherits from. Everything is handled internally in matplotlib, to draw the rasterized text `this <https://github.com/matplotlib/matplotlib/blob/3a4fdea8d23207d67431973fe5df1811605c4132/lib/matplotlib/text.py#L672>`_ function is used. The text can be rendered in two ways, the first one is by using the default renderer and the second way is by using PathEffectRenderer that is used to add effects like outlines, anti-aliasing etc. It is a very rigid way of rendering text and is designed to be used internally.

Did I get stuck anywhere?
-------------------------
No, I did not get stuck anywhere.

What is coming up next week?
----------------------------
Hopefully everything is resolved by the end of this week and next week I will hopefully submit my final code in a gist format.

**See you guys next week!**