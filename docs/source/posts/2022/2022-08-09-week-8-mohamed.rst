Week 8: Back to the shader-based version of the `Timeline`
==========================================================

.. post:: August 9 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- First, I made some modifications to the main animation PR. Then as Filipi requested, I integrated the vertex shader that I was implementing a few weeks ago to work with the new improved version of the ``Timeline``.

- As for how keyframes are sent to the GPU, the method being used is to send the needed keyframes for each draw. This is heavy because we only roll out the interpolation part, which with linear or step interpolation won't make any difference! Instead, I tried setting all the keyframes at once as a GLSL variable using string manipulation before animation starts. This also was slow to initialize, and the shader was getting bigger and slower to bind and unbind. To solve this problem, I made a uniform that holds all the keyframes of the animation and sent data as vectors, which was faster than string manipulation, also it was faster to render since data are now stored directly in memory, and the shader program was a lot more compact. But this method had an issue; uniforms do not keep data stored as expected! If two or more actors have the same uniform name in their shader program and only one of them was set, the other actor will get this value as well. A way around this is to change the names of the uniforms so that they maintain their data.

- Tested animating 160K billboard spheres animation but this time using translation as well. And they performed very well.

    .. raw:: html

        <iframe id="player" type="text/html"   width="440" height="390" src="https://user-images.githubusercontent.com/63170874/183534269-73bf6158-cd54-4011-9742-0483d67ca802.mp4" frameborder="0"></iframe>


What is coming up next week?
----------------------------

- Fix issues I encountered this week working with GLSL shaders.

- Implement slerp in GLSL as well as figure out a way to optimize sending keyframes to the shader program.

- Figure a way to animate primitives of the same actor by different timelines.


Did you get stuck anywhere?
---------------------------

I had two issues, one mentioned above which was uniforms not being able to hold data. The second one is that VTK does not send specular-related uniforms and ``lightColor0`` to the ``point`` primitive, which are needed for light calculations.
