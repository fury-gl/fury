Week 4 - Finalizing glTF loader
===============================
.. post:: July 12 2022
   :author: Shivam Anand
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

This week I had to travel back to my college since the summer vacations have ended.
I had a coding session with Serge this week, we modified the exporter function and looked at some bugs that I faced.

We managed to use the ``io.load_io`` function. There was a strange issue with loading png files. I had to convert the PIL ``Image`` to a ``rgb`` format and It fixed the issue, turns out png images are stored as ``P`` (pallet) mode.

I also added the ``glb`` format support to the importer. While loading a ``.glb`` model, I noticed that the image data is also stored in the buffer and there can be a ``bufferview`` index to get the buffer.

During this time I also tested the glTF loader with all models in the ``KhronoosGroup/glTF-samples`` repository. Here's the table of models that are working well: To be added.


What is coming up next week?
----------------------------

- Adding tests and merging export function PR.
- Start working on Simple Animations.

Other tasks will be decided after the meeting.


Did you get stuck anywhere?
---------------------------

- To create a texture we needed the RGB values, However ``.png`` images were returning a 2D array when read using PIL. It is fixed by 
   .. code-block :: python

         if pil_image.mode in ['P']:
            pil_image = pil_image.convert('RGB')


- pygltflib's ``load`` method doesnot handle glb files very well, It does not contain the buffer ``uri``. I used ``glb2gltf`` method as of now.