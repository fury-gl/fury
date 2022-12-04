Week 3 - Fixing fetcher, adding tests and docs
==============================================

.. post:: July 04 2022
   :author: Shivam Anand
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- The first task for this week was to fix the glTF fetcher. We noticed that while running tests it reaches the API limit. So we decided to use a JSON file that contains the download URLs for all models.

- Created a function to generate JSON files with download URLs for all models, which can be found `here <https://github.com/xtanion/fury/blob/gltf-json-gen/fury/data/fetcher.py#L330>`_.

- Modified the tests and ``fetcher.py`` to download using the JSON URLs and merged the PR `#616 <https://github.com/fury-gl/fury/pull/616>`_.

- Added docstring for all functions in `#600 <https://github.com/fury-gl/fury/pull/600>`_. Wrote tests for transformation functions.

- Multiple actor support in glTF export. Added tutorial and created a new branch for the same. Here's an example: (The glTF model below is created using FURY and rendered back using the glTF class)

.. image:: https://github.com/xtanion/Blog-Images/blob/main/Screenshot%20from%202022-07-05%2014-51-06.png?raw=true
   :width: 500
   :align: center


What is coming up next week?
----------------------------

I will be doing the following:

- We still need to figure out how to get indices from ``vtkSource`` actors
- I'll be finishing the exporting functions and adding tests for the same and I should be able to create a mergeable PR for the exporting function.

Other tasks will be decided after the meeting.


Did you get stuck anywhere?
---------------------------

No