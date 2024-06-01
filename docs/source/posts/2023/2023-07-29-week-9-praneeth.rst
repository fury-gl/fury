Week 9: TextBlock2D is Finally Merged!
======================================

.. post:: July 29, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc

What did you do this week?
--------------------------
Continuing from the previous week, it seemed like we were almost done with the *TextBlock2D*, but there remained a final task of addressing conflicting issues. Being a core part of the UI, *TextBlock2D* had a few compatibility problems with certain other UI elements.

The default behavior of *TextBox2D* now includes a dynamic bounding box, which scales automatically based on the contained text. Users can customize this option through a simple flag setting. However, this change affected some UI elements like *Combobox2d*, which relied on the default textbox size. Consequently, I had to make updates to ensure compatibility. Additionally, the default initialization of the *TextBlock2D* was completely static, which led to the possibility of the text extending beyond the background and failing certain tests. To tackle this, I made adjustments to the overflow helper function in the *test_elements.py* file. After a few tweaks and issue resolutions, the PR was ready for review and was successfully merged after passing the review process.

.. image:: https://user-images.githubusercontent.com/64432063/258603191-d540105a-0612-450e-8ae3-ca8aa87916e6.gif
   :align: center
   :alt: TextBlock2D with different attributes

Did you get stuck anywhere?
----------------------------
I encountered some peculiar test failures that were indirectly related to the *TextBlock2D* which at first glance didn't came up. Although after some debugging and a thorough line-by-line analysis, I managed to identify and resolve them.

What is coming up next?
------------------------
My next priority will be completing the *SpinBoxUI* now that the *TextBlock2D* is fixed and successfully integrated.
