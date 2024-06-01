Week 3: Resolving Combobox Icon Flaw and TextBox Justification
==============================================================

.. post:: June 17, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc

What did you do this week?
--------------------------

This week, I tackled the **ComboBox2D** icon flaw, which was addressed using Pull Request (PR) `#576 <https://github.com/fury-gl/fury/pull/576>`__. The problem arose when we added a **ComboBox2D** to the **TabUI**. The **TabUI** would propagate the ``set_visibility = true`` for all its child elements, causing the combobox to appear on the screen without the icon change. To fix this issue, PR `#768 <https://github.com/fury-gl/fury/pull/768>`__ updated the ``set_visibility`` method of the UI class, ensuring that the icon change was applied correctly.

.. figure:: https://user-images.githubusercontent.com/98275514/215267056-a0fd94d9-ae6d-4cf8-8475-ab95fe0ef303.png
   :align: center
   :alt: Combobox Icon

Next, I focused on the textbox justification. As discussed in our meeting, I added a new property called **boundingbox** to the **TextBlock2D**. However, I encountered a problem when resizing the **TextBlock2D**. The ``vtkTextActor`` property would switch from ``SetTextScaleModeToNone`` to ``SetTextScaleModeToProp``, which would scale the font according to the position1 (lower left corner) and position2 (upper right corner) of the UI. This inconsistency in font scaling resulted in misaligned text actors. I spent some time investigating this issue, and you can find my progress in the ongoing PR `#803 <https://github.com/fury-gl/fury/pull/803>`__.

Additionally, I started working on creating a Scrollbar component by inheriting the **LineSlider2D**. I made adjustments to the position and other attributes to make it function as a scrollbar. However, I encountered some confusion regarding how to separate the scrollbar component from other elements and determine what should be included in the scrollbar itself.

.. figure:: https://github.com/fury-gl/fury/assets/64432063/d9c8d60e-3ade-49ff-804a-fd0b340b0b24
   :align: center
   :alt: Scrollbar

Did you get stuck anywhere?
---------------------------

I faced a challenge while working on the text justification. It took me several days to identify the root cause of the occasional malfunctioning of the **TextBlock2D**. At last, I found out the reason behind the issue.

What is coming up next?
-----------------------

Next week, I have several tasks lined up. Firstly, I will be working on the **CardUI** PR `#398 <https://github.com/fury-gl/fury/pull/398>`__. Additionally, I plan to complete the segregation of the scrollbar component, ensuring its independence and clarity. Lastly, I will be working on issue `#540 <https://github.com/fury-gl/fury/pull/540>`__, which involves updating the use of the ``numpy_to_vtk_image_data`` utility function.
