Week 8: Another week with TextBlockUI
=====================================

.. post:: July 22, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc

What did you do this week?
--------------------------
This week, I delved deeper into the **TextBlock2D** Bounding Box PR to address the challenges with tests and offsetting issues. In a pair programming session with my mentor, we discovered that the offsetting background problem stemmed from the dynamic nature of the bounding box. The issue arose when the **RingSlider2D** component began with an initial text size larger than the current text, which changed as the value was adjusted between 0-100%. This resulted in problems with offsetting and shrinking the bounding box. To resolve this, we decided to make the dynamic bounding box an optional feature.

Now, the **TextBlock2D** component offers three main features:

1. Completely static background
2. Dynamic bounding box scaled according to the text
3. Font scaling based on the bounding box

After tweaking and testing, all the features work seamlessly.

Did you get stuck anywhere?
----------------------------
The pair programming session with my mentor proved to be immensely helpful, as it guided me through the whole week.

What is coming up next?
------------------------
I will dedicate time to further enhancing the **TreeUI**. My focus will be on updating tree nodes and ensuring proper node positioning during movement.
