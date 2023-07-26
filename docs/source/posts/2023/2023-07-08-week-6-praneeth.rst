Week 6: BoundingBox for TextBlock2D!
====================================

.. post:: July 08, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week, I worked on improving the **TextBlock2D** component in the UI system. I started from scratch to address alignment and scaling issues. When resizing the **TextBlock2D**, the text alignment and justification with the background rectangle were inconsistent. To resolve this, I introduced a new "boundingbox" property that calculates the text bounding box based on its content. Additionally, I separated the scaling mode from the resizing action with the new "auto_font_scale" property, enabling automatic font scaling according to the bounding box. This will provide better alignment, justified text, and smoother font scaling for the **TextBlock2D** component. Try it out at `PR #803 <https://github.com/fury-gl/fury/pull/803>`_.

.. image:: https://github.com/fury-gl/fury/assets/64432063/94212105-7259-48da-8fdc-41ee987bda84
   :align: center
   :alt: TextBlock2D will different justifications

As discussed last week, we also made a decision regarding the scrollbar. After exploring different use cases, we concluded that creating an independent scrollbar is not necessary at the moment. Therefore, we will close the related pull requests. You can find out more about it in the discussion `here <https://github.com/fury-gl/fury/discussions/816>`_.

Did you get stuck anywhere?
---------------------------
Implementing the bounding box feature took some extra time as I needed to carefully consider its impact on other UI elements that rely on the **TextBlock2D** component.

What is coming up next?
-----------------------
Next, I will focus on completing the TextBlock2D Bounding Box PR, which will also indirectly finalize the Spinbox PR.
