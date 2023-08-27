Week 11: Bye Bye SpinBox
========================

.. post:: August 12, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
Building upon the progress of the previous week, a major milestone was reached with the merging of PR `#830 <https://github.com/fury-gl/fury/pull/830>`_. This PR added essential "getters" and "setters" for the new features of ``TextBlock``, making it easier to handle changes. This, in turn, facilitated the integration of ``SpinBoxUI`` with the updated ``TextBlock``.

However, while working on ``SpinBoxUI``, a critical issue emerged. As ``SpinBoxUI`` allows users to input characters and symbols into an editable textbox, it posed a risk of program crashes due to invalid inputs. To counter this, I introduced a validation check to ensure that the input was a valid number. If valid, the input was converted; otherwise, it reverted to the previous value. After thorough testing and review, PR `#499 <https://github.com/fury-gl/fury/pull/499>`_ was successfully merged.

.. image:: https://user-images.githubusercontent.com/64432063/261409747-511e535b-185c-4e70-aaa8-5296c93e5344.gif
   :align: center
   :width: 500
   :alt: SpinBoxUI

Meanwhile, a concern with the textbox's behavior was identified when ``SpinBoxUI`` was scaled to a larger size. Specifically, the text occasionally touched the top or bottom boundary, creating an overflow appearance. Although initial solutions were attempted, the complexity of the issue required further consideration. This issue has been documented in more detail in Issue `#838 <https://github.com/fury-gl/fury/pull/838>`_, where it is marked as a low-priority item.

.. figure:: https://user-images.githubusercontent.com/64432063/133194003-53e2dac6-31e0-444e-b7f1-a9e71545f560.jpeg
   :align: center
   :alt: TextBlock2D text positioning issue


Did you get stuck anywhere?
---------------------------
The challenge of the week centered around addressing the textbox's overflow behavior in ``SpinBoxUI``.

What is coming up next?
-----------------------
Looking ahead, the focus remains on refining the FileDialog component, as the significant progress with ``TextBlock`` and ``SpinBoxUI`` prepares us to shift attention to other aspects of development.
