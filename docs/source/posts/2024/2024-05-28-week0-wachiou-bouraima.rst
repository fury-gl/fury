WEEK 0: The beginning of my journey in Google Summer of Code (GSoC) 2024
========================================================================

.. post:: May 28, 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

Here we go.....
~~~~~~~~~~~~~~~

| Hello and welcome to my GSoC 24 journey, I'm Wachiou Bouraima pronounced (Wasiu Ibrahima).
|
| First of all, I'd like to express my deep gratitude for this immense opportunity.
| In this first article, yes you read "first article" correctly, as it won't be the only one, I'm going to share with you my first adventures in GSoC'24. Happy read


Welcome and integration into the community.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like any good start, this week I have the incredible opportunity to be welcomed into the community by the Core Team and my Mentor himself, during this session the Mentors welcomed us warmly and also congratulated us. They shared with us the rules respected during and after the GSoC program. This session made me feel comfortable and confident.

I also have to admit that the mentors are experienced and willing to share their knowledge, which I really appreciate. I also got to know the DIPY and FURY community members.

Not only that, but I also attended the GSoC'24 summit organised by Google. This session was very informative and aimed to ease our way into GSoC'24, and help us avoid certain mistakes during and after the program.
I was able to meet the other GSoC students, and I must say that they are all very talented and motivated. I am very happy to be part of this community.


Project details
~~~~~~~~~~~~~~~

I will work on the project: **Modernization of the FURY code base to improve readability, maintainability and performance**

This project aims to modernize the FURY code base by implementing keyword-only arguments to improve code clarity and explicit parameter passing. In addition, the integration of lazy loading functionality will optimize performance by loading resources only when they are needed. Finally, active participation in code refactoring efforts will improve the structure and maintainability of the FURY code base. The project will result in a modernized code base, comprehensive unit testing, updated Sphinx documentation and public presentations illustrating the improvements and benefits. Ultimately, the aim is to significantly improve the FURY code base for future developers and users.


Weekly tasks
~~~~~~~~~~~~

I had to work on my first mission of the adventure, which was to create a decorator named **keyword_only** to ensure that all arguments after the first are keyword arguments only. It also checks that all keyword arguments are expected by the function. You can check what I had to do in this Pull Request : https://github.com/fury-gl/fury/pull/888. I've learned a lot from implementing the **keyword_only** decorator


What's next?
~~~~~~~~~~~~

For my next task, I'll first apply the advice and comments from my first task, adding the **keyword_only** decorator after all the necessary reviews and member approval, on all the functions concerned next, then I'll start and finish, integrating the lazy loading feature.

🥰 Thank you for reading. Your comments are most welcome, and I learn from them.
