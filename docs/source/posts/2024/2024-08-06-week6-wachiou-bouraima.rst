WEEK 6: Code reviews, relining and crush challenges
===================================================

.. post:: August 6, 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

Hello everyone,
As my Google Summer of Code (GSoC) 2024 journey progresses, week6 has brought me a series of technical challenges and accomplishments. My main focus has been on code reviews, rebasing and commits squashing, with a few notable lessons learned along the way.


Code Reviews and Merging ``Pull Requests``
-------------------------------------------

One of the main activities this week was receiving and addressing feedback on several of my pull requests. Notably, my mentor `Serge Koudoro <https://github.com/skoudoro/>`__, reviewed and merged the PRs related to the ``warn_on_args_to_kwargs`` decorator and the application of the ``warn_on_args_to_kwargs`` decorator across various modules. The merging of these PRs was a critical step in ensuring that our codebase adhered to the project's evolving standards for clarity and maintainability.


Rebasing and Squashing: Overcoming Challenges
---------------------------------------------

- I performed rebasing and squashing to integrate the latest changes and consolidate commits. This process was challenging due to several conflicts that arose. Resolving these conflicts required a deep dive into Git’s functionality, including:

  - **Conflict Resolution:** Manually resolving merge conflicts that affected several files.

  - **Understanding Git Operations:** Gained hands-on experience with rebasing and squashing, which improved my grasp of version control workflows.

  - **Commit Consolidation:** multiple commits into a single commit to streamline the commit history and enhance readability.

Here are some of the Git commands I used during the rebasing and squashing process:

.. code-block:: bash

    # Rebase the branch onto the upstream/master branch

    git rebase upstream/master -xtheirs

    # Squash the last "n" commits into a single commit
    git rebase -i HEAD~n

    # Continue the rebase process after resolving conflicts
    git rebase --continue

    git rebase --abort

    # Amend the last commit with new changes
    git commit --amend

    # Push the changes to the remote repository
    git push origin branch_name --force


- Merged PRs:

  - `warn_on_args_to_kwargs`: https://github.com/fury-gl/fury/pull/888
  - `actors`: https://github.com/fury-gl/fury/pull/898
  - `animation`: https://github.com/fury-gl/fury/pull/899
  - `data`: https://github.com/fury-gl/fury/pull/900
  - `shares`: https://github.com/fury-gl/fury/pull/901
  - `ui`: https://github.com/fury-gl/fury/pull/902


Technical Insights and Lessons Learned
---------------------------------------

- **Version Control Mastery:** Through the rebasing and squashing process, I gained a deeper understanding of Git’s capabilities and the importance of maintaining a clean commit history.


Acknowledgements
----------------

I want to extend my thanks to my mentor `Serge Koudoro <https://github.com/skoudoro/>`__, for his detailed feedback and guidance. Your support has been crucial in refining my work. I also appreciate the constructive comments from my peers: `Iñigo Tellaetxe Elorriaga <https://github.com/itellaetxe>`_, `Robin Roy <https://github.com/robinroy03>`_, `Kaustav Deka <https://github.com/deka27>`_, which have been instrumental in improving the quality of my contributions.


Did I get stuck anywhere?
--------------------------

While the rebasing and squashing process presented challenges, I was able to overcome them with the help of my mentor `Serge Koudoro <https://github.com/skoudoro>`__ and online resources and documentation.

- `How to Squash Commits in Git <https://www.git-tower.com/learn/git/faq/git-squash>`_
- `Git: Theirs vs Ours <https://dev.to/tariqabughofa/git-theirs-vs-ours-3i7h>`_
- `How to squash and rebase in git <https://youtu.be/AWayLpQHJeE?si=I-fRM0H3icvm9ua8>`_

The experience has enhanced my Git proficiency and prepared me for future code management tasks.


What's next ?
-------------

In the week 7, I plan to:

1. Review Adjustments: I'll be reviewing the feedback provided by my mentor `Serge Koudoro <https://github.com/skoudoro/>`__, on the latest changes to ensure that everything meets ``FURY``'s coding standards.
2. Finalizing Lazy Loading: Once the reviews are completed and approved, my mentor `Serge Koudoro <https://github.com/skoudoro/>`__, will merge the PR related to the lazy loading implementation. This will mark a significant milestone in optimizing the FURY codebase.
3. Sphinx Warning Fixes: I will start addressing Sphinx warnings related to typos in the blog posts to improve the documentation quality.
