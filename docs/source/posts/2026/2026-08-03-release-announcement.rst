
FURY 2.0.0 Released
===================


.. post:: August 03, 2026
   :author: maharshi-gor
   :tags: fury
   :category: release


The FURY project is happy to announce the release of FURY 2.0.0!
FURY is a free and open source software library for scientific visualization and 3D animations.

You can show your support by `adding a star <https://github.com/fury-gl/fury/stargazers>`_ on FURY github project.

This is the first stable release of FURY built on `pygfx`/`wgpu`, replacing the legacy VTK-based rendering pipeline. The **major highlights** of this release are:

.. include:: ../../release_notes/releasev2.0.0.rst
    :start-after: --------------
    :end-before: Details

.. note:: The complete release notes are available :ref:`here <releasev2.0.0>`

If you are upgrading from FURY 0.12.x, please see the :doc:`../../migration_guide_v2.0` for details
on breaking changes and how to update your code.

**To upgrade or install FURY**

Run the following command in your terminal::

    pip install --upgrade fury

or::

    conda install -c conda-forge fury


**Questions or suggestions?**

For any questions go to http://fury.gl, or send an e-mail to fury@python.org
We can also join our `discord community <https://discord.gg/6btFPPj>`_

We would like to thanks to :ref:`all contributors <community>` for this release:

.. include:: ../../release_notes/releasev2.0.0.rst
    :start-after: commits.
    :end-before: We closed


On behalf of the :ref:`FURY developers <community>`,

- Maharshi Gor
