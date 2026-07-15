.. _migration_guide_v2_0:

==================================================
Migration Guide: FURY 0.13.x → Master (v2.0.0)
==================================================

This document catalogs all user-level API changes introduced between FURY
``0.13.x`` and the ``master`` branch (targeting FURY ``v2.0.0``). Every
entry is grounded in the actual source code and the official release notes
spanning ``v2.0.0a1`` through ``v2.0.0a7``. No changes are documented here
unless verified against the live codebase.

.. warning::

   FURY ``master`` is a ground-up rewrite. The rendering backend has
   switched from **VTK** to **PyGfx / WGPU**. All VTK objects
   (``vtkActor``, ``vtkPolyData``, ``vtkMapper``, etc.) have been removed
   from the public API. Scripts that import or manipulate VTK objects
   directly will break without migration.


1. Rendering Backend: VTK → PyGfx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`993`, :ghpull:`953`, :ghpull:`946`,
:ghpull:`978`)

All VTK imports have been removed from the library. FURY now exposes
PyGfx ``WorldObject`` sub-classes as its actor types.

.. code-block:: python

   # FURY 0.13.x (VTK-based)
   import fury
   actor = fury.actor.sphere(centers, colors=colors)
   # Users often interacted directly with the underlying VTK object:
   prop = actor.GetProperty()  # VTK method - BREAKS in master!
   prop.SetOpacity(0.5)

   # FURY master (PyGfx-based)
   import fury
   actor = fury.actor.sphere(centers, colors=colors)
   # Actors are now PyGfx-backed; use native properties instead:
   actor.opacity = 0.5         # Native PyGfx property

**Action**: Remove all direct VTK method calls (like ``GetProperty()``, ``GetMapper()``) and ``import vtk`` calls from user scripts.
Interact with actors exclusively through the FURY public API or PyGfx
``WorldObject`` properties.


2. Color API and Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`965`, :ghpull:`1120`, :ghpull:`1097`)

* All actor creation functions and ``actor_from_primitive`` accept colors
  in **[0, 1] float** or **[0, 255] int** ranges. Values ``> 1.0`` are
  divided by ``255`` automatically via ``fury.colormap.normalize_colors``.
* Hex strings (``"#FF0000"``) are accepted everywhere colors are expected.
* Bug fixed: opacity / alpha transparency was not applied in primitive
  actors when RGBA colors were passed — now resolved.
* Bug fixed: ``point`` actor failed when ``colors=None`` was passed —
  now defaults to red.

.. code-block:: python

   # Both forms now accepted
   fury.actor.sphere(centers, colors=(255, 0, 0))    # 0-255 — auto-normalised
   fury.actor.sphere(centers, colors=(1.0, 0.0, 0.0))  # 0-1 preferred form
   fury.actor.sphere(centers, colors="#FF0000")        # hex accepted

**Action**: Prefer float ``[0, 1]`` colors in new code. Legacy integer
colors will still work but are normalized internally.


3. Actor Base Class and Per-Actor Transform API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1063`)

A new ``fury.actor.core.Actor`` base class mixin exposes spatial
transforms directly on every actor object. There is no longer a need for
``fury.transform`` module calls with an explicit actor argument (though
those still exist).

**New methods on every actor**:

.. code-block:: python

   # FURY 0.13.x (VTK-based transforms)
   import fury
   from fury.transform import euler_matrix
   actor = fury.actor.sphere(centers)
   # Spatial transforms required manual VTK matrix composition:
   rot_matrix = euler_matrix(0.5, 0, 0)    # radians, returns 4x4 ndarray
   # No single-step convenience method existed on the actor object itself

   # FURY master (Native transforms on every actor)
   import fury
   actor = fury.actor.sphere(centers)
   # Every actor now inherits rotate/translate/scale/transform directly:
   actor.rotate((30, 0, 0))          # degrees, XYZ Euler (not radians!)
   actor.translate((1.0, 0.0, 0.0))  # world-space offset
   actor.scale(2.0)                  # uniform; or (2, 1, 0.5) per-axis
   actor.transform(matrix_4x4)       # raw 4×4 numpy matrix
   actor.opacity = 0.5               # native property

The ``fury.transform`` module functions ``rotate()``, ``translate()``,
and ``scale()`` remain available as free functions when an actor is not
involved or a transformation matrix is needed independently.

**New actor-group helpers** (in ``fury.actor.utils``):

.. code-block:: python

   fury.actor.set_opacity(actor, 0.5)
   fury.actor.set_group_opacity(group, 0.5)
   fury.actor.set_group_visibility(group, True)
   fury.actor.apply_affine_to_actor(actor, affine_4x4)
   fury.actor.apply_affine_to_group(group, affine_4x4)


4. Actor Module Restructuring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1014`)

The monolithic ``fury/actor.py`` has been refactored into a package with
the following sub-modules:

+------------------+-----------------------------------------------------------+
| Sub-module       | Contents                                                  |
+==================+===========================================================+
| ``core.py``      | ``Actor``, ``Mesh``, ``Line``, ``Points``, ``Text``,      |
|                  | ``Image``, ``Volume``, ``Group``, ``actor_from_primitive`` |
|                  | ``create_mesh``, ``create_line``, ``create_point``,       |
|                  | ``create_text``, ``create_image``, ``create_axes_helper`` |
|                  | ``line``, ``arrow``, ``axes``                             |
+------------------+-----------------------------------------------------------+
| ``curved.py``    | ``sphere``, ``ellipsoid``, ``cylinder``, ``cone``,        |
|                  | ``streamtube``, ``streamlines``                           |
+------------------+-----------------------------------------------------------+
| ``planar.py``    | ``disk``, ``image``, ``line_projection``, ``marker``,     |
|                  | ``point``, ``ring``, ``square``, ``star``, ``text``,      |
|                  | ``triangle``                                              |
+------------------+-----------------------------------------------------------+
| ``polyhedron.py``| ``box``, ``frustum``, ``icosahedron``, ``octagonalprism``,|
|                  | ``pentagonalprism``, ``rhombicuboctahedron``,             |
|                  | ``superquadric``, ``tetrahedron``, ``triangularprism``    |
+------------------+-----------------------------------------------------------+
| ``slicer.py``    | ``VectorField``, ``SphGlyph``, ``data_slicer``,           |
|                  | ``sph_glyph``, ``vector_field``, ``vector_field_slicer``  |
+------------------+-----------------------------------------------------------+
| ``topology.py``  | ``contour_from_volume``, ``surface``                      |
+------------------+-----------------------------------------------------------+
| ``bio.py``       | ``contour_from_roi``, ``contour_from_label``,             |
|                  | ``peaks_slicer``, ``volume_slicer``                       |
+------------------+-----------------------------------------------------------+
| ``_billboard.py``| ``Billboard``, ``billboard``, ``billboard_sphere``        |
+------------------+-----------------------------------------------------------+
| ``utils.py``     | ``set_opacity``, ``set_group_opacity``,                   |
|                  | ``set_group_visibility``, ``get_slices``, ``show_slices`` |
|                  | ``apply_affine_to_actor``, ``apply_affine_to_group``,     |
|                  | ``read_buffer``                                           |
+------------------+-----------------------------------------------------------+

**Action**: Always import from ``fury.actor`` (the public ``__init__``
facade). Avoid deep internal imports like
``from fury.actor.curved import sphere``.


5. ``actor_from_primitive`` API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`962`, :ghpull:`1125`)

``actor_from_primitive`` is the canonical way to create custom mesh actors.
Its signature is:

.. code-block:: python

   fury.actor.actor_from_primitive(
       vertices,
       faces,
       centers,
       *,
       colors=(1, 0, 0),
       scales=(1, 1, 1),
       directions=(1, 0, 0),
       opacity=None,
       material="phong",     # "phong" or "basic"
       smooth=False,
       enable_picking=True,
       repeat_primitive=True,
       have_tiled_verts=False,
       wireframe=False,
       wireframe_thickness=1.0,
   )

Bug fixed: a redundant ``local.position`` offset was being applied when
``len(centers) > 1``, causing actors to be double-positioned. This is now
corrected. (Ref: :ghissue:`1124`, :ghpull:`1125`)


6. New Actors in master
~~~~~~~~~~~~~~~~~~~~~~~~

The following actors are **new** in master and do not exist in 0.13.x:

+----------------------------+------------------------------------------+
| Actor                      | PR reference                             |
+============================+==========================================+
| ``actor.streamlines``      | :ghpull:`1021`                           |
+----------------------------+------------------------------------------+
| ``actor.streamtube``       | Shader-based rewrite :ghpull:`1038`      |
+----------------------------+------------------------------------------+
| ``actor.peaks_slicer``     | :ghpull:`1018`                           |
+----------------------------+------------------------------------------+
| ``actor.volume_slicer``    | :ghpull:`996`                            |
+----------------------------+------------------------------------------+
| ``actor.data_slicer``      | (slicer sub-module)                      |
+----------------------------+------------------------------------------+
| ``actor.vector_field``     | :ghpull:`992`, :ghpull:`995`             |
+----------------------------+------------------------------------------+
| ``actor.surface``          | :ghpull:`997`, :ghpull:`1040`            |
+----------------------------+------------------------------------------+
| ``actor.contour_from_roi`` | :ghpull:`1053`                           |
+----------------------------+------------------------------------------+
| ``actor.contour_from_volume``| :ghpull:`1053`                         |
+----------------------------+------------------------------------------+
| ``actor.sph_glyph``        | :ghpull:`1009`                           |
+----------------------------+------------------------------------------+
| ``actor.billboard_sphere`` | :ghpull:`1037`                           |
+----------------------------+------------------------------------------+
| ``actor.line_projection``  | :ghpull:`1029`                           |
+----------------------------+------------------------------------------+
| ``actor.read_buffer``      | GPU readback :ghpull:`1098`              |
+----------------------------+------------------------------------------+

The ``actor.image`` actor now supports directional parameters.
(Ref: :ghpull:`1001`)


7. Slicer API: ``get_slices`` / ``show_slices``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`996`, :ghpull:`1068`, :ghpull:`1075`)

Slicer actors (``volume_slicer``, ``data_slicer``) return a ``Group``
object. Two helper utilities act on these groups:

.. code-block:: python

   slicer_group = fury.actor.volume_slicer(data)

   positions = fury.actor.get_slices(slicer_group)   # returns ndarray
   fury.actor.show_slices(slicer_group, (x, y, z))   # moves slices

Blend-mode is now available on the slicer actor. (Ref: :ghpull:`1068`)
Slicer flickering in nearest interpolation mode is fixed.
(Ref: :ghpull:`1075`)


8. ``fury.window`` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`955`, :ghpull:`969`, :ghpull:`1047`,
:ghpull:`1054`, :ghpull:`1060`)

The ``window`` module has been rewritten around PyGfx canvas types.

**``Scene``** constructor signature (replaces old ``Scene``):

.. code-block:: python

   # FURY 0.13.x (VTK-based scene)
   import fury
   scene = fury.window.Scene()
   scene.background((0, 0, 0))    # method call (not a property)
   scene.add(actor)               # mapped to VTK AddActor() internally

   # FURY master (PyGfx-based scene)
   import fury
   scene = fury.window.Scene(
       background=(0, 0, 0, 1),   # RGBA float [0,1] — now in constructor
       skybox=None,               # pygfx Texture cubemap
       lights=None,               # list of gfx Light objects
   )

   scene.add(actor)
   scene.remove(actor)
   scene.background = (0.1, 0.1, 0.1, 1.0) # now a settable property
   scene.set_skybox(cube_map_texture)
   scene.clear()

**``ShowManager``** constructor (new / changed parameters):

.. code-block:: python

   show_m = fury.window.ShowManager(
       scene=scene,
       title="My App",
       size=(800, 600),
       window_type="default",   # "default","glfw","qt","jupyter","offscreen"
       pixel_ratio=1.25,
       camera_light=True,
       screen_config=None,      # NEW: multi-screen layout config
       enable_events=True,
       qt_app=None,             # NEW: existing QApplication instance
       qt_parent=None,          # NEW: existing QWidget parent
       show_fps=False,          # NEW: on-screen FPS overlay
       max_fps=60,              # NEW: cap render rate
       imgui=False,             # NEW: enable ImGui integration
       imgui_draw_function=None,# NEW: ImGui per-frame draw callback
   )

   show_m.start()              # blocking render loop
   show_m.render()             # non-blocking single-frame request
   show_m.snapshot(fname)      # save PNG of current frame
   show_m.close()

**Callback system** (Ref: :ghpull:`1047`):

.. code-block:: python

   # Actual signature: register_callback(func, time, repeat, name, *args)
   show_m.register_callback(my_func, 0.1, True, "my_cb")
   show_m.cancel_callback("my_cb")
   show_m.resize_callback(func)
   show_m.cancel_resize_callback()

**Multi-screen layout** via ``screen_config``:

.. code-block:: python

   # Two vertical columns: left has 1 row, right has 2 rows
   show_m = fury.window.ShowManager(screen_config=[1, 2])
   # Access individual screens:
   left_scene  = show_m.screens[0].scene
   right_top   = show_m.screens[1].scene
   right_bottom = show_m.screens[2].scene


9. Drag Events (``POINTER_DRAG``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1046`)

A new ``EventType.POINTER_DRAG`` is dispatched during pointer-move events
while a pointer button is held down. This enables per-object drag handling:

.. code-block:: python

   # FURY 0.13.x (VTK observers)
   import fury
   show_m = fury.window.ShowManager(scene)
   def vtk_on_drag(obj, event):
       print("Dragging...")
   # Bound to the entire interactor window, not specific objects:
   show_m.iren.AddObserver("MouseMoveEvent", vtk_on_drag)

   # FURY master (PyGfx event system)
   import fury
   from fury.lib import EventType

   def on_drag(event):
       print(event.x, event.y, event.target)

   # Bound directly to the target actor:
   actor.add_event_handler(on_drag, EventType.POINTER_DRAG)


10. ImGui Integration
~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1060`)

``ShowManager`` now supports ImGui for immediate-mode GUI panels rendered
on top of the 3D scene.

.. code-block:: python

   import imgui

   def draw_gui():
       imgui.begin("Debug")
       imgui.text("Hello from ImGui")
       imgui.end()

   show_m = fury.window.ShowManager(imgui=True,
                                    imgui_draw_function=draw_gui)
   show_m.start()

   # Or enable / change at runtime:
   show_m.enable_imgui(imgui_draw_function=draw_gui)
   show_m.set_imgui_render_callback(new_draw_func)
   show_m.disable_imgui()


11. Axes / Orientation Gizmo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1141`)

A navigable axes gizmo can be overlaid on any screen. Clicking an axis
disk re-aligns the camera to that axis.

.. code-block:: python

   show_m.show_axes_gizmo(
       screen=0,
       size=30,
       thickness=2,
       position=None,               # defaults to bottom-left (60, 60)
       labels=["-X","+X","-Y","+Y","-Z","+Z"],
       click_callback=my_callback,  # receives axis direction ndarray
   )


12. UI Sub-system
~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`998`, :ghpull:`999`, :ghpull:`1043`,
:ghpull:`1052`, :ghpull:`1056`, :ghpull:`1118`)

The UI system has been rebuilt from scratch on PyGfx. Many 0.13.x
components are temporarily disabled (``ComboBox2D``, ``ListBox2D``,
``FileMenu2D``, ``DrawPanel``, etc.) while the migration is in progress.
Legacy backward-compatibility code was explicitly removed.
(Ref: :ghpull:`1043`)

**Currently available UI components**:

* ``fury.ui.Rectangle2D``
* ``fury.ui.Disk2D``
* ``fury.ui.TextBlock2D``       (Ref: :ghpull:`1052`)
* ``fury.ui.Panel2D``           (Ref: :ghpull:`999`)
* ``fury.ui.TexturedButton2D``  (Ref: :ghpull:`1056`)
* ``fury.ui.TextButton2D``      (Ref: :ghpull:`1056`)
* ``fury.ui.LineSlider2D``      (Ref: :ghpull:`1118`)
* ``fury.ui.RingSlider2D``
* ``fury.ui.PlaybackPanel``
* ``fury.ui.ImageContainer2D``
* ``fury.ui.Panel2D``, ``fury.ui.TabPanel2D``, ``fury.ui.TabUI``

**Action**: If you relied on ``ComboBox2D``, ``ListBox2D``, ``DrawPanel``,
``TextBox2D``, or ``RangeSlider`` in 0.13.x, these are not yet available
in master. Consider using ImGui panels as a temporary replacement.


13. FPS Display and Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1054`)

Frame-rate control is now first-class:

.. code-block:: python

   show_m = fury.window.ShowManager(show_fps=True, max_fps=60)
   current_fps = show_m.get_fps()


14. GPU Buffer Readback
~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1098`)

Read GPU buffer contents back to a NumPy array:

.. code-block:: python

   data = fury.actor.read_buffer(buffer, sync_cpu=True)
   # Returns np.ndarray (float32) matching the buffer's shape.


15. ``fury.io.save_image`` TIFF Compression Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1163`, :ghissue:`1162`)

In 0.13.x, passing ``compression_type`` to ``save_image()`` for TIFF
files was silently ignored. This is now corrected:

.. code-block:: python

   fury.io.save_image(array, "output.tiff", compression_type="tiff_lzw")


16. ``fury.colormap`` Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1103`)

* ``boys2rgb``: a long-standing bug in the ``z4`` term was corrected
  (``z4 = z2*z2``, not ``z*z2``). (Ref: :ghissue:`857`)
* ``colormap_lookup_table`` (VTK-based) has been **removed** — it is
  commented out in the source.
* ``normalize_colors`` is now a public API and handles hex strings,
  int/float RGB/RGBA, and tiling automatically.


17. ``fury.window.snapshot`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Summary** (Ref: :ghpull:`1050`)

The free function ``fury.window.snapshot`` wraps an offscreen
``ShowManager`` and saves a PNG. Its signature:

.. code-block:: python

   array = fury.window.snapshot(
       scene=scene,            # or actors=list_of_actors
       actors=None,
       screen_config=None,
       fname="output.png",
       return_array=False,
   )


18. Removed / Commented-Out APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following modules and symbols exist in 0.13.x but are **not** in the
master public API (either removed or temporarily commented out pending
v2.0.0 stabilization):

+-------------------------------+--------------------------------------------------+
| Removed / commented out       | Notes                                            |
+===============================+==================================================+
| ``fury.gltf``                 | Optional; pending PyGfx port                     |
+-------------------------------+--------------------------------------------------+
| ``fury.interactor``           | VTK interactor; removed with VTK                 |
+-------------------------------+--------------------------------------------------+
| ``fury.layout``               | Grid/horizontal/vertical layouts pending port    |
+-------------------------------+--------------------------------------------------+
| ``fury.molecular``            | Pending port                                     |
+-------------------------------+--------------------------------------------------+
| ``fury.pick``                 | VTK pick manager removed                         |
+-------------------------------+--------------------------------------------------+
| ``fury.stream``               | Streaming module pending port                    |
+-------------------------------+--------------------------------------------------+
| ``fury.colormap.colormap_lookup_table`` | VTK LUT removed                     |
+-------------------------------+--------------------------------------------------+
| ``fury.io.load_polydata``     | VTK polydata I/O removed                         |
+-------------------------------+--------------------------------------------------+
| ``fury.io.save_polydata``     | VTK polydata I/O removed                         |
+-------------------------------+--------------------------------------------------+
| ``fury.ui.ComboBox2D``        | Pending UI port                                  |
+-------------------------------+--------------------------------------------------+
| ``fury.ui.ListBox2D``         | Pending UI port                                  |
+-------------------------------+--------------------------------------------------+
| ``fury.ui.DrawPanel``         | Pending UI port                                  |
+-------------------------------+--------------------------------------------------+
| ``fury.ui.FileMenu2D``        | Pending UI port                                  |
+-------------------------------+--------------------------------------------------+
| ``fury.ui.TextBox2D``         | Pending UI port                                  |
+-------------------------------+--------------------------------------------------+
| ``fury.ui.RangeSlider``       | Pending UI port                                  |
+-------------------------------+--------------------------------------------------+


Dependency Changes
------------------

+---------------------+--------------------+
| Dependency          | Change             |
+=====================+====================+
| ``vtk``             | Removed            |
+---------------------+--------------------+
| ``pygfx``           | Added (≥ 0.13.0)  |
+---------------------+--------------------+
| ``wgpu``            | Added              |
+---------------------+--------------------+
| ``jupyter_rfb``     | Added (Jupyter)    |
+---------------------+--------------------+
| ``imgui``           | Optional           |
+---------------------+--------------------+
| ``numba``           | Optional           |
+---------------------+--------------------+
