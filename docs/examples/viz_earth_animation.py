"""
Earth Texture Example – FURY v2

This example demonstrates:

1. Fetching dataset textures using FURY's data helpers.
2. Using the new `actor.texture_on_sphere` API introduced in v2.
3. Creating a textured sphere using equirectangular UV mapping.
4. Rendering the result using the WebGPU-based ShowManager.

Why is this example static?
---------------------------
In the master branch, this tutorial includes animation and camera rotation.
However, FURY v2 is still transitioning into a WebGPU backend. Several APIs
that existed in v1/vtk-based FURY — such as:

- scene.set_camera()
- scene.azimuth()
- scene.zoom()
- timer callbacks
- mesh rotation/transforms

are not yet implemented in v2.

To keep the tutorial stable, clean, and fully compatible with the current v2
API, this example focuses on the core concept: mapping a texture onto a sphere.
Once v2 introduces animation and transform APIs, this tutorial can be extended.
"""

import fury

# -----------------------------------------------------------------------------
# 1. Create a Scene
# -----------------------------------------------------------------------------
# The Scene is the main container that holds all 3D actors.
scene = fury.window.Scene()


# -----------------------------------------------------------------------------
# 2. Fetch required data (Earth texture)
# -----------------------------------------------------------------------------
# FURY stores example datasets in a cache. `fetch_viz_textures()` ensures that
# the required texture files exist on the user's system.
fury.data.fetch_viz_textures()

# Load the *file path* of the 8K Earth texture.
# IMPORTANT: In v2 the texture_on_sphere API expects a FILE PATH, not a numpy array.
earth_texture_path = fury.data.read_viz_textures("1_earth_8k.jpg")


# -----------------------------------------------------------------------------
# 3. Create the Earth actor using the new `texture_on_sphere` API
# -----------------------------------------------------------------------------
# `texture_on_sphere` is implemented in `actor/curved.py` in this PR.
# It internally:
# - builds a UV-mapped sphere using primitive.prim_sphere()
# - computes equirectangular texture coordinates
# - passes everything to the v2 `surface()` actor
earth_actor = fury.actor.texture_on_sphere(
    texture=earth_texture_path,
    center=(0.0, 0.0, 0.0),   # Position of the sphere
    radius=1.0,               # Sphere radius
)

# Add the sphere to the scene so it becomes visible.
scene.add(earth_actor)


# -----------------------------------------------------------------------------
# 4. Display the result using a ShowManager
# -----------------------------------------------------------------------------
# In FURY v2, ShowManager creates a WebGPU-powered rendering window.
# Note:
# - Animation callbacks from v1 are not available yet.
# - Camera manipulation helpers are also not exposed in v2 yet.
showm = fury.window.ShowManager(
    scene=scene,
    size=(900, 768),
)


# -----------------------------------------------------------------------------
# 5. Start the rendering window
# -----------------------------------------------------------------------------
# Running this file opens a window showing a textured Earth sphere.
# This serves as a minimal but fully functional v2 tutorial.
if __name__ == "__main__":
    showm.start()
