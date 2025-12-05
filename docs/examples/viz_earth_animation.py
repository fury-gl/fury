"""Earth texture example for FURY v2.

This example demonstrates:

- Fetching texture data from the FURY dataset.
- Using the new ``actor.texture_on_sphere`` API implemented in v2.
- Rendering a textured sphere without relying on deprecated v1 animation APIs.

A minimal static example is used because ShowManager and animation callbacks
are being updated for the v2 WebGPU backend.
"""

import fury

# -----------------------------------------------------------------------------
# 1. Scene and data
# -----------------------------------------------------------------------------

scene = fury.window.Scene()

# Fetch textures (download if missing)
fury.data.fetch_viz_textures()

# Path to Earth texture file
earth_file = fury.data.read_viz_textures("1_earth_8k.jpg")

# -----------------------------------------------------------------------------
# 2. Create Earth sphere using v2 texture API
# -----------------------------------------------------------------------------

earth_actor = fury.actor.texture_on_sphere(
    earth_file,
    center=(0.0, 0.0, 0.0),
    radius=1.0,
)

scene.add(earth_actor)

# -----------------------------------------------------------------------------
# 3. Show window (v2 minimal form)
# -----------------------------------------------------------------------------

showm = fury.window.ShowManager(
    scene=scene,
    size=(900, 768),
)

if __name__ == "__main__":
    showm.start()
