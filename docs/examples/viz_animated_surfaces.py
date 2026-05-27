"""
=====================
Animated 2D Functions
=====================

Demonstrates the real-time evaluation and animation of 2D mathematical surfaces
"""

#########################################################################################
# Import the required libraries and define the grid resolution parameters.
import numpy as np
from fury import actor, colormap, ui, window

N_POINTS = 128
SPACING = 3.0

#########################################################################################
# Generate a flat coordinate grid mapping and initialize the global animation state.
x_vals = np.linspace(-1.0, 1.0, N_POINTS)
y_vals = np.linspace(-1.0, 1.0, N_POINTS)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
x_flat = x_grid.ravel()
y_flat = y_grid.ravel()

state = {"time": 0.0, "dt": 0.05}

#########################################################################################
# Define the mathematical equations and assign corresponding colormaps.
eq1 = "np.abs(np.sin(x*2*np.pi*np.cos(time/2)))**1*np.cos(time/2)\
    *np.abs(np.cos(y*2*np.pi*np.sin(time/2)))**1*np.sin(time/2)*1.2"
eq2 = "((x**2 - y**2)/(x**2 + y**2 + 1e-5))**(2)*np.cos(6*np.pi*x*y-1.8*time)*0.24"
eq3 = "(np.sin(np.pi*2*x-np.sin(1.8*time))*np.cos(np.pi*2*y+np.cos(1.8*time)))*0.48"
eq4 = "np.cos(24*np.sqrt(x**2 + y**2) - 2*time)*0.18"

equations = [eq1, eq2, eq3, eq4]
cmap_names = ["hot", "plasma", "viridis", "ocean"]


#########################################################################################
# Define helper functions to construct grid faces and evaluate dynamic equations.
def generate_grid_faces(n):
    """Generate continuous index faces for 3D surface mesh generation."""
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            p0 = i * n + j
            p1 = i * n + (j + 1)
            p2 = (i + 1) * n + j
            p3 = (i + 1) * n + (j + 1)
            faces.append([p0, p1, p2])
            faces.append([p1, p3, p2])
    return np.array(faces, dtype=np.int32)


def evaluate_equation(equation, x, y, t):
    """Evaluate equations mathematically within an isolated NumPy namespace."""
    return eval(equation, {"np": np, "x": x, "y": y, "t": t, "time": t})


#########################################################################################
# Define the geometry update routine to modify active vertex and color buffers directly.
def update_surface_geometry(surf, x, y, t):
    """Update existing GPU buffers directly to maintain rendering performance."""
    z = evaluate_equation(surf.equation, x, y, t)
    xyz = np.vstack([x, y, z]).T

    v = np.copy(z)
    max_val = np.max(np.abs(v))
    v /= max_val if max_val > 0 else 1.0
    colors = np.asarray(colormap.create_colormap(v, name=surf.cmap_name))

    if colors.ndim == 2 and colors.shape[1] == 3:
        colors = np.hstack([colors, np.ones((len(colors), 1), dtype=colors.dtype)])

    surf.geometry.positions.data[:, :] = xyz.astype(np.float32)
    surf.geometry.positions.update_full()

    surf.geometry.colors.data[:, :] = colors.astype(np.float32)
    surf.geometry.colors.update_full()


#########################################################################################
# Initialize the 3D scene viewport and spawn horizontal surface actors.
scene = window.Scene()
scene.background = (0.1, 0.1, 0.15)
scene.add(actor.axes())

faces = generate_grid_faces(N_POINTS)
surfaces = []

for i in range(4):
    xyz_init = np.vstack([x_flat, y_flat, np.zeros_like(x_flat)]).T
    colors_init = np.ones((len(x_flat), 4), dtype=np.float32)

    surf = actor.surface(xyz_init, faces=faces, colors=colors_init)
    surf.equation = equations[i]
    surf.cmap_name = cmap_names[i]

    surf.local.position = (i * SPACING - 1.5 * SPACING, 0.0, 0.0)

    scene.add(surf)
    surfaces.append(surf)

#########################################################################################
# Add a 2D text overlay to display the legend.
hud_legend = ui.TextBlock2D(
    text="Fury Animated 2D Functions\n"
    "F1: Standing Wave | F2: Hyperbolic | F3: Wave Packet | F4: Concentric Ripples",
    position=(30, 30),
    font_size=16,
    color=(0.9, 0.9, 0.95),
    bold=True,
    dynamic_bbox=True,
)
scene.add(hud_legend)

#########################################################################################
# Initialize the ShowManager, position the virtual camera, and register the update loop.
if __name__ == "__main__":
    show_m = window.ShowManager(
        scene=scene, size=(1024, 768), title="Fury Mathematical Functions"
    )

    camera = show_m.screens[0].camera
    camera.local.position = (0.0, -8.0, 5.0)
    camera.look_at((0.0, 0.0, 0.0))

    def update_playback_frame():
        state["time"] += state["dt"]
        for surf in surfaces:
            update_surface_geometry(surf, x_flat, y_flat, state["time"])

    show_m.register_callback(update_playback_frame, 0.03, True, "FunctionsLoop")

    show_m.start()
