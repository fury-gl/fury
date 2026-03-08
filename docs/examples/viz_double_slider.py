"""
=====================================
LineDoubleSlider2D Demo
=====================================

This example demonstrates the ``LineDoubleSlider2D`` UI component,
which lets users select a range by dragging two handles on a track.
"""

from fury import ui, window


# Create the scene
scene = window.Scene()
scene.background = (0.1, 0.1, 0.15)

# --- Horizontal double slider (disk handles) ---
horizontal_slider = ui.LineDoubleSlider2D(
    position=(100, 400),
    initial_values=(20, 80),
    min_value=0,
    max_value=100,
    length=300,
    orientation="horizontal",
    shape="disk",
    outer_radius=12,
    text_template="{value:.0f}",
)


def on_horizontal_change(slider):
    print(
        f"Horizontal range: {slider.left_disk_value:.1f} - "
        f"{slider.right_disk_value:.1f}"
    )


horizontal_slider.on_change = on_horizontal_change
scene.add(horizontal_slider)

# --- Vertical double slider (square handles) ---
vertical_slider = ui.LineDoubleSlider2D(
    position=(500, 100),
    initial_values=(10, 90),
    min_value=0,
    max_value=100,
    length=300,
    orientation="vertical",
    shape="square",
    handle_side=15,
    text_template="{value:.0f}",
)


def on_vertical_change(slider):
    print(
        f"Vertical range: {slider.bottom_disk_value:.1f} - "
        f"{slider.top_disk_value:.1f}"
    )


vertical_slider.on_change = on_vertical_change
scene.add(vertical_slider)

# --- Label ---
label = ui.TextBlock2D(
    text="LineDoubleSlider2D Demo",
    font_size=22,
    color=(1, 1, 1),
    dynamic_bbox=True,
)
label.set_position((150, 50))
scene.add(label)

# Show the scene
window.show(scene, title="FURY LineDoubleSlider2D Example", size=(700, 500))
