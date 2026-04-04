"""Demo script showcasing the RingSlider2D UI element."""

from fury import ui, window


def main():
    """Create a window with a RingSlider2D widget and start the event loop."""

    scene = window.Scene()

    # Place the slider roughly at the center of an 800x800 window
    ring = ui.RingSlider2D(center=(400, 400), initial_value=90)
    scene.add(ring)

    showm = window.ShowManager(scene=scene, size=(800, 800))
    showm.start()


if __name__ == "__main__":
    main()
