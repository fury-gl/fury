from fury import ui, window

scene = window.Scene()

slider = ui.RangeSlider(
    position=(300, 300),
    min_value=0,
    max_value=100
)

scene.add(slider)

show_manager = window.ShowManager(scene=scene)
show_manager.start()
