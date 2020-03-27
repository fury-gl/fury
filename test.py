from fury import ui, window





values = ['Rectangle', 'Disks', 'Image', "Button Panel",
          "Line and Ring Slider", "Range Slider"]

checkbox_list = ui.RadioButton(values, padding=1,checked_labels=['Rectangle'], font_size=18, font_family='Arial', position=(0, 0))


current_size = (800, 800)
show_manager = window.ShowManager(size=current_size, title="DIPY UI Example")

show_manager.scene.add(checkbox_list)


interactive = True

if interactive:
    show_manager.start()

window.record(show_manager.scene, size=current_size, out_path="viz_ui.png")
