from fury import ui, window

scroll_1 = ui.ScrollBar(100, 20, 0.5, position=(100, 100))
scroll_2 = ui.ScrollBar(100, 20, 0.5, position=(100, 70), orientation="horizontal")


sm = window.ShowManager(size=(500, 500))
sm.scene.add(scroll_1, scroll_2)
sm.start()