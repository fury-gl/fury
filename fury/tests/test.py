from fury import ui, window
tab = ui.TabUI(nb_tabs=4, size=(300, 300), position=(50, 50))
sm = window.ShowManager(size=(500, 500))
sm.scene.add(tab)
sm.start()