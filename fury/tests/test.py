from fury import ui, window
tab = ui.TabUI(nb_tabs=1)
sm = window.ShowManager()
sm.scene.add(tab)
sm.start()