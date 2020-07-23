from fury import ui, window
tab = ui.TabUI(nb_tabs=4, size=(300, 300), position=(50, 50))
tab.tabs[0].content_panel.add_element(ui.Rectangle2D(), (0.5, 0.5))
sm = window.ShowManager(size=(500, 500))
# sm.scene.background((1, 0, 0))
sm.scene.add(tab)
sm.start()