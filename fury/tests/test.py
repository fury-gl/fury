from fury import ui, window
tab = ui.TabUI(nb_tabs=4, size=(300, 300), position=(50, 50), draggable=True)
tab.tabs[0].title = "Red"
tab.tabs[1].title = "Green"
tab.tabs[2].title = "Blue"
tab.tabs[3].title = "White"
tab.tabs[0].add_element(ui.Rectangle2D(size=(100, 100), color=(1, 0, 0)), (0.5, 0.5))
tab.tabs[1].add_element(ui.Rectangle2D(size=(10, 10), color=(0, 1, 0)), (0.5, 0.5))
tab.tabs[2].add_element(ui.Rectangle2D(size=(80, 100), color=(0, 0, 1)), (0.5, 0.5))
tab.tabs[3].add_element(ui.Rectangle2D(size=(100, 20), color=(1, 1, 1)), (0.5, 0.5))
sm = window.ShowManager(size=(500, 500))
# sm.scene.background((1, 0, 0))
sm.scene.add(tab)
sm.start()