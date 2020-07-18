from fury import utils, ui, window

text = ui.TextBlock2D(text="Hello wassup", position=(50, 50), color=(1, 0, 0))
rectangle = ui.Rectangle2D(color=(1, 1, 1), position=(50, 50), size=(100, 50))

sm = window.ShowManager()
sm.scene.add(rectangle, text)
text.message = utils.clip_overflow(text, rectangle.size[0])
print(text.size, rectangle.size)
sm.start()