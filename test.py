from fury import ui, window
from fury.io import load_sprite_sheet

TARGET_FPS = 20
FRAME_TIME = (1.0 / TARGET_FPS) * 1000
CURRENT_SPRITE_IDX = 0

sprite_url = 'https://raw.githubusercontent.com/'\
             'antrikshmisri/DATA/master/fury/0yKFTBQ.png'

sprite_sheet = load_sprite_sheet(sprite_url, 5, 5, as_vtktype=True)
img_container = ui.ImageContainer2D(img_path=sprite_url, position=(100, 100), size=(200, 200))


def timer_callback(_obj, _evt):
    global CURRENT_SPRITE_IDX, show_manager
    CURRENT_SPRITE_IDX += 1
    sprite = list(sprite_sheet.values())[CURRENT_SPRITE_IDX % len(sprite_sheet)]
    img_container.set_img(sprite)
    i_ren = show_manager.scene.GetRenderWindow()\
        .GetInteractor().GetInteractorStyle()

    i_ren.force_render()

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Sprite Sheet")

show_manager.scene.add(img_container)
show_manager.scene.background((0.2, 0.2, 0.2))
show_manager.initialize()

show_manager.add_timer_callback(True, int(FRAME_TIME), timer_callback)

# To interact with the UI, set interactive = True
interactive = True

if interactive:
    show_manager.start()