from fury.layout import VerticalLayout
from fury.tests.test_layout import get_default_cubes
from fury import window

cube_first, cube_second = get_default_cubes()
cube_third, cube_fourth = get_default_cubes()

grid = VerticalLayout(cell_shape='rect')

grid.apply([cube_first, cube_second, cube_third, cube_fourth])

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Bullet List Example")

show_manager.scene.add(cube_first, cube_second, cube_third, cube_fourth)

# To interact with the UI, set interactive = True
interactive = True

if interactive:
    show_manager.start()