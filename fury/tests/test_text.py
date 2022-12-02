import numpy as np
import shutil
from fury import text_tools
from fury import actor, window
import fury


def test_create_atlas():
    if not text_tools.FREETYPE_AVAILABLE:
        print('Bitmap text not tested (FREETYPE is not available)')
        return

    name = 'test_fonts_1238210930'
    font_path_atlas = text_tools.FONT_PATH_USER
    font_path = f'{fury.__path__[0]}/data/files/FreeMono.ttf'
    text_tools.create_new_font(
        name=name,
        font_size_res=10, font_path=font_path, show=False)
    fonts = text_tools.list_fonts_available(True)
    shutil.rmtree(f'{font_path_atlas}/{name}')
    if name not in fonts.keys():
        raise FileNotFoundError(f'Font {name} was not created')


def test_bitmap_actor():
    N = 10
    colors = (0, 0.8, .5)
    colors_spheres = colors

    labels = ['AbrBac..ooo0123_///::adabra_ 1664123!@/?*...)(']
    centers = np.random.normal(0, 10, size=(N, 3))
    if N > 1:
        min_s = 5
        max_s = 10
        chars = [chr(i) for i in range(32, 134)]
        labels = [
            ''.join(
                    np.random.choice(
                        chars,
                        size=np.random.randint(min_s, max_s)
                    )
                )
            for i in range(N)
        ]
        colors = []
        colors_spheres = []
        for label in labels:
            c = np.random.uniform(0, 1, size=3)
            colors_spheres.append(c)
            for _ in label:
                colors.append(c)
    spheres = actor.markers(centers, colors_spheres, scales=.1)
    my_text_actor = actor.label_fast(
        centers, labels, colors=colors, scales=.1,
        y_offset_ratio=1,
        align='center', font_name='FreeMono')
    showm = window.ShowManager(size=(700, 200))
    showm.scene.add(my_text_actor)
    showm.scene.add(spheres)
