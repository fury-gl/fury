from numpy.random import random
from fury import text_tools
from fury import actor, window
import fury


def test_create_atlas():
    if not text_tools._FREETYPE_AVAILABLE:
        print('Bitmap text not tested (FREETYPE is not available)')
        return

    name = 'test_fonts_1238210930'
    font_path_atlas = text_tools._FONT_PATH_USER
    font_path = f'{fury.__path__[0]}/data/files/FreeMono.ttf'
    text_tools.create_new_font(
        name=name,
        font_size_res=10, font_path=font_path, show=False)
    fonts = text_tools.list_fonts_available(True)
    shutil.rmtree(f'{font_path_atlas}/{name}')
    if name not in fonts.keys():
        raise FileNotFoundError(f'Font {name} was not created')


def test_atlas():
    if not text_tools._FREETYPE_AVAILABLE:
        print('Bitmap text not tested (FREETYPE is not available)')
        return

    texture_atlas = text_tools.TextureAtlas(num_chanels=1)
    font_path = f'{fury.__path__[0]}/data/files/FreeMono.ttf'
    tf = text_tools.TextureFont(texture_atlas, font_path, 100)
    tf.load('a sdf123')

def test_text_bitmap_actor():
    interactive = False
    char2pos = text_tools.get_ascii_chars()[1]
    chars = list(char2pos.keys())
    N = 10 
    colors = (0, 0.8, .5)
    scales = 1
    labels = ['Abracadabra 1664123!@']
    centers = np.random.normal(0, 10, size=(N, 3))
    if N > 1:
        # colors = np.random.uniform(0, 1, size=(N, 3))
        min_s = 5
        max_s = 10
        chars = [chr(i) for i in range(32, 134)]
        labels = [
            f'Sphere {i}!@'
            # ''.join(
            #         np.random.choice(
            #             chars,
            #             size=np.random.randint(min_s, max_s)
            #         )
            #     )
            for i in range(N)
        ]
        colors = []
        colors_spheres = []
        for label in labels:
            c = np.random.uniform(0, 1, size=3)
            colors_spheres.append(c)
            for _ in label:
                colors.append(c)
    spheres = actor.sphere(centers, colors)
    my_text_actor = actor_text.bitmap_labels(
        centers, labels, colors=colors, scales=scales)
    showm = window.ShowManager()
    showm.scene.add(my_text_actor)
    showm.scene.add(spheres)
    if interactive:
        showm.initialize()
        showm.start()

