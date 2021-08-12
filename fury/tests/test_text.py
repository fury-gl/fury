import numpy as np
from fury import text_tools
from fury import actor, window
import fury


def test_create_bitmap():
    save_path = f'{fury.__path__[0]}/data/files/FreeMono'
    font_path = f'{fury.__path__[0]}/data/files/FreeMono.ttf'
    text_tools.create_bitmap_font(
        50, show=False, font_path=font_path, save_path=save_path)


def test_atlas():
    texture_atlas = text_tools.TextureAtlas(depth=1)
    font_path = f'{fury.__path__[0]}/data/files/FreeMono.ttf'
    tf = text_tools.TextureFont(texture_atlas, font_path, 100)
    tf.load('a sdf123') 


def test_get_positions_text_billboards():
    img_arr, char2pos = text_tools.create_bitmap_font(
        12, show=False)
    chars = list(char2pos.keys())
    N = 5
    centers = np.random.normal(0, 1, size=(N, 3))
    min_s = 5
    max_s = 10
    labels = [
        ''.join(
                np.random.choice(
                    chars,
                    size=np.random.randint(min_s, max_s)
                )
            )
        for i in range(N)
    ]
    num_labels = len(''.join(labels))
    padding, labels_positions, uv = text_tools.get_positions_labels_billboards(
        labels, centers, char2pos)
    assert padding.shape[0] == num_labels
    assert labels_positions.shape[0] == num_labels
    assert uv.shape[0] == num_labels*4
    assert padding.shape[1] == 3
    assert labels_positions.shape[1] == 3
    assert uv.shape[1] == 2


def test_bitmap_actor():
    interactive = False
    N = 10
    colors = (0, 0.8, .5)
    colors_spheres = colors
    scales = 1
    labels = ['Abracadabra 1664123!@/?*)(']
    centers = np.random.normal(0, 10, size=(N, 3))
    if N > 1:
        # colors = np.random.uniform(0, 1, size=(N, 3))
        # min_s = 5
        # max_s = 10
        labels = [
            f'Sphere{i}' if i % 2 == 0 else f'Sphere {i}'
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
    spheres = actor.markers(centers, colors_spheres)
    my_text_actor = actor.bitmap_labels(
        centers, labels, colors=colors, scales=scales,
        align='center', font_size=51)
    showm = window.ShowManager(size=(500, 400))
    showm.scene.add(my_text_actor)
    showm.scene.add(spheres)
    if interactive:
        showm.initialize()
        showm.start()

