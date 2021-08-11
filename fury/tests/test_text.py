from numpy.random import random
from fury import text_tools
from fury import actor, window
from fury import actor_text
import numpy as np


def test_create_bitmap():
    img_arr, char2pos = text_tools.create_bitmap_font(
        #12, show=False,)
        50, show=False, save_path='/home/devmessias/phd/fury/fury/data/files/font.png')


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

