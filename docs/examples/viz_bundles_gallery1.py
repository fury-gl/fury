import matplotlib.pyplot as plt
import numpy as np
import os

from dipy.io.streamline import load_tractogram
from dipy.data.fetcher import get_bundle_atlas_hcp842
from dipy.stats.analysis import assignment_map
from fury import actor, ui, window
from fury.colormap import line_colors
from fury.lib import numpy_support
from fury.utils import numpy_to_vtk_colors
from io import BytesIO
from PIL import Image


def lmm_plot(p_vals, bundle_label, title, p_vals_thrs=.05, thrs_colors='red',
             bar_color=(0, 1, 0), lw=.6, ms=.7, ylim=(0, 4)):
    num_p_vals = len(p_vals)
    x = np.arange(num_p_vals)
    fig, ax = plt.subplots()
    if isinstance(p_vals_thrs, list) and isinstance(thrs_colors, list):
        legend_handles = []
        for thr, color in zip(p_vals_thrs, thrs_colors):
            dotted = np.ones(num_p_vals) * -np.log10(thr)
            legend, = ax.plot(x, dotted, color=color, marker='.',
                              linestyle='solid', linewidth=lw, markersize=ms,
                              label='p-value < {}'.format(thr))
            legend_handles.append(legend)
        thr_leg = ax.legend(handles=legend_handles, loc='upper right')
    else:
        dotted = np.ones(num_p_vals) * -np.log10(p_vals_thrs)
        legend, = ax.plot(x, dotted, color=thrs_colors, marker='.',
                          linestyle='solid', linewidth=lw, markersize=ms,
                          label='p-value < {}'.format(p_vals_thrs))
        thr_leg = ax.legend(handles=[legend], loc='upper right')
    ax.add_artist(thr_leg)
    log_p_vals = -np.log10(p_vals)
    legend = ax.bar(x, log_p_vals, color=bar_color, label=bundle_label)
    #ax.legend(handles=[legend], loc='upper left')
    ax.set_ylim(ylim)
    ax.set_ylabel('-log10(p-value)')
    ax.set_xlabel('Segment Number')
    ax.set_title(title)
    return fig, ax


if __name__ == '__main__':
    atlas, bundles = get_bundle_atlas_hcp842()
    bundles_dir = os.path.dirname(bundles)
    stats_dir = '/run/media/guaje/Data/Downloads/buan_flow/lmm_plots'

    tractograms = ['AF_L.trk', 'AF_R.trk', 'CST_L.trk', 'CST_R.trk']
    stats = ['AF_L_fa_pvalues.npy', 'AF_R_fa_pvalues.npy',
             'CST_L_fa_pvalues.npy', 'CST_R_fa_pvalues.npy']
    buan_highlights = [(1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)]
    num_cols = len(tractograms)

    scene = window.Scene()

    buan_thr = .05

    list_actors = [None] * num_cols * 2
    list_labels = [''] * num_cols * 2
    for i, (stat, tract) in enumerate(zip(stats, tractograms)):
        # Get label from tract name
        label = os.path.splitext(tract)[0]
        list_labels[num_cols + i] = label

        # Load stats
        stat_file = os.path.join(stats_dir, stat)
        p_values = np.load(stat_file)

        # Load tractogram
        tract_file = os.path.join(bundles_dir, tract)
        sft = load_tractogram(tract_file, 'same', bbox_valid_check=False)
        bundle = sft.streamlines

        data_length = len(p_values)

        indx = assignment_map(bundle, bundle, data_length)
        ind = np.array(indx)

        num_lines = len(bundle)
        lines_range = range(num_lines)
        points_per_line = [len(bundle[i]) for i in lines_range]
        points_per_line = np.array(points_per_line, np.intp)

        cols_arr = line_colors(bundle)
        colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
        vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
        colors = numpy_support.vtk_to_numpy(vtk_colors)
        colors = (colors - np.min(colors)) / np.ptp(colors)
        mean_color = np.mean(colors, axis=0)

        for j in range(data_length):
            if p_values[j] < buan_thr:
                colors[ind == j] = buan_highlights[i]

        """
        with plt.rc_context({'axes.facecolor': 'black',
                             'axes.labelcolor': 'white', 'text.color': 'white',
                             'xtick.color': 'white', 'ytick.color': 'white'}):
        """
        with plt.style.context('dark_background'):
            fig, ax = lmm_plot(p_values, label, 'FA', p_vals_thrs=[.05, .01],
                               thrs_colors=[buan_highlights[i], 'white'],
                               bar_color=mean_color)
            #fig.show()

        buffer = BytesIO()
        fig.savefig(buffer, dpi='figure', bbox_inches='tight',
                    transparent=True)
        buffer.seek(0)
        img = Image.open(buffer)
        img_arr = np.asarray(img)
        buffer.close()

        list_actors[i] = actor.texture(img_arr)
        list_actors[num_cols + i] = actor.line(bundle, colors=colors)

    """
    stat_file = os.path.join(stats_dir, stats[0])
    p_values = np.load(stat_file)
    fig, ax = lmm_plot(p_values, list_labels[0], 'FA', p_vals_thrs=[.05, .01],
                       thrs_colors=['red', 'black'])
    #fig.show()

    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img = Image.open(buffer)
    img_arr = np.asarray(img)
    buffer.close()

    canvas_actor = actor.texture(img_arr)
    scene.add(canvas_actor)
    window.show(scene)
    """

    grid = ui.GridUI(list_actors, captions=list_labels, cell_padding=10,
                     #cell_shape='square', aspect_ratio=4/3,
                     dim=(2, num_cols), rotation_speed=5, rotation_axis=None)

    show_m = window.ShowManager(scene=scene, size=(1920, 1080),
                                reset_camera=False, order_transparent=True)
    show_m.initialize()

    show_m.scene.add(grid)

    show_m.start()
