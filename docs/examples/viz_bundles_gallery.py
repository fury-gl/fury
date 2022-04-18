import numpy as np
import os

from dipy.io.streamline import load_tractogram
from dipy.data.fetcher import get_bundle_atlas_hcp842
from dipy.stats.analysis import assignment_map
from fury import actor, ui, window
from fury.colormap import line_colors
from fury.lib import numpy_support
from fury.utils import numpy_to_vtk_colors


if __name__ == '__main__':
    atlas, bundles = get_bundle_atlas_hcp842()
    bundles_dir = os.path.dirname(bundles)

    tractograms = ['AF_L.trk', 'AF_R.trk', 'CST_L.trk', 'CST_R.trk']
    #stats = []
    #buan_highlights = [(1, 0, 0), (1, 1, 0), (1, 0, 0)]

    scene = window.Scene()

    #buan_thr = .05

    list_actors = []
    list_labels = []
    for i, tract in enumerate(tractograms):
        tract_file = os.path.join(bundles_dir, tract)
        list_labels.append(os.path.splitext(tract)[0])
        sft = load_tractogram(tract_file, 'same', bbox_valid_check=False)
        bundle = sft.streamlines

        """
        n = len(stat)
        p_values = stat

        indx = assignment_map(bundle, bundle, n)
        ind = np.array(indx)
        """

        num_lines = len(bundle)
        lines_range = range(num_lines)
        points_per_line = [len(bundle[i]) for i in lines_range]
        points_per_line = np.array(points_per_line, np.intp)

        cols_arr = line_colors(bundle)
        colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
        vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
        colors = numpy_support.vtk_to_numpy(vtk_colors)
        colors = (colors - np.min(colors)) / np.ptp(colors)

        """
        for j in range(n):
            if p_values[j] < buan_thr:
                colors[ind == j] = buan_highlights[i]
        """

        list_actors.append(actor.streamtube(bundle, colors=colors, lod=False))

    grid = ui.GridUI(list_actors, captions=list_labels, dim=(2, 2),
                     rotation_speed=5, rotation_axis=None)

    show_m = window.ShowManager(scene=scene, size=(1920, 1080),
                                reset_camera=False, order_transparent=True)
    show_m.initialize()

    show_m.scene.add(grid)

    show_m.start()
