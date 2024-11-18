import os
import numpy as np
from fury.v2.window import ShowManager
from fury.v2.actor import lines

from dipy.io.streamline import load_tractogram
from fury.colormap import distinguishable_colormap

fname = os.path.expanduser(
    '~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/AC.trk')

sft = load_tractogram(fname, 'same', bbox_valid_check=False)
streamlines = sft.streamlines

# Pygfx window setup
show_m = ShowManager()
# renderer.blend_mode = 'weighted'

# Pygfx line porting
nan_buffer = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32)
pygfx_streamlines = streamlines._data.copy().astype(np.float32)
pygfx_offsets = streamlines._offsets.copy().astype(np.float32)
pygfx_lengths = streamlines._lengths.copy().astype(np.float32)

color_gen = distinguishable_colormap()

no_streamlines = len(pygfx_offsets)
no_vertices = len(pygfx_streamlines) + no_streamlines

pygfx_colors = np.zeros((no_vertices, 4), dtype=np.float32)

for i in range(no_streamlines):
    color = next(color_gen)
    start_idx = int(pygfx_offsets[i] + i)
    end_idx = int(start_idx + pygfx_lengths[i])
    pygfx_streamlines = np.insert(
        pygfx_streamlines, end_idx, nan_buffer, axis=0)
    pygfx_colors[start_idx:end_idx] = (*color, 1)

# Pygfx line definition
lines = lines(
    positions=pygfx_streamlines,
    colors=pygfx_colors,
    color_mode='vertex'
)

show_m.scene.add(lines)

# # Pygfx camera setup
# camera = gfx.PerspectiveCamera(100, 16 / 9)
# camera.local.position = (100, 100, 50)
# camera.show_pos((0, 0, 0))
# controller = gfx.OrbitController(camera, register_events=renderer)


@lines.add_event_handler("pointer_down")
def on_pick(event):
    vertex = event.pick_info["vertex_index"]
    print("Vertex : ", vertex)
    print("Vertex Coord: ", pygfx_streamlines[vertex])
    selected = find_line(vertex)
    color = pygfx_colors[selected[0]][:3]
    pygfx_colors[selected[0]:selected[1] + 1] = (*color, 0.5)
    lines.geometry.colors.update_range()
    show_m.update()


def find_line(vertex):
    left = right = vertex

    while not np.isnan(pygfx_streamlines[left][0]):
        left -= 1

    while not np.isnan(pygfx_streamlines[right][0]):
        right += 1

    print(left, right)
    return [left + 1, right - 1]


if __name__ == '__main__':
    show_m.render()
    show_m.start()
