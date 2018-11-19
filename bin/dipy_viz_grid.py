import nibabel as nib

from fury import window, actor
from fury.colormap import distinguishable_colormap


from fury import interactor
from fury.utils import auto_orient

from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects


def main():
    fetch_bundles_2_subjects()
    dix = read_bundles_2_subjects(subj_id='subj_1', metrics=['fa'],
                                bundles=['cg.left', 'cst.right'])

    streamlines = []
    streamlines += [dix['cg.left']]
    streamlines += [dix['cst.right']]

    bg = (0, 0, 0)
    colormap = distinguishable_colormap(bg=bg)

    ren = window.Renderer()
    ren.background(bg)
    ren.projection("parallel")

    actors = []
    texts = []
    for cluster, color in zip(streamlines, colormap):
        print(color)
        stream_actor = actor.line(cluster, [color]*len(cluster), linewidth=1)
        pretty_actor = auto_orient(stream_actor, ren.camera_direction(), data_up=(0, 0, 1), show_bounds=True)
        pretty_actor_aabb = auto_orient(stream_actor, ren.camera_direction(), bbox_type="AABB", show_bounds=True)

        actors.append(stream_actor)
        actors.append(pretty_actor_aabb)
        actors.append(pretty_actor)

        text = actor.text_3d(str(len(cluster)), font_size=32, justification="center", vertical_justification="top")
        texts.append(text)

        text = actor.text_3d("AABB", font_size=32, justification="center", vertical_justification="top")
        texts.append(text)

        text = actor.text_3d("OBB", font_size=32, justification="center", vertical_justification="top")
        texts.append(text)

    grid = actor.grid(actors, texts, cell_padding=(50, 100), cell_shape="rect")
    ren.add(grid)

    ren.reset_camera_tight()
    show_m = window.ShowManager(ren, interactor_style=interactor.InteractorStyleBundlesGrid(actor))
    show_m.start()


if __name__ == "__main__":
    main()
