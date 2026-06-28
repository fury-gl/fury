"""Shared helpers for actor module tests."""

from PIL import Image
import numpy as np
import numpy.testing as npt

from fury import actor, window


def _to_255(colors):
    """
    Convert one or more ``[0, 1]`` RGB colors to integer ``[0, 255]`` tuples.

    Parameters
    ----------
    colors : sequence
        A single ``(r, g, b)`` color or a sequence of them, with components in
        ``[0, 1]``.

    Returns
    -------
    list of tuple
        The colors as integer RGB tuples suitable for
        :func:`fury.window.analyze_snapshot`.
    """
    arr = np.atleast_2d(np.asarray(colors, dtype=float))
    return [tuple(int(round(c * 255)) for c in row[:3]) for row in arr]


def snapshot_report(scene, *, colors=None):
    """
    Render ``scene`` offscreen and analyze the resulting image.

    A thin wrapper around :func:`fury.window.snapshot` (with
    ``return_array=True``) and :func:`fury.window.analyze_snapshot`. Performs a
    real offscreen render -- no mocks, patches, or dummy objects.

    Parameters
    ----------
    scene : Scene
        The scene to render.
    colors : sequence of (r, g, b), optional
        Expected colors in ``[0, 1]``. When given, the returned report's
        ``colors_found`` reports whether each color is present in the image.

    Returns
    -------
    ReportSnapshot
        The analysis report, exposing ``objects`` (count of foreground blobs)
        and ``colors_found``.
    """
    rgb255 = _to_255(colors) if colors is not None else None
    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    return window.analyze_snapshot(arr, colors=rgb255, find_objects=True)


def assert_visibility(target, *, toggle, colors=None):
    """
    Assert ``target`` renders when visible and renders nothing when hidden.

    Drives a real offscreen render through :func:`snapshot_report` and uses the
    background-difference object count from
    :func:`fury.window.analyze_snapshot` as the visibility signal. No mocks,
    patches, or dummy classes are involved -- visibility is toggled through the
    object's real API via ``toggle``.

    Parameters
    ----------
    target : Object
        An actor or UI element. It is added to a fresh
        :class:`~fury.window.Scene` for each render (``window.snapshot`` attaches
        a camera to the scene, so a new scene is used per snapshot).
    toggle : callable
        The object's real visibility setter, invoked as ``toggle(True)`` and
        ``toggle(False)`` (e.g. ``element.set_visibility`` for UI, or
        ``lambda v: setattr(actor, "visible", v)`` for a raw actor).
    colors : sequence of (r, g, b), optional
        Expected colors in ``[0, 1]``. When given, they must be present while
        visible and absent while hidden. Reliable only for flat/``"basic"``
        materials where the rendered color matches exactly.
    """

    def _render():
        scene = window.Scene()
        scene.add(target)
        return snapshot_report(scene, colors=colors)

    toggle(True)
    report = _render()
    assert report.objects >= 1, "expected a visible object to render"
    if colors is not None:
        assert all(report.colors_found), "expected colors missing while visible"

    toggle(False)
    report = _render()
    assert report.objects == 0, "expected nothing to render while hidden"
    if colors is not None:
        assert not any(report.colors_found), "colors present while hidden"

    toggle(True)


def random_png(width, height):
    """
    Generates a random RGB PNG image.

    Parameters
    ----------
    width : int
        Width of the image in pixels.
    height : int
        Height of the image in pixels.

    Returns
    -------
    Image
        The generated image.
    """
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for x in range(width):
        for y in range(height):
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            pixels[x, y] = (r, g, b)

    return image


def validate_actors(actor_type="actor_name", prim_count=1, **kwargs):
    scene = window.Scene()
    typ_actor = getattr(actor, actor_type)
    get_actor = typ_actor(**kwargs)
    scene.add(get_actor)

    centers = kwargs.get("centers", None)

    if centers is not None:
        npt.assert_array_equal(get_actor.local.position, centers[0])

        mean_vertex = np.round(np.mean(get_actor.geometry.positions.view, axis=0))
        npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert get_actor.prim_count == prim_count

    if actor_type == "line":
        return

    # Default material: the actor renders and red dominates the image.
    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    assert report.objects >= 1

    mean_r, mean_g, mean_b, _mean_a = np.mean(arr.reshape(-1, arr.shape[2]), axis=0)
    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255

    # Hidden: nothing renders.
    get_actor.visible = False
    report = window.analyze_snapshot(
        window.snapshot(scene=scene, fname=None, return_array=True), find_objects=True
    )
    assert report.objects == 0
    scene.remove(get_actor)

    # Basic (flat) material: exact red is present, then absent once hidden.
    typ_actor_1 = getattr(actor, actor_type)
    get_actor_1 = typ_actor_1(**{**kwargs, "material": "basic"})
    scene.add(get_actor_1)

    report = window.analyze_snapshot(
        window.snapshot(scene=scene, fname=None, return_array=True),
        colors=(255, 0, 0),
        find_objects=True,
    )
    assert report.objects >= 1
    assert report.colors_found == [True]

    get_actor_1.visible = False
    report = window.analyze_snapshot(
        window.snapshot(scene=scene, fname=None, return_array=True),
        colors=(255, 0, 0),
        find_objects=True,
    )
    assert report.objects == 0
    assert report.colors_found == [False]
    scene.remove(get_actor_1)
