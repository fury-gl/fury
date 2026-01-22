"""Utility functions for manipulating actors in FURY."""

import numpy as np

from fury.lib import AffineTransform, GfxGroup, RecursiveTransform, WorldObject
from fury.material import validate_opacity


def set_group_visibility(group, visibility):
    """Set the visibility of a group of actors.

    Parameters
    ----------
    group : Group
        The group of actors to set visibility for.
    visibility : tuple or list of bool
        If a single boolean value is provided, it sets the visibility for all
        actors in the group. If a tuple or list is provided, it sets the
        visibility for each actor in the group individually.
    """
    if not isinstance(group, GfxGroup):
        raise TypeError("group must be an instance of Group.")

    if not isinstance(visibility, (tuple, list)):
        group.visible = visibility
        return

    if len(visibility) != len(group.children):
        raise ValueError(
            "Length of visibility must match the number of actors in the group."
        )

    for idx, actor in enumerate(group.children):
        actor.visible = visibility[idx]


def set_group_opacity(group, opacity):
    """Set the opacity of the group of actors.

    Parameters
    ----------
    group : Group
        The group of actors to set opacity for.
    opacity : float
        The opacity value to set for the group of actors,
        ranging from 0 (fully transparent) to 1 (opaque).
    """
    if not isinstance(group, GfxGroup):
        raise TypeError("group must be an instance of Group.")

    opacity = validate_opacity(opacity)

    for child in group.children:
        child.material.opacity = opacity


def set_opacity(actor, opacity):
    """Set the opacity of an actor.

    Parameters
    ----------
    actor : WorldObject
        The actor to set opacity for.
    opacity : float
        The opacity value to set for the actor,
        ranging from 0 (fully transparent) to 1 (opaque).
    """
    if not isinstance(actor, WorldObject):
        raise TypeError("actor must be an instance of WorldObject.")

    if isinstance(actor, GfxGroup):
        set_group_opacity(actor, opacity)
        return

    opacity = validate_opacity(opacity)
    actor.material.opacity = opacity


def validate_slices_group(group):
    """Validate the slices in a group.

    Parameters
    ----------
    group : Group
        The group of actors to validate.

    Raises
    ------
    TypeError
        If the group is not an instance of Group.
    ValueError
        If the group does not contain exactly 3 children.
    AttributeError
        If the children do not have the required material plane attribute.
    """
    if not isinstance(group, GfxGroup):
        raise TypeError("group must be an instance of Group.")

    if len(group.children) != 3:
        raise ValueError(
            f"Group must contain exactly 3 children. {len(group.children)}"
        )

    if not (
        hasattr(group.children[0].material, "plane")
        or hasattr(group.children[0], "plane")
    ):
        raise AttributeError(
            "Children do not have the required material plane attribute for slices."
        )


def get_slices(group):
    """Get the current positions of the slices.

    Parameters
    ----------
    group : Group
        The group of actors to get the slices from.

    Returns
    -------
    ndarray
        An array containing the current positions of the slices.
    """
    validate_slices_group(group)
    return np.asarray([child.material.plane[-1] for child in group.children])


def show_slices(group, position):
    """Show the slices at the specified position.

    Parameters
    ----------
    group : Group
        The group of actors to get the slices from.
    position : tuple
        A tuple containing the positions of the slices in the 3D space.
    """
    validate_slices_group(group)

    for i, child in enumerate(group.children):
        if hasattr(child, "plane"):
            a, b, c, _ = child.plane
            child.plane = (a, b, c, position[i])
        else:
            a, b, c, _ = child.material.plane
            child.material.plane = (a, b, c, position[i])


def apply_affine_to_group(group, affine):
    """Apply a transformation to all actors in a group.

    Parameters
    ----------
    group : Group
        The group of actors to apply the transformation to.
    affine : ndarray, shape (4, 4)
        The transformation to apply to the actors in the group.
    """
    if not isinstance(group, GfxGroup):
        raise TypeError("group must be an instance of Group.")

    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("affine must be a 4x4 numpy array.")

    for child in group.children:
        apply_affine_to_actor(child, affine)


def apply_affine_to_actor(actor, affine):
    """Apply a transformation to an actor.

    Parameters
    ----------
    actor : WorldObject
        The actor to apply the transformation to.
    affine : ndarray, shape (4, 4)
        The transformation to apply to the actor.
    """
    if not isinstance(actor, WorldObject):
        raise TypeError("actor must be an instance of WorldObject.")

    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("affine must be a 4x4 numpy array.")

    affine_transform = AffineTransform(
        state_basis="matrix", matrix=affine, is_camera_space=True
    )
    recursive_transform = RecursiveTransform(affine_transform)
    actor.local = affine_transform
    actor.world = recursive_transform
