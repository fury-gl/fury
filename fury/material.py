import vtk
import warnings

VTK_9_PLUS = vtk.vtkVersion.GetVTKMajorVersion() >= 9


def manifest_pbr(actor, metallicity=1, roughness=.5):
    """Apply the Physically Based Rendering properties to the selected actor.

    Parameters
    ----------
    actor : actor
    metallicity : float, optional
        Metallic or non-metallic (dielectric) shading computation value. Values
        must be between 0.0 and 1.0.
    roughness : float, optional
        Parameter used to specify how glossy the actor should be. Values must
        be between 0.0 and 1.0.

    """
    if not VTK_9_PLUS:
        warnings.warn("Your PBR effect can not be apply due to VTK version. "
                      "Please upgrade your VTK version (should be >= 9.0.0).")
        return

    prop = actor.GetProperty()
    prop.SetInterpolationToPBR()
    prop.SetMetallic(metallicity)
    prop.SetRoughness(roughness)


def manifest_standard(actor, ambient_level=.7, diffuse_level=.8,
                      specular_level=.5, specular_power=10,
                      interpolation='phong'):
    """Apply the standard material to the selected actor.

    Parameters
    ----------
    actor : actor
    ambient_level : float, optional
        Metallic or non-metallic (dielectric) shading computation value. Values
        must be between 0.0 and 1.0.
    diffuse_level : float, optional
        Parameter used to specify how glossy the actor should be. Values must
        be between 0.0 and 1.0.
    specular_level : float, optional
        Parameter used to specify how glossy the actor should be. Values must
        be between 0.0 and 1.0.
    specular_power : float, optional
        Parameter used to specify how glossy the actor should be. Values must
        be between 0.0 and 1.0.
    interpolation : float, optional
        Parameter used to specify how glossy the actor should be. Values must
        be between 0.0 and 1.0.

    """
    prop = actor.GetProperty()
    prop.SetAmbient(ambient_level)
    prop.SetDiffuse(diffuse_level)
    prop.SetSpecular(specular_level)
    prop.SetSpecularPower(specular_power)

    if interpolation.lower() == 'phong':
        prop.SetInterpolationToPhong()
    else:
        warnings.warn('Unknown interpolation. Ignoring "{}" interpolation '
                      'option.'.format(interpolation))
