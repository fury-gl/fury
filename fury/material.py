import vtk
import warnings

VTK_9_PLUS = vtk.vtkVersion.GetVTKMajorVersion() >= 9


def manifest_pbr(actor, metallicity=0, roughness=.5):
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
        warnings.warn('Your PBR effect can not be apply due to VTK version. '
                      'Please upgrade your VTK version (should be >= 9.0.0).')
        return

    prop = actor.GetProperty()
    prop.SetInterpolationToPBR()
    prop.SetMetallic(metallicity)
    prop.SetRoughness(roughness)


def manifest_standard(actor, ambient_level=0, ambient_color=(1, 1, 1),
                      diffuse_level=1, diffuse_color=(1, 1, 1),
                      specular_level=0, specular_color=(1, 1, 1),
                      specular_power=1, interpolation='gouraud'):
    """Apply the standard material to the selected actor.

    Parameters
    ----------
    actor : actor
    ambient_level : float, optional
        Ambient lighting coefficient. Value must be between 0.0 and 1.0.
    ambient_color : tuple (3,), optional
        Ambient RGB color where R, G and B should be in the range [0, 1].
    diffuse_level : float, optional
        Diffuse lighting coefficient. Value must be between 0.0 and 1.0.
    diffuse_color : tuple (3,), optional
        Diffuse RGB color where R, G and B should be in the range [0, 1].
    specular_level : float, optional
        Specular lighting coefficient. Value must be between 0.0 and 1.0.
    specular_color : tuple (3,), optional
        Specular RGB color where R, G and B should be in the range [0, 1].
    specular_power : float, optional
        Parameter used to specify the intensity of the specular reflection.
        Value must be between 0.0 and 128.0.
    interpolation : string, optional
        If 'flat', the actor will be shaded using flat interpolation. If
        'gouraud' (default), then the shading will be calculated at the
        vertex level. If 'phong', then the shading will be calculated at the
        fragment level.

    """
    try:
        prop = actor.GetProperty()

        if interpolation.lower() == 'flat':
            prop.SetInterpolationToFlat()
        elif interpolation.lower() == 'gouraud':
            prop.SetInterpolationToGouraud()
        elif interpolation.lower() == 'phong':
            prop.SetInterpolationToPhong()
        else:
            warnings.warn(
                'Unknown interpolation. Ignoring "{}" interpolation option. '
                'And using the default ("{}") option.'.format(
                    interpolation, 'gouraud'), UserWarning)

        prop.SetAmbient(ambient_level)
        prop.SetAmbientColor(ambient_color)
        prop.SetDiffuse(diffuse_level)
        prop.SetDiffuseColor(diffuse_color)
        prop.SetSpecular(specular_level)
        prop.SetSpecularColor(specular_color)
        prop.SetSpecularPower(specular_power)
    except AttributeError:
        warnings.warn('Actor does not have the attribute property. This '
                      'material will not be applied.', UserWarning)
        return

