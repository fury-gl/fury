import warnings


from fury.shaders import add_shader_callback, load, shader_to_actor
from fury.lib import VTK_9_PLUS, VTK_OBJECT, calldata_type


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
        warnings.warn('Your PBR effect cannot be apply due to VTK version. '
                      'Please upgrade your VTK version (should be >= 9.0.0).')
        return

    try:
        prop = actor.GetProperty()
        try:
            prop.SetInterpolationToPBR()
            prop.SetMetallic(metallicity)
            prop.SetRoughness(roughness)
        except AttributeError:
            warnings.warn(
                'PBR interpolation cannot be applied to this actor. The '
                'material will not be applied.')
            return
    except AttributeError:
        warnings.warn('Actor does not have the attribute property. This '
                      'material will not be applied.')
        return


def manifest_principled(actor, subsurface=0, subsurface_color=[0, 0, 0],
                        metallic=0, specular=0, specular_tint=0, roughness=0,
                        anisotropic=0, anisotropic_direction=[0, 1, .5],
                        sheen=0, sheen_tint=0, clearcoat=0, clearcoat_gloss=0):
    """Apply the Principled Shading properties to the selected actor.

    Parameters
    ----------
    actor : actor
    subsurface : float, optional
        Subsurface scattering computation value. Values must be between 0.0 and
        1.0.
    subsurface_color : list, optional
        Subsurface scattering RGB color where R, G and B should be in the range
        [0, 1].
    metallic : float, optional
        Metallic or non-metallic (dielectric) shading computation value. Values
        must be between 0.0 and 1.0.
    specular : float, optional
        Specular lighting coefficient. Value must be between 0.0 and 1.0.
    specular_tint : float, optional
        Specular tint coefficient value. Values must be between 0.0 and 1.0.
    roughness : float, optional
        Parameter used to specify how glossy the actor should be. Values must
        be between 0.0 and 1.0.
    anisotropic : float, optional
        Anisotropy coefficient. Values must be between 0.0 and 1.0.
    anisotropic_direction : list, optional
        Anisotropy direction where X, Y and Z should be in the range [-1, 1].
    sheen : float, optional
        Sheen coefficient. Values must be between 0.0 and 1.0.
    sheen_tint : float, optional
        Sheen tint coefficient value. Values must be between 0.0 and 1.0.
    clearcoat : float, optional
        Clearcoat coefficient. Values must be between 0.0 and 1.0.
    clearcoat_gloss : float, optional
        Clearcoat gloss coefficient value. Values must be between 0.0 and 1.0.

    Returns
    -------
    principled_params : dict
        Dictionary containing the Principled Shading parameters.

    """

    try:
        prop = actor.GetProperty()

        principled_params = {
            'subsurface': subsurface, 'subsurface_color': subsurface_color,
            'metallic': metallic, 'specular': specular,
            'specular_tint': specular_tint, 'roughness': roughness,
            'anisotropic': anisotropic,
            'anisotropic_direction': anisotropic_direction, 'sheen': sheen,
            'sheen_tint': sheen_tint, 'clearcoat': clearcoat,
            'clearcoat_gloss': clearcoat_gloss
        }

        prop.SetSpecular(specular)

        @calldata_type(VTK_OBJECT)
        def uniforms_callback(_caller, _event, calldata=None):
            if calldata is not None:
                calldata.SetUniformf(
                    'subsurface', principled_params['subsurface'])
                calldata.SetUniformf(
                    'metallic', principled_params['metallic'])
                calldata.SetUniformf(
                    'specularTint', principled_params['specular_tint'])
                calldata.SetUniformf(
                    'roughness', principled_params['roughness'])
                calldata.SetUniformf(
                    'anisotropic', principled_params['anisotropic'])
                calldata.SetUniformf('sheen', principled_params['sheen'])
                calldata.SetUniformf(
                    'sheenTint', principled_params['sheen_tint'])
                calldata.SetUniformf(
                    'clearcoat', principled_params['clearcoat'])
                calldata.SetUniformf(
                    'clearcoatGloss', principled_params['clearcoat_gloss'])

                calldata.SetUniform3f(
                    'subsurfaceColor', principled_params['subsurface_color'])
                calldata.SetUniform3f(
                    'anisotropicDirection', principled_params[
                        'anisotropic_direction'])

        add_shader_callback(actor, uniforms_callback)

        fs_dec_code = load('bxdf_dec.frag')
        fs_impl_code = load('bxdf_impl.frag')

        shader_to_actor(actor, 'fragment', decl_code=fs_dec_code)
        shader_to_actor(actor, 'fragment', impl_code=fs_impl_code,
                        block='light')
        return principled_params
    except AttributeError:
        warnings.warn('Actor does not have the attribute property. This '
                      'material will not be applied.')
        return None


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

        interpolation = interpolation.lower()

        if interpolation == 'flat':
            prop.SetInterpolationToFlat()
        elif interpolation == 'gouraud':
            prop.SetInterpolationToGouraud()
        elif interpolation == 'phong':
            prop.SetInterpolationToPhong()
        else:
            message = 'Unknown interpolation. Ignoring "{}" interpolation ' \
                      'option and using the default ("{}") option.'
            message = message.format(interpolation, 'gouraud')
            warnings.warn(message)

        prop.SetAmbient(ambient_level)
        prop.SetAmbientColor(ambient_color)
        prop.SetDiffuse(diffuse_level)
        prop.SetDiffuseColor(diffuse_color)
        prop.SetSpecular(specular_level)
        prop.SetSpecularColor(specular_color)
        prop.SetSpecularPower(specular_power)
    except AttributeError:
        warnings.warn('Actor does not have the attribute property. This '
                      'material will not be applied.')
        return

