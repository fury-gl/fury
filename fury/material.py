import warnings


from fury.shaders import add_shader_callback, load, shader_to_actor
from fury.lib import VTK_OBJECT, calldata_type


class __PBRParams:
    """Helper class to manage PBR parameters.

    Attributes
    ----------
    actor_properties : vtkProperty
        The actor properties.

    Parameters
    ----------
    metallic : float
        Metallic or non-metallic (dielectric) shading computation value. Values
        must be between 0.0 and 1.0.
    roughness : float
        Parameter used to specify how glossy the actor should be. Values must
        be between 0.0 and 1.0.
    anisotropy : float
        Isotropic or anisotropic material parameter. Values must be between
        0.0 and 1.0.
    anisotropy_rotation : float
        Rotation of the anisotropy around the normal in a counter-clockwise
        fashion. Values must be between 0.0 and 1.0. A value of 1.0 means a
        rotation of 2 * pi.
    coat_strength : float
        Strength of the coat layer. Values must be between 0.0 and 1.0 (0.0
        means no clear coat will be modeled).
    coat_roughness : float
        Roughness of the coat layer. Values must be between 0.0 and 1.0.
    base_ior : float
        Index of refraction of the base material. Default is 1.5. Values must
        be between 1.0 and 2.3.
    coat_ior : float
        Index of refraction of the coat material. Default is 1.5. Values must
        be between 1.0 and 2.3.
    """
    def __init__(self, actor_properties, metallic, roughness,
                 anisotropy, anisotropy_rotation, coat_strength,
                 coat_roughness, base_ior, coat_ior):
        self.__actor_properties = actor_properties
        self.__actor_properties.SetMetallic(metallic)
        self.__actor_properties.SetRoughness(roughness)
        self.__actor_properties.SetAnisotropy(anisotropy)
        self.__actor_properties.SetAnisotropyRotation(
            anisotropy_rotation)
        self.__actor_properties.SetCoatStrength(coat_strength)
        self.__actor_properties.SetCoatRoughness(coat_roughness)
        self.__actor_properties.SetBaseIOR(base_ior)
        self.__actor_properties.SetCoatIOR(coat_ior)

    @property
    def metallic(self):
        return self.__actor_properties.GetMetallic()

    @metallic.setter
    def metallic(self, metallic):
        self.__actor_properties.SetMetallic(metallic)

    @property
    def roughness(self):
        return self.__actor_properties.GetRoughness()

    @roughness.setter
    def roughness(self, roughness):
        self.__actor_properties.SetRoughness(roughness)

    @property
    def anisotropy(self):
        return self.__actor_properties.GetAnisotropy()

    @anisotropy.setter
    def anisotropy(self, anisotropy):
        self.__actor_properties.SetAnisotropy(anisotropy)

    @property
    def anisotropy_rotation(self):
        return self.__actor_properties.GetAnisotropyRotation()

    @anisotropy_rotation.setter
    def anisotropy_rotation(self, anisotropy_rotation):
        self.__actor_properties.SetAnisotropyRotation(anisotropy_rotation)

    @property
    def coat_strength(self):
        return self.__actor_properties.GetCoatStrength()

    @coat_strength.setter
    def coat_strength(self, coat_strength):
        self.__actor_properties.SetCoatStrength(coat_strength)

    @property
    def coat_roughness(self):
        return self.__actor_properties.GetCoatRoughness()

    @coat_roughness.setter
    def coat_roughness(self, coat_roughness):
        self.__actor_properties.SetCoatRoughness(coat_roughness)

    @property
    def base_ior(self):
        return self.__actor_properties.GetBaseIOR()

    @base_ior.setter
    def base_ior(self, base_ior):
        self.__actor_properties.SetBaseIOR(base_ior)

    @property
    def coat_ior(self):
        return self.__actor_properties.GetCoatIOR()

    @coat_ior.setter
    def coat_ior(self, coat_ior):
        self.__actor_properties.SetCoatIOR(coat_ior)


def manifest_pbr(actor, metallic=0, roughness=.5, anisotropy=0,
                 anisotropy_rotation=0, coat_strength=0, coat_roughness=0,
                 base_ior=1.5, coat_ior=2):
    """Apply VTK's Physically Based Rendering properties to the selected actor.

    Parameters
    ----------
    actor : actor
    metallic : float, optional
        Metallic or non-metallic (dielectric) shading computation value. Values
        must be between 0.0 and 1.0.
    roughness : float, optional
        Parameter used to specify how glossy the actor should be. Values must
        be between 0.0 and 1.0.
    anisotropy : float, optional
        Isotropic or anisotropic material parameter. Values must be between
        0.0 and 1.0.
    anisotropy_rotation : float, optional
        Rotation of the anisotropy around the normal in a counter-clockwise
        fashion. Values must be between 0.0 and 1.0. A value of 1.0 means a
        rotation of 2 * pi.
    coat_strength : float, optional
        Strength of the coat layer. Values must be between 0.0 and 1.0 (0.0
        means no clear coat will be modeled).
    coat_roughness : float, optional
        Roughness of the coat layer. Values must be between 0.0 and 1.0.
    base_ior : float, optional
        Index of refraction of the base material. Default is 1.5. Values must
        be between 1.0 and 2.3.
    coat_ior : float, optional
        Index of refraction of the coat material. Default is 1.5. Values must
        be between 1.0 and 2.3.

    """
    try:
        prop = actor.GetProperty()
        try:
            prop.SetInterpolationToPBR()

            #pbr_params = {'metallic': metallic, 'roughness': roughness,
            #              'anisotropy': anisotropy,
            #              'anisotropy_rotation': anisotropy_rotation,
            #              'coat_strength': coat_strength,
            #              'coat_roughness': coat_roughness,
            #              'base_ior': base_ior, 'coat_ior': coat_ior}

            pbr_params = __PBRParams(prop, metallic, roughness, anisotropy,
                                     anisotropy_rotation, coat_strength,
                                     coat_roughness, base_ior, coat_ior)
            #prop.SetMetallic(pbr_params['metallic'])
            #prop.SetRoughness(pbr_params['roughness'])
            #prop.SetAnisotropy(pbr_params['anisotropy'])
            #prop.SetAnisotropyRotation(pbr_params['anisotropy_rotation'])
            #prop.SetCoatStrength(pbr_params['coat_strength'])
            #prop.SetCoatRoughness(pbr_params['coat_roughness'])
            #prop.SetBaseIOR(pbr_params['base_ior'])
            #prop.SetCoatIOR(pbr_params['coat_ior'])
            return pbr_params
        except AttributeError:
            warnings.warn(
                'PBR interpolation cannot be applied to this actor. The '
                'material will not be applied.')
            return None
    except AttributeError:
        warnings.warn('Actor does not have the attribute property. This '
                      'material will not be applied.')
        return None


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

