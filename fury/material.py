import os
import warnings

from fury.lib import VTK_OBJECT, calldata_type
from fury.shaders import (
    add_shader_callback,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)


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
            pbr_params = __PBRParams(prop, metallic, roughness, anisotropy,
                                     anisotropy_rotation, coat_strength,
                                     coat_roughness, base_ior, coat_ior)
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


def manifest_principled(actor, subsurface=0, metallic=0, specular=0,
                        specular_tint=0, roughness=0, anisotropic=0,
                        anisotropic_direction=[0, 1, .5], sheen=0,
                        sheen_tint=0, clearcoat=0, clearcoat_gloss=0):
    """Apply the Principled Shading properties to the selected actor.

    Parameters
    ----------
    actor : actor
    subsurface : float, optional
        Subsurface scattering computation value. Values must be between 0.0 and
        1.0.
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
            'subsurface': subsurface, 'metallic': metallic,
            'specular': specular, 'specular_tint': specular_tint,
            'roughness': roughness, 'anisotropic': anisotropic,
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
                    'anisotropicDirection', principled_params[
                        'anisotropic_direction'])

        add_shader_callback(actor, uniforms_callback)

        # Start of shader implementation

        # Adding required constants
        pi = '#define PI 3.14159265359'

        # Adding uniforms
        uniforms = """
        uniform float subsurface;
        uniform float metallic;
        uniform float specularTint;
        uniform float roughness;
        uniform float anisotropic;
        uniform float sheen;
        uniform float sheenTint;
        uniform float clearcoat;
        uniform float clearcoatGloss;

        uniform vec3 anisotropicDirection;
        """

        # Importing functions in order

        # Importing utility functions
        square = import_fury_shader(os.path.join('utils', 'square.glsl'))
        pow5 = import_fury_shader(os.path.join('utils', 'pow5.glsl'))

        # Importing utility function to update the tangent and bitangent
        # vectors given a direction of anisotropy
        update_tan_bitan = import_fury_shader(
            os.path.join('utils', 'update_tan_bitan.glsl')
        )

        # Importing color conversion gamma to linear space function
        gamma_to_linear = import_fury_shader(
            os.path.join('lighting', 'gamma_to_linear.frag')
        )

        # Importing color conversion linear to gamma space function
        linear_to_gamma = import_fury_shader(
            os.path.join('lighting', 'linear_to_gamma.frag')
        )

        # Importing linear-space CIE luminance tint approximation function
        cie_color_tint = import_fury_shader(
            os.path.join('lighting', 'cie_color_tint.frag')
        )

        # Importing Schlick's weight approximation of the Fresnel equation
        schlick_weight = import_fury_shader(
            os.path.join('lighting', 'schlick_weight.frag')
        )

        # Importing Normal Distribution Function (NDF): Generalized
        # Trowbridge-Reitz with param gamma=1 (D_{GTR_1}) needed for the Clear
        # Coat lobe
        gtr1 = import_fury_shader(
            os.path.join('lighting', 'ndf', 'gtr1.frag')
        )

        # Importing Normal Distribution Function (NDF): Generalized
        # Trowbridge-Reitz with param gamma=2 (D_{GTR_2}) needed for the
        # Isotropic Specular lobe
        gtr2 = import_fury_shader(
            os.path.join('lighting', 'ndf', 'gtr2.frag')
        )

        # Importing Normal Distribution Function (NDF): Anisotropic form of the
        # Generalized Trowbridge-Reitz with param gamma=2
        # (D_{GTR_2anisotropic}) needed for the respective Specular lobe
        gtr2_anisotropic = import_fury_shader(
            os.path.join('lighting', 'ndf', 'gtr2_anisotropic.frag')
        )

        # Importing Geometry Shadowing and Masking Function (GF): Smith Ground
        # Glass Unknown (G_{GGX}) needed for the Isotropic Specular and Clear
        # Coat lobes
        smith_ggx = import_fury_shader(
            os.path.join('lighting', 'gf', 'smith_ggx.frag')
        )

        # Importing Geometry Shadowing and Masking Function (GF): Anisotropic
        # form of the Smith Ground Glass Unknown (G_{GGXanisotropic}) needed
        # for the respective Specular lobe
        smith_ggx_anisotropic = import_fury_shader(
            os.path.join('lighting', 'gf', 'smith_ggx_anisotropic.frag')
        )

        # Importing Principled components functions
        diffuse = import_fury_shader(
            os.path.join('lighting', 'principled', 'diffuse.frag')
        )
        subsurface = import_fury_shader(
            os.path.join('lighting', 'principled', 'subsurface.frag')
        )
        sheen = import_fury_shader(
            os.path.join('lighting', 'principled', 'sheen.frag')
        )
        specular_isotropic = import_fury_shader(
            os.path.join('lighting', 'principled', 'specular_isotropic.frag')
        )
        specular_anisotropic = import_fury_shader(
            os.path.join('lighting', 'principled', 'specular_anisotropic.frag')
        )
        clearcoat = import_fury_shader(
            os.path.join('lighting', 'principled', 'clearcoat.frag')
        )

        # Putting all the functions together before passing them to the actor
        fs_dec = compose_shader([
            pi, uniforms, square, pow5, update_tan_bitan, gamma_to_linear,
            linear_to_gamma, cie_color_tint, schlick_weight, gtr1, gtr2,
            gtr2_anisotropic, smith_ggx, smith_ggx_anisotropic, diffuse,
            subsurface, sheen, specular_isotropic, specular_anisotropic,
            clearcoat
        ])

        # Adding shader functions to actor
        shader_to_actor(actor, 'fragment', decl_code=fs_dec)

        # Start of the implementation code
        start_comment = "//Disney's Principled BRDF"

        # Preparing vectors and values
        normal = 'vec3 normal = normalVCVSOutput;'
        # VTK's default system is retroreflective, which means view = light
        view = 'vec3 view = normalize(-vertexVC.xyz);'
        # Since VTK's default setup is retroreflective we only need to
        # calculate one single dot product
        dot_n_v = 'float dotNV = clamp(dot(normal, view), 1e-5, 1);'

        dot_n_v_validation = """
        if(dotNV < 0)
            fragOutput0 = vec4(vec3(0), opacity);
        """

        # To work with anisotropic distributions is necessary to have a tangent
        # and bitangent vector per point on the surface
        tangent = 'vec3 tangent = vec3(.0);'
        bitangent = 'vec3 bitangent = vec3(.0);'
        # The shader function updateTanBitan aligns tangents and bitangents
        # according to a direction of anisotropy
        update_aniso_vecs = """
        updateTanBitan(normal, anisotropicDirection, tangent, bitangent);
        """

        # Calculating dot products with tangent and bitangent
        dot_t_v = 'float dotTV = dot(tangent, view);'
        dot_b_v = 'float dotBV = dot(bitangent, view);'

        # Converting color to linear space
        linear_color = 'vec3 linColor = gamma2Linear(diffuseColor);'

        # Calculating linear-space CIE luminance tint approximation
        tint = 'vec3 tint = calculateTint(linColor);'

        # Since VTK's default setup is retroreflective we only need to
        # calculate one single Schlick's weight
        fsw = 'float fsw = schlickWeight(dotNV);'

        # Calculating the diffuse coefficient
        diff_coeff = """
        float diffCoeff = evaluateDiffuse(roughness, fsw, fsw, dotNV);
        """

        # Calculating the subsurface coefficient
        subsurf_coeff = """
        float subsurfCoeff = evaluateSubsurface(roughness, fsw, fsw, dotNV,
            dotNV, dotNV);
        """

        # Calculating the sheen irradiance
        sheen_rad = """
        vec3 sheenRad = evaluateSheen(sheen, sheenTint, tint, fsw);
        """

        # Calculating the specular irradiance
        spec_rad = """
        vec3 specRad = evaluateSpecularAnisotropic(specularIntensity,
            specularTint, metallic, anisotropic, roughness, tint, linColor,
            fsw, dotNV, dotTV, dotBV, dotNV, dotTV, dotBV, dotNV, dotTV,
            dotBV);
        """

        # Calculating the clear coat coefficient
        clear_coat_coef = """
        float coatCoeff = evaluateClearcoat(clearcoat, clearcoatGloss, fsw,
            dotNV, dotNV, dotNV);
        """

        # Starting to put all together
        # Initializing the radiance vector
        radiance = 'vec3 rad = (1 / PI) * linColor;'
        # Adding mix between the diffuse and the subsurface coefficients
        # controlled by the subsurface parameter
        diff_subsurf_mix = 'rad *= mix(diffCoeff, subsurfCoeff, subsurface);'
        # Adding sheen radiance
        sheen_add = 'rad += sheenRad;'
        # Balancing energy using metallic
        metallic_balance = 'rad *= (1 - metallic);'
        # Adding specular radiance
        specular_add = 'rad += specRad;'
        # Adding clear coat coefficient
        clearcoat_add = 'rad += coatCoeff;'

        # Initializing the color vector using the final radiance and VTK's
        # additional information
        color = 'vec3 color = rad * lightColor0;'
        # Converting color back to gamma space
        gamma_color = 'color = linear2Gamma(color);'
        # Clamping color values
        color_clamp = 'color = clamp(color, vec3(0), vec3(1));'

        # Fragment shader output
        frag_output = 'fragOutput0 = vec4(color, opacity);'

        # Putting all the implementation together before passing it to the
        # actor
        fs_impl = compose_shader([
            start_comment, normal, view, dot_n_v, dot_n_v_validation, tangent,
            bitangent, update_aniso_vecs, dot_t_v, dot_b_v, linear_color, tint,
            fsw, diff_coeff, subsurf_coeff, sheen_rad, spec_rad,
            clear_coat_coef, radiance, diff_subsurf_mix, sheen_add,
            metallic_balance, specular_add, clearcoat_add, color, gamma_color,
            color_clamp, frag_output
        ])

        # Adding shader implementation to actor
        shader_to_actor(actor, 'fragment', impl_code=fs_impl, block='light')

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
