// A single f32 cannot represent every 26-bit glyph ID. Two 13-bit components
// are exact in f32 and let pygfx generate ordinary float varyings without
// patching its generated integer declarations to add @interpolate(flat).
fn billboard_encode_glyph_index(index: u32) -> vec2<f32> {
    return vec2<f32>(f32(index & 8191u), f32(index >> 13u));
}

fn billboard_decode_glyph_index(parts: vec2<f32>) -> u32 {
    // All six vertices carry the same pair; round away interpolation roundoff
    // before reconstructing the integer for the unchanged 26-bit pick field.
    let bits = vec2<u32>(round(parts));
    return bits.x | (bits.y << 13u);
}

fn billboard_quad_corner(index: u32) -> vec2<f32> {
    switch index % 6u {
        case 0u: { return vec2<f32>(-0.5, -0.5); }
        case 1u: { return vec2<f32>(0.5, -0.5); }
        case 2u: { return vec2<f32>(-0.5, 0.5); }
        case 3u: { return vec2<f32>(0.5, -0.5); }
        case 4u: { return vec2<f32>(0.5, 0.5); }
        default: { return vec2<f32>(-0.5, 0.5); }
    }
}

fn impostor_sphere_hit(origin: vec3<f32>, direction: vec3<f32>, radius: f32, discriminant_tolerance: f32) -> f32 {
    let a = dot(direction, direction);
    if (a <= 0.0) {
        return -1.0;
    }
    let b = dot(direction, origin);
    let c = dot(origin, origin) - radius * radius;
    var discriminant = b * b - a * c;
    if (discriminant < 0.0) {
        if (discriminant > -discriminant_tolerance) {
            discriminant = 0.0;
        } else {
            return -1.0;
        }
    }
    let sqrt_disc = sqrt(discriminant);
    let near_t = (-b - sqrt_disc) / a;
    if (near_t >= 0.0) {
        return near_t;
    }
    let far_t = (-b + sqrt_disc) / a;
    if (far_t >= 0.0) {
        return far_t;
    }
    return -1.0;
}

fn impostor_depth(world_pos: vec3<f32>) -> f32 {
    let clip_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);
    return clip_pos.z / clip_pos.w;
}

$$ if lighting == 'phong'
struct ReflectedLight {
    direct_diffuse: vec3<f32>,
    direct_specular: vec3<f32>,
    indirect_diffuse: vec3<f32>,
    indirect_specular: vec3<f32>,
};

fn impostor_phong(world_pos: vec3<f32>, normal: vec3<f32>, view_dir: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let physical_albedo = albedo;
    let specular_strength = 1.0;

    var reflected_light: ReflectedLight = ReflectedLight(
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
    );

    var geometry: GeometricContext;
    geometry.position = world_pos;
    geometry.normal = normal;
    geometry.view_dir = view_dir;

    var material: BlinnPhongMaterial;
    material.diffuse_color = physical_albedo;
    material.specular_color = srgb2physical(u_material.specular_color.rgb);
    material.specular_shininess = u_material.shininess;
    material.specular_strength = specular_strength;

    {$ include 'pygfx.light_punctual.wgsl' $}

    let ambient_color = u_ambient_light.color.rgb;
    var irradiance = getAmbientLightIrradiance(ambient_color);
    RE_IndirectDiffuse(irradiance, geometry, material, &reflected_light);

    var emissive_color = srgb2physical(u_material.emissive_color.rgb) * u_material.emissive_intensity;

    var physical_color = reflected_light.direct_diffuse +
        reflected_light.direct_specular +
        reflected_light.indirect_diffuse +
        reflected_light.indirect_specular +
        emissive_color;

    if (all(physical_color == vec3<f32>(0.0))) {
        let fallback_light = normalize(vec3<f32>(0.3, 0.5, 0.8));
        let fallback_diffuse = max(dot(normal, fallback_light), 0.0);
        physical_color = physical_albedo * clamp(0.3 + 0.7 * fallback_diffuse, 0.0, 1.0);
    }
    return physical_color;
}
$$ endif
