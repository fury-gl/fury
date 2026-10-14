{$ include 'pygfx.std.wgsl' $}
{$ include 'pygfx.light_phong.wgsl' $}
{$ include 'fury.billboard_common.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) index: u32,
    @builtin(instance_index) glyph_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    let glyph = i32(in.glyph_index);
    let center = (u_wobject.world_transform * vec4<f32>(load_s_positions(glyph), 1.0)).xyz;
    let axis0 = (u_wobject.world_transform * vec4<f32>(load_s_ellipsoid_axes(glyph * 3).xyz, 0.0)).xyz;
    let axis1 = (u_wobject.world_transform * vec4<f32>(load_s_ellipsoid_axes(glyph * 3 + 1).xyz, 0.0)).xyz;
    let axis2 = (u_wobject.world_transform * vec4<f32>(load_s_ellipsoid_axes(glyph * 3 + 2).xyz, 0.0)).xyz;

    let cofactor0 = cross(axis1, axis2);
    let determinant = dot(axis0, cofactor0);
    var inverse_row0 = vec3<f32>(0.0);
    var inverse_row1 = vec3<f32>(0.0);
    var inverse_row2 = vec3<f32>(0.0);
    if (determinant != 0.0) {
        inverse_row0 = cofactor0 / determinant;
        inverse_row1 = cross(axis2, axis0) / determinant;
        inverse_row2 = cross(axis0, axis1) / determinant;
    }

    // A center-plane quad underestimates perspective silhouettes. Project the
    // conservative world AABB instead, with full viewport coverage at the near plane.
    let extent = sqrt(axis0 * axis0 + axis1 * axis1 + axis2 * axis2);
    let world_to_clip = u_stdinfo.projection_transform * u_stdinfo.cam_transform;
    var screen_min = vec2<f32>(1e30);
    var screen_max = vec2<f32>(-1e30);
    var crosses_near = false;
    var outside_left = true;
    var outside_right = true;
    var outside_bottom = true;
    var outside_top = true;
    var outside_near = true;
    var outside_far = true;
    var outside_camera = true;
    for (var corner = 0u; corner < 8u; corner += 1u) {
        let signs = vec3<f32>(
            select(-1.0, 1.0, (corner & 1u) != 0u),
            select(-1.0, 1.0, (corner & 2u) != 0u),
            select(-1.0, 1.0, (corner & 4u) != 0u)
        );
        let clip = world_to_clip * vec4<f32>(center + signs * extent, 1.0);
        outside_left = outside_left && clip.x < -clip.w;
        outside_right = outside_right && clip.x > clip.w;
        outside_bottom = outside_bottom && clip.y < -clip.w;
        outside_top = outside_top && clip.y > clip.w;
        outside_near = outside_near && clip.z < 0.0;
        outside_far = outside_far && clip.z > clip.w;
        outside_camera = outside_camera && clip.w <= 0.0;
        crosses_near = crosses_near || clip.z <= 0.0 || clip.w <= 0.0;
        if (clip.w > 0.0) {
            let ndc = clip.xy / clip.w;
            screen_min = min(screen_min, ndc);
            screen_max = max(screen_max, ndc);
        }
    }
    if (crosses_near) {
        screen_min = vec2<f32>(-1.0);
        screen_max = vec2<f32>(1.0);
    } else {
        screen_min = max(screen_min, vec2<f32>(-1.0));
        screen_max = min(screen_max, vec2<f32>(1.0));
    }
    let culled = determinant == 0.0 || outside_left || outside_right ||
        outside_bottom || outside_top || outside_near || outside_far || outside_camera;
    if (culled) {
        screen_min = vec2<f32>(-2.0);
        screen_max = screen_min;
    }

    let quad_uv = billboard_quad_corner(in.index) + vec2<f32>(0.5);
    let ndc_xy = mix(screen_min, screen_max, quad_uv);
    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_xy, 0.5, 1.0);
    varyings.ndc_xy = vec2<f32>(ndc_xy);
    varyings.world_pos = vec3<f32>(center);
    varyings.ellipsoid_center = vec3<f32>(center);
    varyings.inverse_row0 = vec3<f32>(inverse_row0);
    varyings.inverse_row1 = vec3<f32>(inverse_row1);
    varyings.inverse_row2 = vec3<f32>(inverse_row2);
    varyings.color = vec4<f32>(load_s_colors(glyph));
    // Each vertex carries the same two exact 13-bit float components. A single
    // f32 cannot represent all 26-bit IDs; decode and round before integer packing.
    varyings.glyph_index_parts = vec2<f32>(billboard_encode_glyph_index(in.glyph_index));
    return varyings;
}

@fragment
fn fs_main(input: Varyings) -> FragmentOutput {
    var varyings = input;
    let near_pos = ndc_to_world_pos(vec4<f32>(varyings.ndc_xy, 0.0, 1.0));
    let far_pos = ndc_to_world_pos(vec4<f32>(varyings.ndc_xy, 1.0, 1.0));
    var ray_origin = u_stdinfo.cam_transform_inv[3].xyz;
    if (is_orthographic()) {
        ray_origin = near_pos;
    }
    let ray_dir = normalize(far_pos - ray_origin);
    let relative_origin = ray_origin - varyings.ellipsoid_center;
    let local_origin = vec3<f32>(
        dot(varyings.inverse_row0, relative_origin),
        dot(varyings.inverse_row1, relative_origin),
        dot(varyings.inverse_row2, relative_origin)
    );
    // Keep this direction unnormalized: t must remain a world-space ray distance.
    let local_direction = vec3<f32>(
        dot(varyings.inverse_row0, ray_dir),
        dot(varyings.inverse_row1, ray_dir),
        dot(varyings.inverse_row2, ray_dir)
    );
    var t = impostor_sphere_hit(local_origin, local_direction, 1.0, 0.0);
    if (t < 0.0) {
        discard;
    }
    var world_pos = ray_origin + ray_dir * t;
    var depth = impostor_depth(world_pos);
    if (depth < 0.0) {
        // A clipped entry surface can leave the exit surface inside the frustum.
        t = -2.0 * dot(local_direction, local_origin) /
            dot(local_direction, local_direction) - t;
        world_pos = ray_origin + ray_dir * t;
        depth = impostor_depth(world_pos);
    }
    if (t < 0.0 || depth < 0.0 || depth > 1.0) {
        discard;
    }
    varyings.world_pos = vec3<f32>(world_pos);
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let local_hit = local_origin + local_direction * t;
    let world_normal = normalize(
        varyings.inverse_row0 * local_hit.x +
        varyings.inverse_row1 * local_hit.y +
        varyings.inverse_row2 * local_hit.z
    );
    let alpha = varyings.color.a * u_material.opacity;
    do_alpha_test(alpha);
    let albedo = srgb2physical(varyings.color.rgb);
    let physical_color = impostor_phong(world_pos, world_normal, -ray_dir, albedo);

    var out: FragmentOutput;
    out.color = vec4<f32>(physical_color, alpha);
    out.depth = depth;
    $$ if write_pick
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(billboard_decode_glyph_index(varyings.glyph_index_parts), 26) +
        pick_pack(0u, 18)
    );
    $$ endif
    return out;
}
