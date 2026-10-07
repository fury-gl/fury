{$ include 'pygfx.std.wgsl' $}
{$ include 'pygfx.light_phong.wgsl' $}
{$ include 'fury.utils.wgsl' $}
{$ include 'fury.billboard_common.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    // Generate quad vertices for billboard impostor
    let billboard_index = i32(in.index) / 6;
    let vertex_in_quad = i32(in.index) % 6;
    let local_pos = billboard_quad_corner(in.index);

    let raw_center = load_s_positions(billboard_index * 6);
    let world_center = u_wobject.world_transform * vec4<f32>(raw_center.xyz, 1.0);

    let cam_right = vec3<f32>(u_stdinfo.cam_transform_inv[0].xyz);
    let cam_up = vec3<f32>(u_stdinfo.cam_transform_inv[1].xyz);

    let raw_size = load_s_normals(billboard_index * 6);
    let size = raw_size.xy;

    let billboard_offset = local_pos.x * cam_right * size.x + local_pos.y * cam_up * size.y;
    let world_pos = world_center.xyz + billboard_offset;

    let clip_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);

    var tex_coord: vec2<f32>;
    switch vertex_in_quad {
        case 0: { tex_coord = vec2<f32>(0.0, 0.0); }
        case 1: { tex_coord = vec2<f32>(1.0, 0.0); }
        case 2: { tex_coord = vec2<f32>(0.0, 1.0); }
        case 3: { tex_coord = vec2<f32>(1.0, 0.0); }
        case 4: { tex_coord = vec2<f32>(1.0, 1.0); }
        default: { tex_coord = vec2<f32>(0.0, 1.0); }
    }

    var varyings: Varyings;
    varyings.position = vec4<f32>(clip_pos);
    varyings.world_pos = vec3<f32>(world_pos);
    varyings.glyph_index_parts = vec2<f32>(billboard_encode_glyph_index(u32(billboard_index)));
    $$ if color_buffer_channels == 4
    varyings.color = vec4<f32>(load_s_colors(billboard_index * 6));
    $$ elif color_buffer_channels == 3
    varyings.color = vec4<f32>(load_s_colors(billboard_index * 6), 1.0);
    $$ elif color_buffer_channels == 2
    let cvalue = load_s_colors(billboard_index * 6);
    varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
    $$ elif color_buffer_channels == 1
    let cvalue = load_s_colors(billboard_index * 6);
    varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
    $$ endif
    varyings.texcoord_vert = vec2<f32>(tex_coord);
    varyings.billboard_center = vec3<f32>(world_center.x, world_center.y, world_center.z);
    varyings.billboard_right = vec3<f32>(cam_right.x, cam_right.y, cam_right.z);
    varyings.billboard_up = vec3<f32>(cam_up.x, cam_up.y, cam_up.z);
    varyings.billboard_size = vec2<f32>(size);

    return varyings;
}


@fragment
fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let uv = varyings.texcoord_vert.xy;
    let coord = uv * 2.0 - vec2<f32>(1.0);
    let radius_sq = dot(coord, coord);
    if (radius_sq > 1.0) {
        discard;
    }

    let smooth_edge = fwidth(radius_sq);
    let mask = clamp(1.0 - smoothstep(1.0 - smooth_edge, 1.0 + smooth_edge, radius_sq), 0.0, 1.0);

    let radius = 0.5 * varyings.billboard_size.x;
    let center = varyings.billboard_center;
    let plane_pos = varyings.world_pos;
    let cam_pos = u_stdinfo.cam_transform_inv[3].xyz;
    let cam_forward = normalize((u_stdinfo.cam_transform_inv * vec4<f32>(0.0, 0.0, -1.0, 0.0)).xyz);
    let ortho = is_orthographic();

    var ray_origin = cam_pos;
    var ray_dir = plane_pos - cam_pos;
    if (ortho) {
        ray_origin = plane_pos;
        ray_dir = -cam_forward;
    } else {
        let dir_len = length(ray_dir);
        if (dir_len > 0.0) {
            ray_dir = ray_dir / dir_len;
        } else {
            ray_dir = cam_forward;
        }
    }
    ray_dir = normalize(ray_dir);

    let t = impostor_sphere_hit(ray_origin - center, ray_dir, radius, 1e-4);
    if (t < 0.0) {
        discard;
    }
    let world_pos = ray_origin + ray_dir * t;
    var world_normal = normalize(world_pos - center);
    if (!is_front) {
        world_normal = -world_normal;
    }

    var view_dir = -ray_dir;
    if (ortho) {
        view_dir = ray_dir;
    }
    view_dir = normalize(view_dir);

    var diffuse_color = vec4<f32>(srgb2physical(varyings.color.rgb), varyings.color.a);
    diffuse_color.a *= u_material.opacity * mask;
    do_alpha_test(diffuse_color.a);

    let physical_color = impostor_phong(world_pos, world_normal, view_dir, diffuse_color.rgb);

    var out: FragmentOutput;
    out.color = vec4<f32>(physical_color, diffuse_color.a);
    out.depth = impostor_depth(world_pos);
    $$ if write_pick
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(billboard_decode_glyph_index(varyings.glyph_index_parts), 26) +
        pick_pack(0u, 18)
    );
    $$ endif
    return out;
}
