{$ include 'pygfx.std.wgsl' $}
{$ include 'fury.utils.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    // Generate quad vertices for billboard
    // Each billboard uses 6 vertices (2 triangles to form a quad)
    let billboard_index = i32(in.index) / 6;
    let vertex_in_quad = i32(in.index) % 6;

    // Quad vertices in local space (counter-clockwise winding)
    var local_pos: vec2<f32>;
    switch vertex_in_quad {
        case 0: { local_pos = vec2<f32>(-0.5, -0.5); } // bottom left
        case 1: { local_pos = vec2<f32>(0.5, -0.5); }  // bottom right
        case 2: { local_pos = vec2<f32>(-0.5, 0.5); }  // top left
        case 3: { local_pos = vec2<f32>(0.5, -0.5); }  // bottom right
        case 4: { local_pos = vec2<f32>(0.5, 0.5); }   // top right
        default: { local_pos = vec2<f32>(-0.5, 0.5); } // top left
    }

    // Load billboard center position from storage buffer. Each center is
    // duplicated 6 times in the geometry buffer, so pick the first occurrence.
    let raw_center = load_s_positions(billboard_index * 6);
    let world_center = u_wobject.world_transform * vec4<f32>(raw_center.xyz, 1.0);

    // Get camera right and up vectors in world space
    // Extract right and up vectors from inverse camera transform
    let cam_right = vec3<f32>(u_stdinfo.cam_transform_inv[0].xyz);
    let cam_up = vec3<f32>(u_stdinfo.cam_transform_inv[1].xyz);

    // Fetch per-billboard sizes encoded in normals buffer (duplicated per vertex)
    let raw_size = load_s_normals(billboard_index * 6);
    let size = raw_size.xy;

    // Calculate billboard vertex position in world space
    let billboard_offset = local_pos.x * cam_right * size.x + local_pos.y * cam_up * size.y;
    let world_pos = world_center.xyz + billboard_offset;

    // Transform to clip space
    let clip_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);

    var varyings: Varyings;
    varyings.position = vec4<f32>(clip_pos);
    varyings.world_pos = vec3<f32>(world_pos);

    // Load color if available - colors are duplicated 6x like positions
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

    return varyings;
}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    // Use the color from vertex shader
    let color = varyings.color;
    let physical_color = srgb2physical(color.rgb);
    let opacity = color.a * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    var out: FragmentOutput;
    out.color = out_color;

    return out;
}
