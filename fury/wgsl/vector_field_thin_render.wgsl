{$ include 'pygfx.std.wgsl' $}
{$ include 'fury.utils.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) index : u32,
};

const DATA_SHAPE = vec3<i32>{{ data_shape }};
const NUM_VECTORS = i32({{ num_vectors }});

@vertex
fn vs_main(in: VertexInput) -> Varyings {

    let i0 = i32(in.index);

    let raw_pos = load_s_positions(i0);
    var wpos = u_wobject.world_transform * vec4<f32>(raw_pos.xyz, 1.0);
    let center = flatten_to_3d(i0 / (NUM_VECTORS * 2), DATA_SHAPE);
    var w_center = u_wobject.world_transform * vec4<f32>(vec3<f32>(center), 1.0);
    let cross_section = u_material.cross_section.xyz;
    let visibility = u_material.visibility.xyz;
    let diff = wpos.xyz - w_center.xyz;

    if (!all(visibility == vec3<i32>(-1))) {
        w_center = vec4<f32>(
            snap_cross_section_center(
                w_center.xyz, cross_section, u_wobject.world_transform
            ),
            w_center.w
        );
        wpos = vec4<f32>(w_center.xyz + diff, wpos.w);
    }

    let npos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos;

    var varyings: Varyings;
    varyings.position = vec4<f32>(npos);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(npos));

    let color = load_s_colors(i0);

    varyings.color = vec4<f32>(color, 1.0);

    varyings.center = vec3<f32>(w_center.xyz);
    varyings.cross_section = vec3<f32>(cross_section);
    varyings.visibility = vec3<i32>(visibility);

    return varyings;
}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let cross_section = varyings.cross_section;
    let visibility = varyings.visibility;
    if !all(visibility == vec3<i32>(-1)) && !visible_cross_section(varyings.center, cross_section, visibility) {
        discard;
    }

    let color = varyings.color;
    let physical_color = srgb2physical(color.rgb);
    let opacity = color.a * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    var out: FragmentOutput;
    out.color = out_color;

    return out;
}
