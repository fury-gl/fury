{{ bindings_code }}

const N_LINES = u32({{ n_lines }});
const OUT_CAPACITY = u32({{ out_capacity }});
const COLOR_CHANNELS = u32({{ color_channels }});
const ROI_ENABLED = {{ roi_enabled }}u;
const ROI_DIM = vec3<i32>({{ roi_dim_x }}, {{ roi_dim_y }}, {{ roi_dim_z }});
const ROI_ORIGIN = vec3<f32>({{ roi_origin_x }}, {{ roi_origin_y }}, {{ roi_origin_z }});

fn roi_enabled() -> bool {
    return ROI_ENABLED == 1u;
}

fn world_to_grid(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p - ROI_ORIGIN + 0.5));
}

fn mask_idx_from_grid(g: vec3<i32>) -> i32 {
    if (any(g < vec3<i32>(0)) || any(g >= ROI_DIM)) {
        return -1;
    }
    return (g.z * ROI_DIM.y + g.y) * ROI_DIM.x + g.x;
}

fn mask_value(g: vec3<i32>) -> bool {
    let idx = mask_idx_from_grid(g);
    if (idx < 0) {
        return false;
    }
    return s_roi_mask[u32(idx)] > 0u;
}

fn point_in_roi(p: vec3<f32>) -> bool {
    if (any(ROI_DIM <= vec3<i32>(0))) {
        return false;
    }
    return mask_value(world_to_grid(p));
}

fn load_position(flat_index: u32) -> vec3<f32> {
    let base = flat_index * 3u;
    if (base + 2u >= arrayLength(&s_line_positions)) {
        return vec3<f32>(0.0);
    }
    return vec3<f32>(
        s_line_positions[base],
        s_line_positions[base + 1u],
        s_line_positions[base + 2u],
    );
}

fn load_color(flat_index: u32) -> vec4<f32> {
    if (COLOR_CHANNELS == 0u) {
        return vec4<f32>(1.0);
    }
    let base = flat_index * COLOR_CHANNELS;
    if (base >= arrayLength(&s_line_colors)) {
        return vec4<f32>(1.0);
    }
    var color = vec4<f32>(1.0);
    color.x = s_line_colors[base];
    if (COLOR_CHANNELS > 1u && base + 1u < arrayLength(&s_line_colors)) {
        color.y = s_line_colors[base + 1u];
    }
    if (COLOR_CHANNELS > 2u && base + 2u < arrayLength(&s_line_colors)) {
        color.z = s_line_colors[base + 2u];
    }
    if (COLOR_CHANNELS > 3u && base + 3u < arrayLength(&s_line_colors)) {
        color.w = s_line_colors[base + 3u];
    }
    return color;
}

fn write_position(out_index: u32, value: vec3<f32>, capacity: u32) {
    let base = out_index * 3u;
    if (base + 2u >= capacity) {
        return;
    }
    s_out_positions[base] = value.x;
    s_out_positions[base + 1u] = value.y;
    s_out_positions[base + 2u] = value.z;
}

fn write_color(out_index: u32, value: vec4<f32>, capacity: u32) {
    if (COLOR_CHANNELS == 0u) {
        return;
    }
    let base = out_index * COLOR_CHANNELS;
    if (base >= capacity) {
        return;
    }
    s_out_colors[base] = value.x;
    if (COLOR_CHANNELS > 1u && base + 1u < capacity) {
        s_out_colors[base + 1u] = value.y;
    }
    if (COLOR_CHANNELS > 2u && base + 2u < capacity) {
        s_out_colors[base + 2u] = value.z;
    }
    if (COLOR_CHANNELS > 3u && base + 3u < capacity) {
        s_out_colors[base + 3u] = value.w;
    }
}

fn line_hits_roi(line_offset: u32, line_length: u32) -> bool {
    if (!roi_enabled()) {
        return true;
    }
    for (var i: u32 = 0u; i < line_length; i += 1u) {
        let pos = load_position(line_offset + i);
        if (point_in_roi(pos)) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let line_idx = gid.x;
    if (line_idx >= N_LINES) {
        return;
    }

    let line_length = s_line_lengths[line_idx];
    if (line_length == 0u) {
        return;
    }
    let line_offset = s_line_offsets[line_idx];

    let pos_capacity = arrayLength(&s_out_positions);
    let color_capacity = arrayLength(&s_out_colors);
    let nanf = bitcast<f32>(0x7fc00000u);
    let nan3 = vec3<f32>(nanf);
    let nan4 = vec4<f32>(nanf);
    let keep_line = line_hits_roi(line_offset, line_length);

    for (var i: u32 = 0u; i < line_length; i += 1u) {
        let out_idx = line_offset + i;
        if (out_idx >= OUT_CAPACITY) {
            break;
        }
        let pos = load_position(out_idx);
        let col = load_color(out_idx);
        if (keep_line) {
            write_position(out_idx, pos, pos_capacity);
            write_color(out_idx, col, color_capacity);
        } else {
            write_position(out_idx, nan3, pos_capacity);
            write_color(out_idx, nan4, color_capacity);
        }
    }
}
