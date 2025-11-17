{{ bindings_code }}

const MAX_LINE_LENGTH = u32({{ max_line_length }});
const TUBE_SIDES = u32({{ tube_sides }});
const N_LINES = u32({{ n_lines }});
const TUBE_RADIUS = f32({{ tube_radius }});
const COLOR_CHANNELS = u32({{ color_channels }});
const PI = 3.14159265359;

const USE_END_CAPS = {{ end_caps }}u;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_v = length(v);
    if (len_v > 1e-6) {
        return v / len_v;
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn get_line_point(line_idx: u32, point_idx: u32, line_length: u32) -> vec3<f32> {
    if (line_length == 0u) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let last_idx = min(line_length - 1u, MAX_LINE_LENGTH - 1u);
    let clamped_idx = min(point_idx, last_idx);
    let base_idx = (line_idx * MAX_LINE_LENGTH + clamped_idx) * 3u;

    if (base_idx + 2u >= arrayLength(&s_line_data)) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    return vec3<f32>(
        s_line_data[base_idx],
        s_line_data[base_idx + 1u],
        s_line_data[base_idx + 2u],
    );
}

fn rodrigues_rotate(v: vec3<f32>, axis: vec3<f32>, sin_angle: f32, cos_angle: f32) -> vec3<f32> {
    return v * cos_angle + cross(axis, v) * sin_angle + axis * dot(axis, v) * (1.0 - cos_angle);
}

fn get_tangent(line_idx: u32, point_idx: u32, line_length: u32) -> vec3<f32> {
    if (line_length <= 1u) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }

    let point = get_line_point(line_idx, point_idx, line_length);
    var tangent: vec3<f32>;
    if (point_idx == 0u) {
        tangent = get_line_point(line_idx, 1u, line_length) - point;
    } else if (point_idx == line_length - 1u) {
        tangent = point - get_line_point(line_idx, point_idx - 1u, line_length);
    } else {
        tangent = get_line_point(line_idx, point_idx + 1u, line_length)
            - get_line_point(line_idx, point_idx - 1u, line_length);
    }
    return safe_normalize(tangent);
}

fn write_vertex_color(vertex_idx: u32, color: vec4<f32>, color_len: u32) {
    if (COLOR_CHANNELS == 0u || color_len == 0u) {
        return;
    }

    let base = vertex_idx * COLOR_CHANNELS;
    if (base >= color_len) {
        return;
    }

    s_vertex_colors[base] = color.x;
    if (COLOR_CHANNELS > 1u && base + 1u < color_len) {
        s_vertex_colors[base + 1u] = color.y;
    }
    if (COLOR_CHANNELS > 2u && base + 2u < color_len) {
        s_vertex_colors[base + 2u] = color.z;
    }
    if (COLOR_CHANNELS > 3u && base + 3u < color_len) {
        s_vertex_colors[base + 3u] = color.w;
    }
}

@compute @workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let line_idx = gid.x;
    if (line_idx >= N_LINES) {
        return;
    }

    let line_length = s_line_lengths[line_idx];
    if (line_length < 2u) {
        return;
    }

    let base_vertex_idx = s_vertex_offsets[line_idx];
    let base_triangle_idx = s_triangle_offsets[line_idx];

    let pos_len = arrayLength(&s_vertex_positions);
    let norm_len = arrayLength(&s_vertex_normals);
    let color_len = arrayLength(&s_vertex_colors);
    if (pos_len < 3u || norm_len < 3u) {
        return;
    }

$$ if color_channels == 4
    let line_color = load_s_line_colors(i32(line_idx));
$$ elif color_channels == 3
    let line_color_rgb = load_s_line_colors(i32(line_idx));
    let line_color = vec4<f32>(line_color_rgb, 1.0);
$$ elif color_channels == 2
    let line_color_rg = load_s_line_colors(i32(line_idx));
    let line_color = vec4<f32>(line_color_rg, 0.0, 1.0);
$$ elif color_channels == 1
    let line_color_scalar = load_s_line_colors(i32(line_idx));
    let line_color = vec4<f32>(line_color_scalar, line_color_scalar, line_color_scalar, 1.0);
$$ else
    let line_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
$$ endif
    let vertices_per_point = TUBE_SIDES;

    var tangents_arr: array<vec3<f32>, MAX_LINE_LENGTH>;
    for (var i = 0u; i < line_length; i++) {
        tangents_arr[i] = get_tangent(line_idx, i, line_length);
    }

    var normals_arr: array<vec3<f32>, MAX_LINE_LENGTH>;
    var binormals_arr: array<vec3<f32>, MAX_LINE_LENGTH>;

    var base_tangent = tangents_arr[0u];
    if (length(base_tangent) < 1e-6) {
        base_tangent = vec3<f32>(0.0, 0.0, 1.0);
        tangents_arr[0u] = base_tangent;
    }
    var reference = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(dot(base_tangent, reference)) > 0.99) {
        reference = vec3<f32>(1.0, 0.0, 0.0);
    }
    var base_binormal = safe_normalize(cross(base_tangent, reference));
    if (length(base_binormal) < 1e-6) {
        base_binormal = vec3<f32>(0.0, 1.0, 0.0);
    }
    var base_normal = safe_normalize(cross(base_binormal, base_tangent));
    normals_arr[0u] = base_normal;
    binormals_arr[0u] = base_binormal;

    for (var i = 1u; i < line_length; i++) {
        let prev_tangent = tangents_arr[i - 1u];
        let curr_tangent = tangents_arr[i];
        var axis = cross(prev_tangent, curr_tangent);
        let sin_angle = length(axis);
        let cos_angle = clamp(dot(prev_tangent, curr_tangent), -1.0, 1.0);

        var normal = normals_arr[i - 1u];
        var binormal = binormals_arr[i - 1u];
        if (sin_angle > 1e-6) {
            let axis_norm = axis / sin_angle;
            normal = rodrigues_rotate(normal, axis_norm, sin_angle, cos_angle);
            binormal = rodrigues_rotate(binormal, axis_norm, sin_angle, cos_angle);
        }

        normals_arr[i] = safe_normalize(normal);
        binormals_arr[i] = safe_normalize(binormal);
    }

    for (var i = 0u; i < line_length; i++) {
        let point = get_line_point(line_idx, i, line_length);
        let tangent = tangents_arr[i];
        let normal = normals_arr[i];
        let binormal = binormals_arr[i];

        for (var j = 0u; j < TUBE_SIDES; j++) {
            let angle = 2.0 * PI * f32(j) / f32(TUBE_SIDES);
            let cos_a = cos(angle);
            let sin_a = sin(angle);

            let offset = normal * cos_a + binormal * sin_a;
            let vertex_pos = point + offset * TUBE_RADIUS;
            let vertex_normal = safe_normalize(offset);

            let vertex_idx = base_vertex_idx + i * vertices_per_point + j;
            let pos_base = vertex_idx * 3u;
            if (pos_base + 2u < pos_len
                && pos_base + 2u < norm_len) {
                s_vertex_positions[pos_base] = vertex_pos.x;
                s_vertex_positions[pos_base + 1u] = vertex_pos.y;
                s_vertex_positions[pos_base + 2u] = vertex_pos.z;

                s_vertex_normals[pos_base] = vertex_normal.x;
                s_vertex_normals[pos_base + 1u] = vertex_normal.y;
                s_vertex_normals[pos_base + 2u] = vertex_normal.z;
                write_vertex_color(vertex_idx, line_color, color_len);
            }
        }

        if (i < line_length - 1u) {
            for (var j = 0u; j < TUBE_SIDES; j++) {
                let next_j = (j + 1u) % TUBE_SIDES;

                let v0 = base_vertex_idx + i * vertices_per_point + j;
                let v1 = base_vertex_idx + i * vertices_per_point + next_j;
                let v2 = base_vertex_idx + (i + 1u) * vertices_per_point + j;
                let v3 = base_vertex_idx + (i + 1u) * vertices_per_point + next_j;

                let tri_base = base_triangle_idx + (i * TUBE_SIDES + j) * 2u;
                let idx_base = tri_base * 3u;
                if (idx_base + 5u < arrayLength(&s_indices)) {
                    s_indices[idx_base] = v0;
                    s_indices[idx_base + 1u] = v1;
                    s_indices[idx_base + 2u] = v3;
                    s_indices[idx_base + 3u] = v0;
                    s_indices[idx_base + 4u] = v3;
                    s_indices[idx_base + 5u] = v2;
                }
            }
        }
    }

    if (USE_END_CAPS == 1u && line_length >= 2u) {
        let ring_vertex_count = line_length * TUBE_SIDES;
        let start_center_idx = base_vertex_idx + ring_vertex_count;
        let end_center_idx = start_center_idx + 1u;

        let start_point = get_line_point(line_idx, 0u, line_length);
        let end_point = get_line_point(line_idx, line_length - 1u, line_length);
        let start_normal = safe_normalize(-tangents_arr[0u]);
        let end_normal = safe_normalize(tangents_arr[line_length - 1u]);

        if (start_center_idx * 3u + 2u < pos_len && start_center_idx * 3u + 2u < norm_len) {
            let base = start_center_idx * 3u;
            s_vertex_positions[base] = start_point.x;
            s_vertex_positions[base + 1u] = start_point.y;
            s_vertex_positions[base + 2u] = start_point.z;
            s_vertex_normals[base] = start_normal.x;
            s_vertex_normals[base + 1u] = start_normal.y;
            s_vertex_normals[base + 2u] = start_normal.z;
        }
        write_vertex_color(start_center_idx, line_color, color_len);

        if (end_center_idx * 3u + 2u < pos_len && end_center_idx * 3u + 2u < norm_len) {
            let base = end_center_idx * 3u;
            s_vertex_positions[base] = end_point.x;
            s_vertex_positions[base + 1u] = end_point.y;
            s_vertex_positions[base + 2u] = end_point.z;
            s_vertex_normals[base] = end_normal.x;
            s_vertex_normals[base + 1u] = end_normal.y;
            s_vertex_normals[base + 2u] = end_normal.z;
        }
        write_vertex_color(end_center_idx, line_color, color_len);

        let side_triangles = (line_length - 1u) * TUBE_SIDES * 2u;
        var tri_idx = base_triangle_idx + side_triangles;

        // Start cap
        for (var j = 0u; j < TUBE_SIDES; j++) {
            let next_j = (j + 1u) % TUBE_SIDES;
            let idx_base = tri_idx * 3u;
            s_indices[idx_base] = base_vertex_idx + next_j;
            s_indices[idx_base + 1u] = base_vertex_idx + j;
            s_indices[idx_base + 2u] = start_center_idx;
            tri_idx += 1u;
        }

        // End cap
        let end_ring_base = base_vertex_idx + (line_length - 1u) * TUBE_SIDES;
        for (var j = 0u; j < TUBE_SIDES; j++) {
            let next_j = (j + 1u) % TUBE_SIDES;
            let idx_base = tri_idx * 3u;
            s_indices[idx_base] = end_ring_base + j;
            s_indices[idx_base + 1u] = end_ring_base + next_j;
            s_indices[idx_base + 2u] = end_center_idx;
            tri_idx += 1u;
        }
    }
}
