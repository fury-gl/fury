{{ bindings_code }}
{$ include 'fury.utils.wgsl' $}

// @group(1) @binding(0)
// var<storage, read_write> s_scaled_vertice: array<f32>;

const DATA_SHAPE = vec3<i32>{{ data_shape }};
const NUM_COEFFS = i32({{ n_coeffs }});
const VERTICES_PER_GLYPH = i32({{ vertices_per_glyph }});
const FACES_PER_GLYPH = i32({{ faces_per_glyph }});

fn calculate_deformation(first_coeff_id: i32, sphere_vertex_id: i32) -> f32 {
    var radii: f32 = 0.0;
    for (var i: i32 = 0; i < NUM_COEFFS; i++) {
        radii += s_coeffs[first_coeff_id + i] * s_sf_func[sphere_vertex_id + i];
    }
    radii = max(radii, 0.0);
    radii = min(radii, 1.0);
    return radii;
}

fn calculate_position_from_deformation(vertex_id: i32, radii: f32, scale: f32, center: vec3<i32>) -> vec3<f32> {
    let x = s_sphere[vertex_id * 3] * radii * scale + f32(center.x);
    let y = s_sphere[vertex_id * 3 + 1] * radii * scale + f32(center.y);
    let z = s_sphere[vertex_id * 3 + 2] * radii * scale + f32(center.z);
    return vec3<f32>(x, y, z);
}

fn calculate_relative_position(position: vec3<f32>, center: vec3<i32>) -> vec3<f32> {
    let x = position.x - f32(center.x);
    let y = position.y - f32(center.y);
    let z = position.z - f32(center.z);
    return vec3<f32>(x, y, z);
}

fn update_normals(first_normal_id: i32, center: vec3<i32>) {
    var ab = vec3<f32>(0.0);
    var ac = vec3<f32>(0.0);
    var normal = vec3<f32>(0.0);

    var a = vec3<f32>(0.0);
    var b = vec3<f32>(0.0);
    var c = vec3<f32>(0.0);

    let n_indices = FACES_PER_GLYPH * 3;

    for (var i: i32 = 0; i < n_indices; i += 3) {

        a = load_s_scaled_vertice(s_indices[i] + first_normal_id);
        b = load_s_scaled_vertice(s_indices[i + 1] + first_normal_id);
        c = load_s_scaled_vertice(s_indices[i + 2] + first_normal_id);

        ab = b - a;
        ac = c - a;

        if (length(ab) > 1e-4 && length(ac) > 1e-4) {
            ab = normalize(ab);
            ac = normalize(ac);
            if (abs(dot(ab, ac)) < 0.999) {
                normal = normalize(cross(ab, ac));
                s_normals[(s_indices[i] + first_normal_id) * 3] += normal.x;
                s_normals[(s_indices[i] + first_normal_id) * 3 + 1] += normal.y;
                s_normals[(s_indices[i] + first_normal_id) * 3 + 2] += normal.z;
                s_normals[(s_indices[i + 1] + first_normal_id) * 3] += normal.x;
                s_normals[(s_indices[i + 1] + first_normal_id) * 3 + 1] += normal.y;
                s_normals[(s_indices[i + 1] + first_normal_id) * 3 + 2] += normal.z;
                s_normals[(s_indices[i + 2] + first_normal_id) * 3] += normal.x;
                s_normals[(s_indices[i + 2] + first_normal_id) * 3 + 1] += normal.y;
                s_normals[(s_indices[i + 2] + first_normal_id) * 3 + 2] += normal.z;
            }
        }
    }
}


@compute @workgroup_size{{ workgroup_size }}
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let voxel_id = flatten_from_3d(vec3<i32>(global_id), vec3<i32>{{ workgroup_size }});

    if (voxel_id >= DATA_SHAPE.x * DATA_SHAPE.y * DATA_SHAPE.z) {
        return;
    }

    let valid_voxel: bool = s_coeffs[voxel_id * NUM_COEFFS] > 1e-4;

    let center = flatten_to_3d(voxel_id, DATA_SHAPE);
    let first_vertex_id = voxel_id * VERTICES_PER_GLYPH;

    for (var i: i32 = 0; i < VERTICES_PER_GLYPH; i++) {
        var radii: f32 = 0.0;

        if valid_voxel {
            radii = calculate_deformation(voxel_id * NUM_COEFFS, i * NUM_COEFFS);
        }


        let current_vertex = (first_vertex_id + i) * 3;
        let position = calculate_position_from_deformation(i, radii, 3, center);

        s_positions[current_vertex] = position.x;
        s_positions[current_vertex + 1] = position.y;
        s_positions[current_vertex + 2] = position.z;

        let scaled_vertice = vec3<f32>(radii) * load_s_sphere(i);
        s_scaled_vertice[current_vertex] = scaled_vertice.x;
        s_scaled_vertice[current_vertex + 1] = scaled_vertice.y;
        s_scaled_vertice[current_vertex + 2] = scaled_vertice.z;


        let color = scaled_color(scaled_vertice);
        s_colors[current_vertex] = color.x;
        s_colors[current_vertex + 1] = color.y;
        s_colors[current_vertex + 2] = color.z;
    }

    if (valid_voxel) {
        update_normals(first_vertex_id, center);
    }

}
