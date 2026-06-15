{{ bindings_code }}

const N_NODES = i32({{ n_nodes }});

fn get_pos(i: i32) -> vec3<f32> {
    let idx = i * 3;
    return vec3<f32>(s_positions[idx], s_positions[idx+1], s_positions[idx+2]);
}

fn set_pos(i: i32, p: vec3<f32>) {
    let idx = i * 3;
    s_positions[idx] = p.x;
    s_positions[idx+1] = p.y;
    s_positions[idx+2] = p.z;
}

@compute @workgroup_size{{ workgroup_size }}
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = i32(global_id.x);
    if (i >= N_NODES) { return; }

    let pos_i = get_pos(i);
    var force = vec3<f32>(0.0);

    let k = u_material.k;
    let k_sq = k * k;

    // 1. Repulsion
    for (var j: i32 = 0; j < N_NODES; j++) {
        if (i == j) { continue; }

        let pos_j = get_pos(j);
        let delta = pos_i - pos_j;
        let dist_sq = dot(delta, delta);
        let dist = sqrt(dist_sq);

        if (dist > 0.001) {
            force += (delta / dist_sq) * k_sq * u_material.repulsion_strength;
        } else {
            force += vec3<f32>(1.0, 0.0, 0.0) * k_sq;
        }
    }

    // 2. Attraction
    let start = s_offsets[i];
    let count = s_counts[i];

    for (var idx: i32 = 0; idx < count; idx++) {
        let neighbor_idx = s_adj[start + idx];
        let pos_j = get_pos(neighbor_idx);
        let delta = pos_i - pos_j;
        let dist_sq = dot(delta, delta);
        let dist = sqrt(dist_sq);

        force -= delta * dist / k;
    }

    // 3. Integration
    var vel = s_velocities[i].xyz;
    vel = (vel + force * 0.01) * u_material.damping;

    let current_speed = length(vel);
    let max_speed = k * 2.0;
    if (current_speed > max_speed) {
        vel = normalize(vel) * max_speed;
    }

    let new_pos = pos_i + vel * u_material.speed;

    set_pos(i, new_pos);
    s_velocities[i] = vec4<f32>(vel, 0.0);
}
