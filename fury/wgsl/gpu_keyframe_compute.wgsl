{{ bindings_code }}

const N_ELEMENTS: u32 = {{ n_elements }}u;
const N_KEYS: u32 = {{ n_keys }}u;
const REPEAT: u32 = {{ repeat }}u;
const STRIDE: u32 = {{ stride }}u;
const VALUE_COMPONENTS: u32 = {{ value_components }}u;
const ROW_THREADS: u32 = {{ row_threads }}u;

fn key_value(k: u32, i: u32, c: u32) -> f32 {
    return s_values[(k * N_ELEMENTS + i) * VALUE_COMPONENTS + c];
}

$$ if interpolation == "cubic"
fn key_slope(k: u32, i: u32, c: u32) -> f32 {
    return s_slopes[(k * N_ELEMENTS + i) * VALUE_COMPONENTS + c];
}
$$ endif

fn interpolate(k0: u32, k1: u32, i: u32, c: u32, s: f32, h: f32) -> f32 {
    let v0 = key_value(k0, i, c);
    let v1 = key_value(k1, i, c);
$$ if interpolation == "cubic"
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    return h00 * v0 + h10 * h * key_slope(k0, i, c) + h01 * v1 + h11 * h * key_slope(k1, i, c);
$$ else
    return mix(v0, v1, s);
$$ endif
}

@compute @workgroup_size{{ workgroup_size }}
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * ROW_THREADS;
    if (i >= N_ELEMENTS) { return; }

    let t = clamp(u_material.time, s_times[0], s_times[N_KEYS - 1u]);
    var lo = 0u;
    var hi = N_KEYS - 1u;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (s_times[mid] <= t) { lo = mid; } else { hi = mid; }
    }
    let k1 = min(lo + 1u, N_KEYS - 1u);
    let h = s_times[k1] - s_times[lo];
    let s = select(0.0, (t - s_times[lo]) / h, h > 0.0);

$$ for ch in channels
    {
        let v = interpolate(lo, k1, i, {{ 0 if value_components == 1 else loop.index0 }}u, s, h) * u_material.scale;
        for (var r = 0u; r < REPEAT; r++) {
            let row = i * REPEAT + r;
$$ if target_vec
            s_target[row][{{ ch }}] = v;
$$ else
            s_target[row * STRIDE + {{ ch }}u] = v;
$$ endif
        }
    }
$$ endfor
}
