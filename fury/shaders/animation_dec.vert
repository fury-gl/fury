

in vec4 scalarColor;
uniform float time;

out float t;
out vec4 vertexColorVSOutput;


mat3 xyz_to_rgb_mat = mat3(3.24048134, -1.53715152, -0.49853633,
  -0.96925495, 1.87599, 0.04155593,
  0.05564664, -0.20404134, 1.05731107);

// Interpolation methods
const int STEP = 0;
const int LINEAR = 1;
const int BEZIER = 2;
const int HSV = 3;
const int XYZ = 4;
const int Slerp = 5;

struct Keyframe {
  float t;
  vec3 value;
  vec3 inCp;
  vec3 outCp;
};

struct Keyframes {
  Keyframe[6] keyframes;
  int method;
  int count;
};


uniform Keyframes position_k;
uniform Keyframes scale_k;
uniform Keyframes color_k;
uniform Keyframes opacity_k;


Keyframe get_next_keyframe(Keyframes keyframes, float t, bool first) {
  int start = 0;
  if (!first) start++;
  for (int i = start; i < keyframes.count; i++)
    if (keyframes.keyframes[i].t > t) return keyframes.keyframes[i];
  return keyframes.keyframes[keyframes.count - 1];
}

Keyframe get_previous_keyframe(Keyframes keyframes, float t, bool last) {
  int start = keyframes.count - 1;
  if (!last) start--;
  for (int i = start; i >= 0; i--)
    if (keyframes.keyframes[i].t <= t) return keyframes.keyframes[i];
  return keyframes.keyframes[keyframes.count - 1];
}

bool has_one_keyframe(Keyframes k) {
  if (k.count == 1)
    return true;
  return false;
}

bool is_interpolatable(Keyframes k) {
  return bool(k.count);
}

float get_time_tau_clamped(float t, float t0, float t1){
    return clamp((t - t0) / (t1 - t0), 0, 1);
}

float get_time_tau(float t, float t0, float t1){
    return (t - t0) / (t1 - t0);
}

vec3 lerp(Keyframes k, float t) {
  if (has_one_keyframe(k)) return k.keyframes[0].value;
  Keyframe k0 = k.keyframes[0];
  Keyframe k1 = k.keyframes[1];
  float dt = get_time_tau_clamped(t, k0.t, k1.t);
  return mix(k0.value, k1.value, dt);
}

vec3 cubic_bezier(Keyframes k, float t) {
  if (has_one_keyframe(k)) return k.keyframes[0].value;
  Keyframe k0 = get_previous_keyframe(k, t, false);
  Keyframe k1 = get_next_keyframe(k, t, false);
  float dt = get_time_tau_clamped(t, k0.t, k1.t);
  vec3 E = mix(k0.value, k0.outCp, dt);
  vec3 F = mix(k0.outCp, k1.inCp, dt);
  vec3 G = mix(k1.inCp, k1.value, dt);

  vec3 H = mix(E, F, dt);
  vec3 I = mix(F, G, dt);

  vec3 P = mix(H, I, dt);

  return P;
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float clip(float x) {
  if (x > 1)
    return 1;
  else if (x < 0)
    return 0;
  else return x;
}
vec3 xyz2rgb(vec3 c) {
  c = c * xyz_to_rgb_mat;
  float po = 1 / 2.4;
  if (c.x > 0.0031308) c.x = 1.055 * pow(c.x, po) - 0.055;
  else c.y *= 12.92;
  if (c.y > 0.0031308) c.y = 1.055 * pow(c.y, po) - 0.055;
  else c.y *= 12.92;
  if (c.z > 0.0031308) c.z = 1.055 * pow(c.z, po) - 0.055;
  else c.z *= 12.92;

  c.x = clip(c.x);
  c.y = clip(c.y);
  c.z = clip(c.z);

  return c;
}

vec3 lab2xyz(vec3 col) {
  float l = col.x;
  float a = col.y;
  float b = col.z;
  col.y = (l + 16.) / 116.;
  col.x = (a / 500.) + col.y;
  col.z = col.y - (b / 200.);
  return col;
}

vec3 interp(Keyframes k, float t) {
  if (k.method == LINEAR) return lerp(k, t);
  else if (k.method == BEZIER) return cubic_bezier(k, t);
  else if (k.method == HSV) return hsv2rgb(lerp(k, t));
  else if (k.method == XYZ) return xyz2rgb(lab2xyz(lerp(k, t)));
  else if (k.method == STEP) return k.keyframes[0].value;
}

mat4 transformation(vec3 position, vec3 scale) {
  return mat4(
    vec4(scale.x, 0.0, 0.0, 0.0),
    vec4(0.0, scale.y, 0.0, 0.0),
    vec4(0.0, 0.0, scale.z, 0.0),
    vec4(position, 1.0));
}