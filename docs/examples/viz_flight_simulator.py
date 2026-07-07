import numpy as np
from fury import window, actor, ui
from fury.window import EventType

V_FWD = np.array([0.0, 0.0, 1.0])
V_UP = np.array([0.0, 1.0, 0.0])
V_R = np.array([1.0, 0.0, 0.0])
V_ZERO = np.array([0.0, 0.0, 0.0])

state = {
    "is_playing": True,
    "score": 0.0,
    "speed": 0.0,
    "max_speed": 180.0,
    "takeoff_speed": 60.0,
    "player_pos": np.array([0.0, 2.0, 0.0]),
    "player_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    "cam_pos": np.array([0.0, 50.0, -50.0]),
    "cam_up": np.array([0.0, 1.0, 0.0]),
    "pitch_rate": 0.0,
    "roll_rate": 0.0,
    "yaw_rate": 0.0,
    "keys": set(),
    "tick_count": 0,
}

SPAWN_DIST = 1500.0
DESPAWN_DIST = 300.0
SPAWN_DIST_SQ = (SPAWN_DIST * 2.0) ** 2
CLOUD_COUNT = 70
TREE_COUNT = 120
MOUNTAIN_COUNT = 30


def axis_angle_to_quat(axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    s = np.sin(angle_rad / 2.0)
    c = np.cos(angle_rad / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])


def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def rotate_vector(quat, vec):
    q_vec = quat[:3]
    q_w = quat[3]
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return vec + 2.0 * (q_w * uv + uuv)


def get_surface_height(pos):
    return 0.55 if abs(pos[0]) <= 12.0 else 0.0


scene = window.Scene()
scene.background = (0.45, 0.65, 0.95)

ground = actor.box(
    centers=np.array([V_ZERO]), colors=(0.28, 0.52, 0.28), scales=(2000, 1, 2000)
)
scene.add(ground)

sun = actor.sphere(centers=np.array([V_ZERO]), colors=(1.0, 0.95, 0.85), radii=35.0)
scene.add(sun)


class RunwayManager:
    def __init__(self):
        self.runway = actor.box(
            centers=np.array([V_ZERO]),
            colors=(0.15, 0.15, 0.15),
            scales=(24, 1.1, 2000),
        )
        scene.add(self.runway)

        self.dashes = []
        for z in range(-1000, 2000, 40):
            dash = actor.box(
                centers=np.array([V_ZERO]),
                colors=(0.9, 0.9, 0.9),
                scales=(0.6, 0.02, 12.0),
            )
            dash.local.position = [0.0, 0.56, float(z)]
            scene.add(dash)
            self.dashes.append(dash)

        self.lights = []
        c = (0.95, 0.8, 0.2)
        for z in range(-1000, 2000, 50):
            l1 = actor.sphere(centers=np.array([V_ZERO]), colors=c, radii=0.25)
            l1.local.position = [-12.0, 0.58, float(z)]
            l2 = actor.sphere(centers=np.array([V_ZERO]), colors=c, radii=0.25)
            l2.local.position = [12.0, 0.58, float(z)]
            scene.add(l1)
            scene.add(l2)
            self.lights.extend([l1, l2])

    def update(self, p_pos):
        pz = p_pos[2]
        self.runway.local.position = [0.0, 0.0, pz]

        for dash in self.dashes:
            dz = dash.local.position[2]
            if dz < pz - 1000:
                dash.local.position = [0.0, 0.56, dz + 3000]
            elif dz > pz + 2000:
                dash.local.position = [0.0, 0.56, dz - 3000]

        for light in self.lights:
            lz = light.local.position[2]
            if lz < pz - 1000:
                light.local.position = [light.local.position[0], 0.58, lz + 3000]
            elif lz > pz + 2000:
                light.local.position = [light.local.position[0], 0.58, lz - 3000]


runway_mgr = RunwayManager()


class Aircraft:
    def __init__(self):
        c0 = np.array([V_ZERO])
        self.fuselage = actor.cylinder(
            centers=c0,
            directions=np.array([V_FWD]),
            colors=(0.9, 0.9, 0.9),
            height=3.2,
            radii=0.45,
        )
        self.nose = actor.cone(
            centers=c0,
            directions=np.array([V_FWD]),
            colors=(0.85, 0.2, 0.2),
            height=1.0,
            radii=0.45,
        )
        self.canopy = actor.ellipsoid(
            centers=c0, lengths=(0.4, 0.35, 1.1), colors=(0.3, 0.8, 0.95), opacity=0.65
        )
        self.wings = actor.box(
            centers=c0, colors=(0.9, 0.9, 0.9), scales=(6.8, 0.08, 1.2)
        )
        self.wing_stripe_l = actor.box(
            centers=c0, colors=(0.85, 0.2, 0.2), scales=(0.4, 0.1, 1.22)
        )
        self.wing_stripe_r = actor.box(
            centers=c0, colors=(0.85, 0.2, 0.2), scales=(0.4, 0.1, 1.22)
        )
        self.wing_light_l = actor.sphere(centers=c0, colors=(0.9, 0.1, 0.1), radii=0.08)
        self.wing_light_r = actor.sphere(centers=c0, colors=(0.1, 0.9, 0.1), radii=0.08)
        self.tail_fin = actor.box(
            centers=c0, colors=(0.85, 0.2, 0.2), scales=(0.1, 1.2, 0.8)
        )
        self.tail_wings = actor.box(
            centers=c0, colors=(0.9, 0.9, 0.9), scales=(2.2, 0.08, 0.6)
        )
        self.prop_spinner = actor.sphere(
            centers=c0, colors=(0.25, 0.25, 0.25), radii=0.15
        )
        self.prop_blades = actor.box(
            centers=c0, colors=(0.1, 0.1, 0.1), scales=(0.06, 1.4, 0.03)
        )
        self.strut_l = actor.box(
            centers=c0, colors=(0.6, 0.6, 0.6), scales=(0.06, 0.9, 0.06)
        )
        self.strut_r = actor.box(
            centers=c0, colors=(0.6, 0.6, 0.6), scales=(0.06, 0.9, 0.06)
        )
        self.wheel_l = actor.sphere(centers=c0, colors=(0.15, 0.15, 0.15), radii=0.2)
        self.wheel_r = actor.sphere(centers=c0, colors=(0.15, 0.15, 0.15), radii=0.2)
        self.strut_t = actor.box(
            centers=c0, colors=(0.6, 0.6, 0.6), scales=(0.05, 0.8, 0.05)
        )
        self.wheel_t = actor.sphere(centers=c0, colors=(0.15, 0.15, 0.15), radii=0.15)

        self.parts = [
            self.fuselage,
            self.nose,
            self.canopy,
            self.wings,
            self.wing_stripe_l,
            self.wing_stripe_r,
            self.wing_light_l,
            self.wing_light_r,
            self.tail_fin,
            self.tail_wings,
            self.prop_spinner,
            self.prop_blades,
            self.strut_l,
            self.strut_r,
            self.wheel_l,
            self.wheel_r,
            self.strut_t,
            self.wheel_t,
        ]
        for p in self.parts:
            scene.add(p)
        self.prop_angle = 0.0

        self.offsets = {
            self.fuselage: np.array([0.0, 0.0, 0.0]),
            self.nose: np.array([0.0, 0.0, 1.6]),
            self.canopy: np.array([0.0, 0.35, 0.3]),
            self.wings: np.array([0.0, -0.05, 0.0]),
            self.wing_stripe_l: np.array([-2.0, -0.04, 0.0]),
            self.wing_stripe_r: np.array([2.0, -0.04, 0.0]),
            self.wing_light_l: np.array([-3.4, -0.04, 0.0]),
            self.wing_light_r: np.array([3.4, -0.04, 0.0]),
            self.tail_fin: np.array([0.0, 0.6, -1.6]),
            self.tail_wings: np.array([0.0, 0.0, -1.8]),
            self.prop_spinner: np.array([0.0, 0.0, 2.1]),
            self.prop_blades: np.array([0.0, 0.0, 2.12]),
            self.strut_l: np.array([-0.4, -0.85, 0.5]),
            self.strut_r: np.array([0.4, -0.85, 0.5]),
            self.wheel_l: np.array([-0.4, -1.35, 0.5]),
            self.wheel_r: np.array([0.4, -1.35, 0.5]),
            self.strut_t: np.array([0.0, -0.85, -1.5]),
            self.wheel_t: np.array([0.0, -1.3, -1.5]),
        }

    def update_transform(self, pos, quat, dt=0.02):
        self.prop_angle = (self.prop_angle + 1200.0 * dt) % 360.0
        prop_spin_quat = axis_angle_to_quat(V_FWD, self.prop_angle)
        prop_total_quat = quat_mult(quat, prop_spin_quat)

        for p in self.parts:
            p.local.position = pos + rotate_vector(quat, self.offsets[p])
            p.local.rotation = prop_total_quat if p is self.prop_blades else quat


player = Aircraft()
player.update_transform(state["player_pos"], state["player_quat"], 0.0)


class Tree:
    def __init__(self):
        c0 = np.array([V_ZERO])
        self.trunk = actor.cylinder(
            centers=c0,
            directions=np.array([V_UP]),
            colors=(0.4, 0.25, 0.15),
            height=3.5,
            radii=0.35,
        )
        self.foliage1 = actor.cone(
            centers=c0,
            directions=np.array([V_UP]),
            colors=(0.15, 0.45, 0.2),
            height=5.0,
            radii=1.8,
        )
        self.foliage2 = actor.cone(
            centers=c0,
            directions=np.array([V_UP]),
            colors=(0.2, 0.55, 0.25),
            height=4.0,
            radii=1.3,
        )
        self.parts = [self.trunk, self.foliage1, self.foliage2]
        for p in self.parts:
            scene.add(p)

    def set_position(self, pos):
        self.trunk.local.position = pos + np.array([0.0, 1.75, 0.0])
        self.foliage1.local.position = pos + np.array([0.0, 3.5, 0.0])
        self.foliage2.local.position = pos + np.array([0.0, 5.5, 0.0])


class Mountain:
    def __init__(self):
        h = np.random.uniform(65.0, 135.0)
        r = h * np.random.uniform(0.65, 0.85)
        self.height = h
        self.radius = r
        cap_h = h * 0.25
        c0 = np.array([V_ZERO])
        self.base = actor.cone(
            centers=c0,
            directions=np.array([V_UP]),
            colors=(0.42, 0.4, 0.4),
            height=h,
            radii=r,
        )
        self.cap = actor.cone(
            centers=c0,
            directions=np.array([V_UP]),
            colors=(0.95, 0.95, 0.98),
            height=cap_h,
            radii=r * 0.25 * 1.04,
        )
        self.parts = [self.base, self.cap]
        for p in self.parts:
            scene.add(p)

    def set_position(self, pos):
        self.base.local.position = pos + np.array([0.0, self.height / 2.0, 0.0])
        self.cap.local.position = pos + np.array(
            [0.0, self.height - (self.height * 0.25) / 2.0, 0.0]
        )


class WorldManager:
    def __init__(self):
        self.clouds = []
        self.trees = [Tree() for _ in range(TREE_COUNT)]
        self.mountains = [Mountain() for _ in range(MOUNTAIN_COUNT)]

        for tree in self.trees:
            self._spawn_object(tree, initial=True)
        for mtn in self.mountains:
            self._spawn_object(mtn, initial=True)

        for _ in range(CLOUD_COUNT):
            base_radius = np.random.uniform(5.0, 10.0)
            cloud_centers = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [-base_radius * 0.7, -base_radius * 0.1, 0.0],
                    [base_radius * 0.7, -base_radius * 0.1, 0.0],
                    [0.0, base_radius * 0.3, -base_radius * 0.2],
                ]
            )
            cloud_radii = np.array(
                [base_radius, base_radius * 0.7, base_radius * 0.8, base_radius * 0.6]
            )
            cloud = actor.sphere(
                centers=cloud_centers, colors=(0.95, 0.95, 0.98), radii=cloud_radii
            )
            self.clouds.append(cloud)
            self._spawn_object(cloud, initial=True)
            scene.add(cloud)

    def _spawn_object(self, obj, initial=False):
        p_pos = state["player_pos"]
        fwd = rotate_vector(state["player_quat"], V_FWD)
        dist = np.random.uniform(-100.0 if initial else 150.0, SPAWN_DIST * 1.5)
        sc = p_pos + fwd * dist

        if isinstance(obj, Mountain):
            x_offset = (1.0 if np.random.rand() > 0.5 else -1.0) * np.random.uniform(
                160.0, 350.0
            )
            obj.set_position(
                np.array(
                    [sc[0] + x_offset, 0.0, sc[2] + np.random.uniform(-100.0, 100.0)]
                )
            )
        elif isinstance(obj, Tree):
            x_offset = (1.0 if np.random.rand() > 0.5 else -1.0) * np.random.uniform(
                18.0, 150.0
            )
            obj.set_position(
                np.array(
                    [sc[0] + x_offset, 0.0, sc[2] + np.random.uniform(-80.0, 80.0)]
                )
            )
        else:
            spawn_y = max(60.0, p_pos[1] + np.random.uniform(-20.0, 150.0))
            obj.local.position = np.array(
                [
                    sc[0] + np.random.uniform(-150.0, 150.0),
                    spawn_y,
                    sc[2] + np.random.uniform(-100.0, 100.0),
                ]
            )

    def update(self):
        p_pos = state["player_pos"]
        fwd = rotate_vector(state["player_quat"], V_FWD)

        for pool in [self.clouds, self.trees, self.mountains]:
            for obj in pool:
                if isinstance(obj, Tree):
                    obj_pos = np.array(obj.trunk.local.position)
                elif isinstance(obj, Mountain):
                    obj_pos = np.array(obj.base.local.position)
                else:
                    obj_pos = np.array(obj.local.position)

                vec_to_obj = obj_pos - p_pos
                if (
                    np.dot(vec_to_obj, fwd) < -DESPAWN_DIST
                    or np.sum(vec_to_obj**2) > SPAWN_DIST_SQ
                ):
                    self._spawn_object(obj)


world = WorldManager()

hud_score = ui.TextBlock2D(
    text="SCORE: 00000",
    position=(30, 710),
    size=(300, 30),
    font_size=22,
    color=(0.0, 0.9, 1.0),
    bold=True,
)
hud_alt = ui.TextBlock2D(
    text="ALTITUDE: 0 m",
    position=(30, 675),
    size=(300, 30),
    font_size=18,
    color=(0.0, 0.8, 0.9),
)
hud_speed = ui.TextBlock2D(
    text="AIRSPEED: 0 kts",
    position=(30, 640),
    size=(300, 30),
    font_size=18,
    color=(0.0, 0.8, 0.9),
)

scene.add(hud_score)
scene.add(hud_alt)
scene.add(hud_speed)


def restart_game():
    state["is_playing"] = True
    state["score"] = 0.0
    state["speed"] = 0.0
    state["player_pos"] = np.array([0.0, 2.0, 0.0])
    state["player_quat"] = np.array([0.0, 0.0, 0.0, 1.0])
    state["cam_pos"] = np.array([0.0, 150.0, -50.0])
    state["cam_up"] = np.array([0.0, 1.0, 0.0])
    state["pitch_rate"] = 0.0
    state["roll_rate"] = 0.0
    state["yaw_rate"] = 0.0
    state["tick_count"] = 0
    for pool in [world.clouds, world.trees, world.mountains]:
        for obj in pool:
            world._spawn_object(obj, initial=True)


def on_key_down(event):
    state["keys"].add(event.key.lower())
    if event.key.lower() == "r":
        restart_game()


def on_key_up(event):
    if event.key.lower() in state["keys"]:
        state["keys"].remove(event.key.lower())


def check_collisions():
    p_pos = state["player_pos"]

    # Tree collisions
    for tree in world.trees:
        tree_base = np.array(tree.trunk.local.position) - np.array([0.0, 1.75, 0.0])
        horiz_dist = np.linalg.norm(p_pos[[0, 2]] - tree_base[[0, 2]])
        if horiz_dist < 3.0 and p_pos[1] < 7.5:
            return True

    # Mountain collisions
    for mtn in world.mountains:
        mtn_base = np.array(mtn.base.local.position) - np.array(
            [0.0, mtn.height / 2.0, 0.0]
        )
        h = mtn.height
        r = mtn.radius
        if p_pos[1] < h:
            horiz_dist = np.linalg.norm(p_pos[[0, 2]] - mtn_base[[0, 2]])
            r_at_y = r * (1.0 - p_pos[1] / h)
            if horiz_dist < r_at_y + 3.0:
                return True

    return False


def consume_mouse(event):
    pass


def game_tick(showm):
    dt = 0.02
    if not state["is_playing"]:
        return
    keys = state["keys"]

    fwd = rotate_vector(state["player_quat"], V_FWD)
    up = rotate_vector(state["player_quat"], V_UP)
    right = rotate_vector(state["player_quat"], V_R)

    min_y = get_surface_height(state["player_pos"]) + 1.55

    if " " in keys:
        state["speed"] += 50.0 * dt
    elif "enter" in keys:
        state["speed"] -= 50.0 * dt
    else:
        state["speed"] -= 5.0 * dt

    state["speed"] -= fwd[1] * 25.0 * dt
    state["speed"] = np.clip(
        state["speed"],
        40.0 if state["player_pos"][1] > min_y + 1.0 else 0.0,
        state["max_speed"],
    )

    target_pitch, target_roll, target_yaw = 0.0, 0.0, 0.0

    # Corrected Aerodynamic Mappings based on Right-Hand Rule Matrix rotation
    if "s" in keys:
        target_pitch += 80.0  # Nose Down
    if "w" in keys:
        target_pitch -= 80.0  # Nose Up
    if "e" in keys:
        target_roll += 120.0  # Roll Left
    if "q" in keys:
        target_roll -= 120.0  # Roll Right
    if "d" in keys:
        target_yaw -= 50.0  # Yaw Left
    if "a" in keys:
        target_yaw += 50.0  # Yaw Right

    # Doubled the responsiveness for sharp, immediate control surface bite
    state["pitch_rate"] += (target_pitch - state["pitch_rate"]) * 12.0 * dt
    state["roll_rate"] += (target_roll - state["roll_rate"]) * 12.0 * dt
    state["yaw_rate"] += (target_yaw - state["yaw_rate"]) * 12.0 * dt

    # Fixed the death-loop sign error on pitch auto-stabilization
    if not ("a" in keys or "d" in keys):
        state["roll_rate"] += (-right[1] * 100.0) * dt
    if not ("w" in keys or "s" in keys):
        state["pitch_rate"] += (fwd[1] * 50.0) * dt

    qx = axis_angle_to_quat(V_R, state["pitch_rate"] * dt)
    qy = axis_angle_to_quat(V_UP, state["yaw_rate"] * dt)
    qz = axis_angle_to_quat(V_FWD, state["roll_rate"] * dt)
    state["player_quat"] = quat_mult(
        state["player_quat"], quat_mult(qy, quat_mult(qx, qz))
    )

    lift = (
        25.0
        * np.clip(state["speed"] / state["takeoff_speed"], 0.0, 1.0)
        * max(0.0, up[1])
    )

    vel = fwd * state["speed"]
    vel[1] += lift - 25.0
    state["player_pos"] += vel * dt

    if state["player_pos"][1] <= min_y:
        state["player_pos"][1] = min_y
        if state["speed"] < state["takeoff_speed"]:
            fwd_flat = np.array([fwd[0], 0.0, fwd[2]])
            if np.linalg.norm(fwd_flat) > 0.001:
                state["player_quat"] = axis_angle_to_quat(
                    V_UP, np.degrees(np.arctan2(fwd_flat[0], fwd_flat[2]))
                )
                state["pitch_rate"], state["roll_rate"] = 0.0, 0.0

    ground.local.position = [state["player_pos"][0], -0.5, state["player_pos"][2]]
    sun.local.position = [state["player_pos"][0], 300.0, state["player_pos"][2] + 800.0]

    runway_mgr.update(state["player_pos"])
    player.update_transform(state["player_pos"], state["player_quat"], dt)
    world.update()

    state["score"] += state["speed"] * dt * 0.1
    hud_score.message = f"SCORE: {int(state['score']):05d}"
    hud_alt.message = f"ALTITUDE: {max(0, int(state['player_pos'][1] - min_y))} m"
    hud_speed.message = f"AIRSPEED: {int(state['speed'])} kts"

    if check_collisions():
        state["is_playing"] = False
        restart_game()

    fwd_now = rotate_vector(state["player_quat"], V_FWD)
    up_now = rotate_vector(state["player_quat"], V_UP)

    target_cam_pos = state["player_pos"] - fwd_now * 50.0 + up_now * 8.0

    cam_min_y = get_surface_height(state["cam_pos"]) + 1.0
    if target_cam_pos[1] < cam_min_y:
        target_cam_pos[1] = cam_min_y

    state["cam_pos"] += (target_cam_pos - state["cam_pos"]) * 10.0 * dt

    state["cam_up"] += (up_now - state["cam_up"]) * 5.0 * dt
    state["cam_up"] /= np.linalg.norm(state["cam_up"])

    camera = showm.screens[0].camera if hasattr(showm, "screens") else scene.camera()
    camera.local.position = state["cam_pos"]
    camera.look_at(state["player_pos"])
    camera.reference_up = state["cam_up"]

    showm.render()


if __name__ == "__main__":
    showm = window.ShowManager(
        scene=scene, size=(1024, 768), title="True 3D Flight Simulator"
    )
    showm.renderer.add_event_handler(on_key_down, EventType.KEY_DOWN)
    showm.renderer.add_event_handler(on_key_up, EventType.KEY_UP)

    # Block orbit controller from messing with the chase camera
    showm.screens[0].controller.enabled = False

    showm.register_callback(game_tick, 0.01, True, "GameLoop", showm)
    showm.start()
