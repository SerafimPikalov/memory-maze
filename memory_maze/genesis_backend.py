"""
Genesis physics backend for Memory Maze.

Replaces the MuJoCo/dm_control stack with Genesis for GPU-accelerated
physics and rendering. Implements the same gym.Env interface.
"""

import math
import os
from collections import namedtuple

# Prevent matplotlib (imported by Genesis) from initializing Tk, which
# crashes in spawned subprocesses on macOS (Tk requires main thread).
os.environ.setdefault("MPLBACKEND", "Agg")
# Prevent Genesis's viewer.py from creating a Tk root at import time —
# this crashes in spawned subprocesses on macOS (Cocoa NSApplication error).
os.environ.setdefault("GENESIS_SKIP_TK_INIT", "1")

import labmaze
import numpy as np
from dm_control.locomotion.arenas import covering

try:
    import genesis as gs
    import torch
except ImportError:
    gs = None
    torch = None

try:
    import gym
    from gym import spaces
except ImportError:
    gym = None
    spaces = None

# BatchRenderer (Madrona) availability detection — CUDA-only, Linux x86-64
try:
    import gs_madrona
    _BATCH_RENDERER_AVAILABLE = True
except ImportError:
    _BATCH_RENDERER_AVAILABLE = False


def _use_batch_renderer():
    """Use BatchRenderer when gs_madrona is installed and CUDA backend is active."""
    return (_BATCH_RENDERER_AVAILABLE
            and gs is not None
            and gs._initialized
            and gs.device.type == 'cuda')

# ---------------------------------------------------------------------------
# Constants from the MuJoCo reference implementation (maze.py, tasks.py)
# ---------------------------------------------------------------------------

TARGET_COLORS = [
    np.array([170, 38, 30]) / 220,   # red
    np.array([99, 170, 88]) / 220,   # green
    np.array([39, 140, 217]) / 220,  # blue
    np.array([93, 105, 199]) / 220,  # purple
    np.array([220, 193, 59]) / 220,  # yellow
    np.array([220, 128, 107]) / 220, # salmon
]

# Discrete action set: same 6 actions as DiscreteActionSetWrapper
ACTION_SET = [
    np.array([0.0, 0.0]),    # noop
    np.array([-1.0, 0.0]),   # forward
    np.array([0.0, -1.0]),   # left
    np.array([0.0, +1.0]),   # right
    np.array([-1.0, -1.0]),  # forward + left
    np.array([-1.0, +1.0]),  # forward + right
]

# Walker physics from jumping_ball_with_head.xml + RollingBallWithFriction
WALKER_RADIUS = 0.2
WALKER_SHELL_MASS = 1.0
WALKER_BALLAST_MASS = 20.0
WALKER_TOTAL_MASS = WALKER_SHELL_MASS + WALKER_BALLAST_MASS  # 21 kg
WALKER_CAMERA_HEIGHT = 0.3

# Actuator params from XML
ROLL_GEAR = -50.0       # general actuator gear for roll joint
STEER_GEAR = 30.0       # motor gear for steer joint
ROLL_DAMPING = 5.0      # from RollingBallWithFriction
STEER_DAMPING = 20.0    # from RollingBallWithFriction

# MuJoCo camera
CAMERA_FOV = 80  # fovy from XML

# Target detection
TARGET_RADIUS = 0.6
TARGET_ACTIVATION_GAP = 2 * TARGET_RADIUS  # gap=2*radius in MuJoCo target_sphere

# Timing
DEFAULT_CONTROL_FREQ = 4.0
DEFAULT_PHYSICS_TIMESTEP = 0.005
DEFAULT_CONTROL_TIMESTEP = 1.0 / DEFAULT_CONTROL_FREQ  # 0.25s

# Max pre-allocated walls (covers 15x15 mazes generously)
MAX_WALLS = 64

# Maze config per size (maze_size -> (n_targets, time_limit, max_rooms, room_max_size))
MAZE_CONFIGS = {
    9:  (3, 250, 6, 5),
    11: (4, 500, 6, 5),
    13: (5, 750, 6, 5),
    15: (6, 1000, 9, 3),
}

# ---------------------------------------------------------------------------
# Wall segment extraction (uses dm_control.locomotion.arenas.covering)
# ---------------------------------------------------------------------------

WallSegment = namedtuple('WallSegment', ['pos', 'half_size'])


def extract_wall_segments(maze, xy_scale=2.0, z_height=1.5):
    """Extract wall box positions and sizes from a labmaze text grid.

    Returns list of WallSegment(pos=[x,y,z], half_size=[hx,hy,hz]).
    Uses the same coordinate conversion as MazeWithTargets._make_wall_geoms().
    """
    x_offset = (maze.width - 1) / 2.0
    y_offset = (maze.height - 1) / 2.0

    segments = []
    # Find all wall characters in the entity layer
    wall_chars = set()
    for row in maze.entity_layer:
        for c in row:
            if c not in (' ', 'P', 'G'):
                wall_chars.add(c)

    for wc in wall_chars:
        walls = covering.make_walls(maze.entity_layer, wall_char=wc, make_odd_sized_walls=True)
        for wall in walls:
            wall_mid_y = (wall.start.y + wall.end.y - 1) / 2.0
            wall_mid_x = (wall.start.x + wall.end.x - 1) / 2.0
            pos = np.array([
                (wall_mid_x - x_offset) * xy_scale,
                -(wall_mid_y - y_offset) * xy_scale,
                z_height / 2.0,
            ])
            half_size = np.array([
                (wall.end.x - wall_mid_x - 0.5) * xy_scale,
                (wall.end.y - wall_mid_y - 0.5) * xy_scale,
                z_height / 2.0,
            ])
            segments.append(WallSegment(pos=pos, half_size=half_size))

    return segments


def extract_positions(maze, token, xy_scale=2.0):
    """Extract world positions for a given token (P=spawn, G=target) from maze."""
    x_offset = (maze.width - 1) / 2.0
    y_offset = (maze.height - 1) / 2.0
    positions = []
    for y in range(maze.height):
        for x in range(maze.width):
            if maze.entity_layer[y, x] == token:
                pos = np.array([
                    (x - x_offset) * xy_scale,
                    -(y - y_offset) * xy_scale,
                    0.0,
                ])
                positions.append(pos)
    return positions


# ---------------------------------------------------------------------------
# Phase 1: Static Scene Builder
# ---------------------------------------------------------------------------

class GenesisMazeScene:
    """Manages the Genesis scene with walls, floor, walker, targets, and camera."""

    def __init__(
        self,
        maze_size=9,
        n_targets=3,
        xy_scale=2.0,
        z_height=1.5,
        camera_resolution=64,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
        physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
        max_rooms=6,
        room_min_size=3,
        room_max_size=5,
        target_height_above_ground=-0.6,
    ):
        assert gs is not None, "Genesis is not installed. Install with: pip install genesis-world"

        self.maze_size = maze_size
        self.n_targets = n_targets
        self.xy_scale = xy_scale
        self.z_height = z_height
        self.camera_resolution = camera_resolution
        self.control_timestep = control_timestep
        self.physics_timestep = physics_timestep
        self.max_rooms = max_rooms
        self.room_min_size = room_min_size
        self.room_max_size = room_max_size
        self.target_height_above_ground = target_height_above_ground

        # Number of substeps per control step
        self.n_substeps = max(1, int(round(control_timestep / physics_timestep)))

        # Outer maze dimensions (with exterior walls)
        self.outer_size = maze_size + 2

        # Create Genesis scene
        if _use_batch_renderer():
            renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
        else:
            renderer = gs.renderers.Rasterizer()

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=physics_timestep,
                substeps=1,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
            ),
            renderer=renderer,
        )

        # BatchRenderer requires explicit lights (Rasterizer uses default OpenGL lighting)
        if _use_batch_renderer():
            self.scene.add_light(
                pos=(0, 0, 10), dir=(0, 0, -1),
                directional=True, intensity=1.0, color=(1.0, 1.0, 1.0),
            )

        # --- Floor ---
        self.floor = self.scene.add_entity(
            morph=gs.morphs.Plane(pos=(0, 0, 0)),
            material=gs.materials.Rigid(friction=0.5),
            surface=gs.surfaces.Default(color=(0.4, 0.5, 0.6, 1.0)),
        )

        # --- Pre-allocate wall entities ---
        self.wall_entities = []
        for i in range(MAX_WALLS):
            wall = self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(0, 0, -10),  # Start underground
                    size=(1, 1, z_height),  # Will be resized per maze
                    fixed=True,
                ),
                material=gs.materials.Rigid(friction=0.5),
                surface=gs.surfaces.Default(color=(0.8, 0.7, 0.3, 1.0)),  # Yellow-ish walls
            )
            self.wall_entities.append(wall)

        # --- Walker (rolling ball) ---
        self.walker = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0, 0, WALKER_RADIUS),
                radius=WALKER_RADIUS,
                fixed=False,
            ),
            material=gs.materials.Rigid(
                friction=0.5,
                rho=WALKER_TOTAL_MASS / ((4.0 / 3.0) * math.pi * WALKER_RADIUS**3),
            ),
            surface=gs.surfaces.Default(color=(0.757, 0.757, 0.757, 1.0)),
        )

        # --- Targets (non-colliding colored spheres) ---
        self.target_entities = []
        self.target_colors = list(TARGET_COLORS)
        for i in range(n_targets):
            color = self.target_colors[i]
            target = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    pos=(0, 0, -10),  # Start underground
                    radius=TARGET_RADIUS,
                    fixed=True,
                    collision=False,  # Targets don't block movement
                ),
                surface=gs.surfaces.Default(
                    color=(float(color[0]), float(color[1]), float(color[2]), 1.0),
                ),
            )
            self.target_entities.append(target)

        # --- Camera (egocentric, attached to walker) ---
        self.camera = self.scene.add_camera(
            res=(camera_resolution, camera_resolution),
            pos=(0, 0, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
            lookat=(0, 1, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
            fov=CAMERA_FOV,
            near=0.05,
            far=50.0,
        )

        # State tracking
        self._built = False
        self._maze = None
        self._target_world_positions = []  # Current world positions of placed targets
        self._walker_heading = 0.0  # Current heading angle in radians

    def build(self):
        """Build the Genesis scene. Must be called once before stepping."""
        self.scene.build()
        self._built = True

        # Configure walker DOF damping after build
        # DOFs: [tx, ty, tz, rx, ry, rz]
        # Translational damping emulates roll friction (ROLL_DAMPING=5.0)
        # Rotational z-damping emulates steer friction (STEER_DAMPING=20.0)
        self.walker.set_dofs_damping(np.array([
            ROLL_DAMPING, ROLL_DAMPING, 0.0,  # x, y, z translation
            0.0, 0.0, STEER_DAMPING,          # rx, ry, rz rotation
        ]))

    def reset(self, rng):
        """Reset the scene: regenerate maze, place walker and targets."""
        assert self._built, "Scene must be built before reset"

        # Generate new maze
        seed = rng.randint(2147483648)
        self._maze = labmaze.RandomMaze(
            height=self.outer_size,
            width=self.outer_size,
            max_rooms=self.max_rooms,
            room_min_size=self.room_min_size,
            room_max_size=self.room_max_size,
            spawns_per_room=1,
            objects_per_room=1,
            random_seed=seed,
        )

        # Configure walls
        wall_segments = extract_wall_segments(self._maze, self.xy_scale, self.z_height)
        self._configure_walls(wall_segments)

        # Place walker at random spawn
        spawn_positions = extract_positions(self._maze, 'P', self.xy_scale)
        if spawn_positions:
            spawn_idx = rng.randint(len(spawn_positions))
            spawn_pos = spawn_positions[spawn_idx]
        else:
            spawn_pos = np.array([0.0, 0.0, 0.0])

        self._walker_heading = rng.uniform(0, 2 * math.pi)
        self.walker.set_pos(np.array([spawn_pos[0], spawn_pos[1], WALKER_RADIUS]))
        # Set walker rotation as quaternion (rotation around z-axis)
        qw = math.cos(self._walker_heading / 2)
        qz = math.sin(self._walker_heading / 2)
        self.walker.set_quat(np.array([qw, 0.0, 0.0, qz]))
        # Zero all DOF velocities (6 DOFs: 3 translational + 3 rotational)
        self.walker.set_dofs_velocity(np.zeros(6))

        # Place targets at random target positions
        target_positions = extract_positions(self._maze, 'G', self.xy_scale)
        rng.shuffle(target_positions)
        self._target_world_positions = []
        for i in range(self.n_targets):
            if i < len(target_positions):
                tpos = target_positions[i]
                target_z = TARGET_RADIUS + self.target_height_above_ground
                world_pos = np.array([tpos[0], tpos[1], target_z])
                self.target_entities[i].set_pos(world_pos)
                self._target_world_positions.append(world_pos)
            else:
                # Not enough positions, hide underground
                self.target_entities[i].set_pos(np.array([0.0, 0.0, -10.0]))
                self._target_world_positions.append(np.array([0.0, 0.0, -10.0]))

        # Update camera to walker position
        self._update_camera()

    def _configure_walls(self, wall_segments):
        """Reposition pre-allocated wall entities for the current maze layout."""
        for i, wall_entity in enumerate(self.wall_entities):
            if i < len(wall_segments):
                seg = wall_segments[i]
                wall_entity.set_pos(seg.pos)
            else:
                # Move unused walls underground
                wall_entity.set_pos(np.array([0.0, 0.0, -10.0]))

    def apply_action(self, continuous_action):
        """Apply a continuous [roll, steer] action to the walker.

        The MuJoCo walker uses velocity actuators with damping. We approximate
        this by applying forces/torques via Genesis DOF force control.

        MuJoCo actuators:
          roll: general actuator, gear=-50, controls hinge on x-axis
          steer: motor, gear=30, controls hinge on z-axis

        With damping (from RollingBallWithFriction):
          roll_damping=5.0, steer_damping=20.0

        The effective force: F = gear * ctrl - damping * velocity

        Genesis sphere DOFs: [tx, ty, tz, rx, ry, rz]
        We apply force along the walker's heading direction (roll)
        and torque around z-axis (steer).
        """
        roll_cmd, steer_cmd = continuous_action[0], continuous_action[1]

        # Get heading for directional force
        heading = self._get_walker_heading()
        cos_h, sin_h = math.cos(heading), math.sin(heading)

        # Roll force components in world frame
        # Negative roll_cmd = forward in MuJoCo convention
        roll_force = ROLL_GEAR * roll_cmd
        fx = roll_force * cos_h
        fy = roll_force * sin_h

        # Steer torque around z-axis
        steer_torque = STEER_GEAR * steer_cmd

        # DOF forces: [fx, fy, fz, tx, ty, tz]
        # Add damping via control_dofs_force (Genesis handles per-step application)
        forces = np.array([fx, fy, 0.0, 0.0, 0.0, steer_torque])
        self.walker.control_dofs_force(forces)

    def step(self):
        """Step physics for one control timestep (multiple substeps)."""
        for _ in range(self.n_substeps):
            self.scene.step(update_visualizer=False)
        self._update_camera()

    def _get_walker_heading(self):
        """Get walker heading angle from its quaternion."""
        quat = self.walker.get_quat()
        q = quat.cpu().numpy() if hasattr(quat, 'cpu') else np.asarray(quat)
        # Extract yaw from quaternion (w, x, y, z)
        heading = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                             1 - 2 * (q[2]**2 + q[3]**2))
        return heading

    def _update_camera(self):
        """Update camera position/orientation to follow walker."""
        pos = self.walker.get_pos()
        walker_pos = pos.cpu().numpy() if hasattr(pos, 'cpu') else np.asarray(pos)
        heading = self._get_walker_heading()

        # Camera sits on top of the ball, looking forward
        cam_pos = np.array([
            walker_pos[0],
            walker_pos[1],
            walker_pos[2] + WALKER_CAMERA_HEIGHT,
        ])
        # Look direction: forward along heading, slightly downward
        look_dist = 1.0
        lookat = np.array([
            walker_pos[0] + look_dist * math.cos(heading),
            walker_pos[1] + look_dist * math.sin(heading),
            walker_pos[2] + WALKER_CAMERA_HEIGHT - 0.1,
        ])
        self.camera.set_pose(pos=cam_pos, lookat=lookat)

    def get_walker_position(self):
        """Get walker [x, y, z] position as numpy array."""
        pos = self.walker.get_pos()
        return pos.cpu().numpy() if hasattr(pos, 'cpu') else np.asarray(pos)

    def render_egocentric(self):
        """Render egocentric camera view. Returns uint8 numpy [H, W, 3]."""
        result = self.camera.render(rgb=True, depth=False, segmentation=False)
        # render() returns a tuple: (rgb, depth, segmentation, normal)
        rgb = result[0]
        if hasattr(rgb, 'cpu'):
            rgb = rgb.cpu().numpy()
        return np.asarray(rgb, dtype=np.uint8)

    def check_target_contacts(self, walker_pos):
        """Check which targets are within activation distance of walker.

        Returns boolean array of shape (n_targets,).
        """
        contacts = np.zeros(self.n_targets, dtype=bool)
        for i in range(self.n_targets):
            tpos = self._target_world_positions[i]
            if tpos[2] < -5:  # Underground = not active
                continue
            dist = np.linalg.norm(walker_pos[:2] - tpos[:2])
            contacts[i] = dist < TARGET_ACTIVATION_GAP
        return contacts

    def hide_target(self, index):
        """Move target underground (invisible)."""
        self.target_entities[index].set_pos(np.array([0.0, 0.0, -10.0]))
        self._target_world_positions[index] = np.array([0.0, 0.0, -10.0])

    def show_target(self, index, position):
        """Move target to given position (visible)."""
        self.target_entities[index].set_pos(position)
        self._target_world_positions[index] = position.copy()


# ---------------------------------------------------------------------------
# Phase 4: Full Gym Wrapper
# ---------------------------------------------------------------------------

class GenesisMemoryMazeEnv(gym.Env):
    """Drop-in gym.Env replacement for Memory Maze using Genesis backend.

    Implements the same interface as the MuJoCo-based MemoryMaze:
    - action_space: Discrete(6)
    - observation_space: Box(0, 255, (resolution, resolution, 3), uint8)
    - reset() -> obs (HWC uint8)
    - step(action) -> (obs, reward, done, info)
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(
        self,
        maze_size=9,
        n_targets=None,
        time_limit=None,
        camera_resolution=64,
        seed=None,
        good_visibility=False,
        control_freq=DEFAULT_CONTROL_FREQ,
        **kwargs,
    ):
        super().__init__()
        assert gs is not None, "Genesis not installed"
        assert gym is not None, "gym not installed"

        # Look up defaults from maze config
        cfg = MAZE_CONFIGS.get(maze_size, (3, 250, 6, 5))
        if n_targets is None:
            n_targets = cfg[0]
        if time_limit is None:
            time_limit = cfg[1]
        max_rooms = cfg[2]
        room_max_size = cfg[3]

        self._maze_size = maze_size
        self._n_targets = n_targets
        self._time_limit = time_limit
        self._camera_resolution = camera_resolution
        self._control_freq = control_freq

        control_timestep = 1.0 / control_freq
        z_height = 0.4 if good_visibility else 1.5
        target_height = 0.5 if good_visibility else -0.6

        # Max steps per episode
        self._max_steps = int(time_limit * control_freq)

        # Gym spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(camera_resolution, camera_resolution, 3),
            dtype=np.uint8,
        )

        # RNG
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # Initialize Genesis (skip if already initialized)
        # Single-env mode is used by forked actor processes — CUDA can't be
        # reinitialized in forked subprocesses, so always use CPU backend.
        if not gs._initialized:
            gs.init(backend=gs.cpu, logging_level='warning')

        # Build scene
        self._scene = GenesisMazeScene(
            maze_size=maze_size,
            n_targets=n_targets,
            z_height=z_height,
            camera_resolution=camera_resolution,
            control_timestep=control_timestep,
            max_rooms=max_rooms,
            room_max_size=room_max_size,
            target_height_above_ground=target_height,
        )
        self._scene.build()

        # Episode state
        self._step_count = 0
        self._current_target_ix = 0
        self._targets_obtained = 0
        self._target_colors = list(TARGET_COLORS)
        self._target_world_positions = []  # Stored for re-showing after collection

    def seed(self, seed=None):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        """Reset environment: regenerate maze, place entities, return first obs."""
        self._scene.reset(self._rng)
        self._step_count = 0
        self._targets_obtained = 0

        # Store target positions for the episode
        self._target_world_positions = [
            pos.copy() for pos in self._scene._target_world_positions
        ]

        # Pick initial target
        self._current_target_ix = self._rng.randint(self._n_targets)

        return self._render_obs()

    def step(self, action):
        """Execute one environment step."""
        # Map discrete action to continuous [roll, steer]
        continuous = ACTION_SET[action]

        # Apply action and step physics
        self._scene.apply_action(continuous)
        self._scene.step()

        # Check target contacts
        walker_pos = self._scene.get_walker_position()
        contacts = self._scene.check_target_contacts(walker_pos)

        # Process reward
        reward = 0.0
        for i in range(self._n_targets):
            if contacts[i] and i == self._current_target_ix:
                reward = 1.0
                self._targets_obtained += 1
                self._pick_new_target()

        # Render observation
        obs = self._render_obs()

        # Check done
        self._step_count += 1
        done = self._step_count >= self._max_steps

        info = {}
        if done:
            info['TimeLimit.truncated'] = True
            info['targets_obtained'] = self._targets_obtained

        return obs, reward, done, info

    def _pick_new_target(self):
        """Pick a new random target (not the one currently being touched)."""
        while True:
            ix = self._rng.randint(self._n_targets)
            if ix != self._current_target_ix:
                self._current_target_ix = ix
                break

    def _render_obs(self):
        """Render egocentric view with target color border."""
        img = self._scene.render_egocentric()

        # Ensure correct resolution
        if img.shape[0] != self._camera_resolution or img.shape[1] != self._camera_resolution:
            # Resize if needed (shouldn't happen normally)
            from PIL import Image
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((self._camera_resolution, self._camera_resolution))
            img = np.array(pil_img)

        # Draw target color border (same as TargetColorAsBorderWrapper)
        color = self._target_colors[self._current_target_ix]
        B = int(2 * math.sqrt(self._camera_resolution / 64))
        border_color = (color * 255 * 0.7).astype(np.uint8)
        img[:, :B] = border_color
        img[:, -B:] = border_color
        img[:B, :] = border_color
        img[-B:, :] = border_color

        return img

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self._render_obs()
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Phase 5: Batched Scene Builder (GPU-parallel environments)
# ---------------------------------------------------------------------------

# Deeper hidden depth for batch mode — wider AABB safety margin
BATCH_HIDDEN_Z = -100.0


class BatchGenesisMazeScene:
    """Manages N parallel Genesis mazes in a single scene.

    Uses ``scene.build(n_envs=N)`` so all environments share the same entity
    pool but have independent state (positions, velocities, etc.) indexed by
    ``envs_idx``.  Each env gets its own random maze via
    ``configure_walls_for_env``.

    The entity layout mirrors ``GenesisMazeScene`` (floor, MAX_WALLS walls,
    1 walker sphere, n_targets target spheres) but every setter/getter takes
    an ``envs_idx`` tensor to address individual environments.
    """

    def __init__(
        self,
        n_envs,
        maze_size=9,
        n_targets=3,
        xy_scale=2.0,
        z_height=1.5,
        camera_resolution=64,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
        physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
        max_rooms=6,
        room_min_size=3,
        room_max_size=5,
        target_height_above_ground=-0.6,
        max_collision_pairs=200,
    ):
        assert gs is not None, "Genesis is not installed"
        assert n_envs >= 1, "n_envs must be >= 1"

        self.n_envs = n_envs
        self.maze_size = maze_size
        self.n_targets = n_targets
        self.xy_scale = xy_scale
        self.z_height = z_height
        self.camera_resolution = camera_resolution
        self.control_timestep = control_timestep
        self.physics_timestep = physics_timestep
        self.max_rooms = max_rooms
        self.room_min_size = room_min_size
        self.room_max_size = room_max_size
        self.target_height_above_ground = target_height_above_ground

        self.n_substeps = max(1, int(round(control_timestep / physics_timestep)))
        self.outer_size = maze_size + 2

        # --- Create scene ---
        if _use_batch_renderer():
            renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
        else:
            renderer = gs.renderers.Rasterizer()

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=physics_timestep,
                substeps=1,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=max_collision_pairs,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                env_separate_rigid=not _use_batch_renderer(),
            ),
            renderer=renderer,
        )

        # BatchRenderer requires explicit lights (Rasterizer uses default OpenGL lighting)
        if _use_batch_renderer():
            self.scene.add_light(
                pos=(0, 0, 10), dir=(0, 0, -1),
                directional=True, intensity=1.0, color=(1.0, 1.0, 1.0),
            )

        # --- Floor ---
        self.floor = self.scene.add_entity(
            morph=gs.morphs.Plane(pos=(0, 0, 0)),
            material=gs.materials.Rigid(friction=0.5),
            surface=gs.surfaces.Default(color=(0.4, 0.5, 0.6, 1.0)),
        )

        # --- Pre-allocate wall pool ---
        self.wall_entities = []
        for _ in range(MAX_WALLS):
            wall = self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(0, 0, BATCH_HIDDEN_Z),
                    size=(1, 1, z_height),
                    fixed=True,
                ),
                material=gs.materials.Rigid(friction=0.5),
                surface=gs.surfaces.Default(color=(0.8, 0.7, 0.3, 1.0)),
            )
            self.wall_entities.append(wall)

        # --- Walker ---
        self.walker = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0, 0, WALKER_RADIUS),
                radius=WALKER_RADIUS,
                fixed=False,
            ),
            material=gs.materials.Rigid(
                friction=0.5,
                rho=WALKER_TOTAL_MASS / ((4.0 / 3.0) * math.pi * WALKER_RADIUS ** 3),
            ),
            surface=gs.surfaces.Default(color=(0.757, 0.757, 0.757, 1.0)),
        )

        # --- Targets (non-colliding) ---
        self.target_entities = []
        self.target_colors = list(TARGET_COLORS)
        for i in range(n_targets):
            color = self.target_colors[i]
            target = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    pos=(0, 0, BATCH_HIDDEN_Z),
                    radius=TARGET_RADIUS,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Default(
                    color=(float(color[0]), float(color[1]), float(color[2]), 1.0),
                ),
            )
            self.target_entities.append(target)

        # --- Cameras ---
        if _use_batch_renderer():
            # BatchRenderer: one camera renders all envs simultaneously
            self.camera = self.scene.add_camera(
                res=(camera_resolution, camera_resolution),
                pos=(0, 0, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
                lookat=(0, 1, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
                fov=CAMERA_FOV, near=0.05, far=50.0,
            )
            self.cameras = None  # Signal: using batch renderer
        else:
            # Rasterizer: per-env cameras with env_idx binding
            self.camera = None
            self.cameras = []
            for i in range(n_envs):
                cam = self.scene.add_camera(
                    res=(camera_resolution, camera_resolution),
                    pos=(0, 0, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
                    lookat=(0, 1, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
                    fov=CAMERA_FOV, near=0.05, far=50.0,
                    env_idx=i,
                )
                self.cameras.append(cam)

        self._built = False

    def build(self):
        """Build the scene with n_envs parallel environments."""
        self.scene.build(n_envs=self.n_envs)
        self._built = True

        # Configure walker damping (broadcasts to all envs)
        damping = np.array([
            ROLL_DAMPING, ROLL_DAMPING, 0.0,
            0.0, 0.0, STEER_DAMPING,
        ])
        self.walker.set_dofs_damping(damping)

    def configure_walls_for_env(self, env_idx, wall_segments):
        """Position the wall pool for a single environment.

        Active walls are placed at their maze positions; unused walls are
        hidden at ``BATCH_HIDDEN_Z``.
        """
        idx_tensor = torch.tensor([env_idx], dtype=torch.int32)
        for i, wall_entity in enumerate(self.wall_entities):
            if i < len(wall_segments):
                seg = wall_segments[i]
                wall_entity.set_pos(
                    np.array(seg.pos, dtype=np.float32),
                    envs_idx=idx_tensor,
                )
            else:
                wall_entity.set_pos(
                    np.array([0.0, 0.0, BATCH_HIDDEN_Z], dtype=np.float32),
                    envs_idx=idx_tensor,
                )

    def set_walker_pose(self, env_idx, pos, heading):
        """Set walker position and heading for a single environment."""
        idx_tensor = torch.tensor([env_idx], dtype=torch.int32)
        self.walker.set_pos(
            np.array([pos[0], pos[1], WALKER_RADIUS], dtype=np.float32),
            envs_idx=idx_tensor,
        )
        qw = math.cos(heading / 2)
        qz = math.sin(heading / 2)
        self.walker.set_quat(
            np.array([qw, 0.0, 0.0, qz], dtype=np.float32),
            envs_idx=idx_tensor,
        )
        self.walker.set_dofs_velocity(
            np.zeros(6, dtype=np.float32),
            envs_idx=idx_tensor,
        )

    def set_target_pos(self, env_idx, target_idx, pos):
        """Set target sphere position for a single environment."""
        idx_tensor = torch.tensor([env_idx], dtype=torch.int32)
        self.target_entities[target_idx].set_pos(
            np.array(pos, dtype=np.float32),
            envs_idx=idx_tensor,
        )

    def hide_target(self, env_idx, target_idx):
        """Hide a target underground for a single environment."""
        self.set_target_pos(env_idx, target_idx,
                            [0.0, 0.0, BATCH_HIDDEN_Z])

    def apply_actions_batched(self, forces_tensor):
        """Apply DOF forces to the walker across all environments.

        Parameters
        ----------
        forces_tensor : array-like, shape (n_envs, 6)
            Per-environment DOF forces [fx, fy, fz, tx, ty, tz].
        """
        self.walker.control_dofs_force(forces_tensor)

    def step(self):
        """Step physics for one control timestep (all envs advance together)."""
        for _ in range(self.n_substeps):
            self.scene.step(update_visualizer=False)

    def get_walker_positions(self):
        """Get walker positions for all environments.

        Returns
        -------
        np.ndarray, shape (n_envs, 3)
        """
        pos = self.walker.get_pos()
        return pos.cpu().numpy() if hasattr(pos, 'cpu') else np.asarray(pos)

    def get_walker_quats(self):
        """Get walker quaternions for all environments.

        Returns
        -------
        np.ndarray, shape (n_envs, 4)
        """
        quat = self.walker.get_quat()
        return quat.cpu().numpy() if hasattr(quat, 'cpu') else np.asarray(quat)

    def update_cameras(self, positions, headings):
        """Update all per-env cameras to follow their walkers.

        Parameters
        ----------
        positions : np.ndarray, shape (n_envs, 3)
        headings : np.ndarray, shape (n_envs,)
        """
        cam_positions = np.stack([
            positions[:, 0],
            positions[:, 1],
            positions[:, 2] + WALKER_CAMERA_HEIGHT,
        ], axis=-1)  # (n_envs, 3)

        look_dist = 1.0
        looktats = np.stack([
            positions[:, 0] + look_dist * np.cos(headings),
            positions[:, 1] + look_dist * np.sin(headings),
            positions[:, 2] + WALKER_CAMERA_HEIGHT - 0.1,
        ], axis=-1)  # (n_envs, 3)

        if self.camera is not None:
            # BatchRenderer: single vectorized call
            self.camera.set_pose(pos=cam_positions, lookat=looktats)
        else:
            # Rasterizer: per-env loop
            for i in range(self.n_envs):
                self.cameras[i].set_pose(pos=cam_positions[i], lookat=looktats[i])

    def render_all(self):
        """Render egocentric views for all environments.

        Returns
        -------
        np.ndarray, shape (n_envs, H, W, 3), dtype uint8
        """
        if self.camera is not None:
            # BatchRenderer: one call returns (n_envs, H, W, 3) CUDA tensor
            rgb = self.camera.render(rgb=True, depth=False, segmentation=False)[0]
            return rgb.cpu().numpy().astype(np.uint8)
        else:
            # Rasterizer: sequential per-env loop
            res = self.camera_resolution
            images = np.empty((self.n_envs, res, res, 3), dtype=np.uint8)
            for i in range(self.n_envs):
                result = self.cameras[i].render(rgb=True, depth=False, segmentation=False)
                rgb = result[0]
                if hasattr(rgb, 'cpu'):
                    rgb = rgb.cpu().numpy()
                images[i] = np.asarray(rgb, dtype=np.uint8)
            return images

    def render_single(self, env_idx):
        """Render egocentric view for a single environment.

        Returns
        -------
        np.ndarray, shape (H, W, 3), dtype uint8
        """
        if self.camera is not None:
            # BatchRenderer: render all, extract one
            rgb = self.camera.render(rgb=True, depth=False, segmentation=False)[0]
            return rgb[env_idx].cpu().numpy().astype(np.uint8)
        else:
            # Rasterizer: render specific camera
            result = self.cameras[env_idx].render(rgb=True, depth=False, segmentation=False)
            rgb = result[0]
            if hasattr(rgb, 'cpu'):
                rgb = rgb.cpu().numpy()
            return np.asarray(rgb, dtype=np.uint8)

    def reset_env(self, env_idx):
        """Reset a single environment to its build-time state."""
        idx_tensor = torch.tensor([env_idx], dtype=torch.int32)
        self.scene.reset(envs_idx=idx_tensor)


# ---------------------------------------------------------------------------
# Phase 6: Batched Gym-like Vectorized Environment
# ---------------------------------------------------------------------------

class BatchGenesisMemoryMazeEnv:
    """Vectorized Memory Maze using a single batched Genesis scene.

    NOT a ``gym.Env`` — exposes a custom vectorized interface where
    ``reset()`` and ``step()`` operate on all N environments simultaneously.

    Auto-reset convention (matches TorchBeast): when ``done=True`` for env *i*,
    that env is immediately reset and the returned observation is the **new**
    episode's first frame.
    """

    def __init__(
        self,
        n_envs,
        maze_size=9,
        seed=None,
        camera_resolution=64,
        good_visibility=False,
        control_freq=DEFAULT_CONTROL_FREQ,
    ):
        assert gs is not None, "Genesis not installed"

        cfg = MAZE_CONFIGS.get(maze_size, (3, 250, 6, 5))
        n_targets = cfg[0]
        time_limit = cfg[1]
        max_rooms = cfg[2]
        room_max_size = cfg[3]

        self._n_envs = n_envs
        self._maze_size = maze_size
        self._n_targets = n_targets
        self._camera_resolution = camera_resolution

        control_timestep = 1.0 / control_freq
        z_height = 0.4 if good_visibility else 1.5
        target_height = 0.5 if good_visibility else -0.6
        self._target_height = target_height
        self._max_steps = int(time_limit * control_freq)
        self._z_height = z_height
        self._max_rooms = max_rooms
        self._room_max_size = room_max_size
        self._xy_scale = 2.0

        # Per-env RNGs (CPU, for maze generation)
        base_seed = seed if seed is not None else 0
        self._rngs = [
            np.random.RandomState(base_seed + i) for i in range(n_envs)
        ]

        # Initialize Genesis if needed
        if not gs._initialized:
            backend = gs.cuda if _BATCH_RENDERER_AVAILABLE else gs.cpu
            gs.init(backend=backend, logging_level='warning')

        # Build batched scene
        self._scene = BatchGenesisMazeScene(
            n_envs=n_envs,
            maze_size=maze_size,
            n_targets=n_targets,
            xy_scale=self._xy_scale,
            z_height=z_height,
            camera_resolution=camera_resolution,
            control_timestep=control_timestep,
            max_rooms=max_rooms,
            room_max_size=room_max_size,
            target_height_above_ground=target_height,
        )
        self._scene.build()

        # Batched episode state
        self._step_counts = np.zeros(n_envs, dtype=np.int32)
        self._walker_headings = np.zeros(n_envs, dtype=np.float64)
        self._current_target_ix = np.zeros(n_envs, dtype=np.int64)
        self._targets_obtained = np.zeros(n_envs, dtype=np.int32)
        # (n_envs, n_targets, 3) — world positions of placed targets
        self._target_positions = np.zeros(
            (n_envs, n_targets, 3), dtype=np.float64
        )

    @property
    def n_envs(self):
        return self._n_envs

    def reset(self):
        """Reset all environments. Returns observations (n_envs, H, W, 3)."""
        for i in range(self._n_envs):
            self._reset_single_env(i)

        # Update cameras and render
        positions = self._scene.get_walker_positions()
        self._scene.update_cameras(positions, self._walker_headings)
        images = self._scene.render_all()

        # Draw target borders
        for i in range(self._n_envs):
            self._draw_border(images[i], self._current_target_ix[i])

        return images

    def step(self, actions):
        """Execute one step for all environments.

        Parameters
        ----------
        actions : array-like, shape (n_envs,)
            Discrete action indices (0-5) per environment.

        Returns
        -------
        obs : np.ndarray, shape (n_envs, H, W, 3), dtype uint8
        rewards : np.ndarray, shape (n_envs,), dtype float32
        dones : np.ndarray, shape (n_envs,), dtype bool
        infos : list[dict]
        """
        actions = np.asarray(actions, dtype=np.int64)

        # 1. Map discrete actions to continuous [roll, steer]
        continuous = np.array([ACTION_SET[a] for a in actions])  # (n_envs, 2)

        # 2. Compute heading-rotated forces for all envs
        forces = np.zeros((self._n_envs, 6), dtype=np.float64)
        cos_h = np.cos(self._walker_headings)
        sin_h = np.sin(self._walker_headings)
        roll_forces = ROLL_GEAR * continuous[:, 0]  # (n_envs,)
        forces[:, 0] = roll_forces * cos_h
        forces[:, 1] = roll_forces * sin_h
        forces[:, 5] = STEER_GEAR * continuous[:, 1]

        # 3. Apply forces and step physics
        self._scene.apply_actions_batched(forces.astype(np.float32))
        self._scene.step()

        # 4. Extract headings from quaternions (vectorized)
        quats = self._scene.get_walker_quats()  # (n_envs, 4)
        self._walker_headings = np.arctan2(
            2 * (quats[:, 0] * quats[:, 3] + quats[:, 1] * quats[:, 2]),
            1 - 2 * (quats[:, 2] ** 2 + quats[:, 3] ** 2),
        )

        # 5. Check target contacts (vectorized distance computation)
        positions = self._scene.get_walker_positions()  # (n_envs, 3)
        rewards = np.zeros(self._n_envs, dtype=np.float32)
        for i in range(self._n_envs):
            for t in range(self._n_targets):
                tpos = self._target_positions[i, t]
                if tpos[2] < -5:  # hidden
                    continue
                dist = np.linalg.norm(positions[i, :2] - tpos[:2])
                if dist < TARGET_ACTIVATION_GAP and t == self._current_target_ix[i]:
                    rewards[i] = 1.0
                    self._targets_obtained[i] += 1
                    self._pick_new_target(i)

        # 6. Update cameras and render
        self._scene.update_cameras(positions, self._walker_headings)
        images = self._scene.render_all()

        # 7. Draw borders
        for i in range(self._n_envs):
            self._draw_border(images[i], self._current_target_ix[i])

        # 8. Check dones and auto-reset
        self._step_counts += 1
        dones = self._step_counts >= self._max_steps

        infos = [{} for _ in range(self._n_envs)]
        for i in range(self._n_envs):
            if dones[i]:
                infos[i]['TimeLimit.truncated'] = True
                infos[i]['targets_obtained'] = int(self._targets_obtained[i])
                # Auto-reset: reset env and return new episode's first obs
                self._reset_single_env(i)
                # Re-render this env after reset
                wp = self._scene.get_walker_positions()
                self._scene.update_cameras(wp, self._walker_headings)
                images[i] = self._scene.render_single(i)
                self._draw_border(images[i], self._current_target_ix[i])

        return images, rewards, dones, infos

    def _reset_single_env(self, env_idx):
        """Reset a single environment: new maze, walker, targets."""
        rng = self._rngs[env_idx]

        # Reset Genesis state for this env
        self._scene.reset_env(env_idx)

        # Generate new maze
        seed = rng.randint(2147483648)
        maze = labmaze.RandomMaze(
            height=self._scene.outer_size,
            width=self._scene.outer_size,
            max_rooms=self._max_rooms,
            room_min_size=3,
            room_max_size=self._room_max_size,
            spawns_per_room=1,
            objects_per_room=1,
            random_seed=seed,
        )

        # Configure walls
        wall_segments = extract_wall_segments(maze, self._xy_scale, self._z_height)
        self._scene.configure_walls_for_env(env_idx, wall_segments)

        # Place walker
        spawn_positions = extract_positions(maze, 'P', self._xy_scale)
        if spawn_positions:
            spawn_pos = spawn_positions[rng.randint(len(spawn_positions))]
        else:
            spawn_pos = np.array([0.0, 0.0, 0.0])

        heading = rng.uniform(0, 2 * math.pi)
        self._walker_headings[env_idx] = heading
        self._scene.set_walker_pose(env_idx, spawn_pos, heading)

        # Place targets
        target_positions = extract_positions(maze, 'G', self._xy_scale)
        rng.shuffle(target_positions)
        target_z = TARGET_RADIUS + self._target_height
        for t in range(self._n_targets):
            if t < len(target_positions):
                tpos = target_positions[t]
                world_pos = np.array([tpos[0], tpos[1], target_z])
                self._scene.set_target_pos(env_idx, t, world_pos)
                self._target_positions[env_idx, t] = world_pos
            else:
                self._scene.hide_target(env_idx, t)
                self._target_positions[env_idx, t] = [0.0, 0.0, BATCH_HIDDEN_Z]

        # Reset counters
        self._step_counts[env_idx] = 0
        self._targets_obtained[env_idx] = 0
        self._current_target_ix[env_idx] = rng.randint(self._n_targets)

    def _pick_new_target(self, env_idx):
        """Pick a new random target for an environment (not the current one)."""
        rng = self._rngs[env_idx]
        current = self._current_target_ix[env_idx]
        while True:
            ix = rng.randint(self._n_targets)
            if ix != current:
                self._current_target_ix[env_idx] = ix
                break

    def _draw_border(self, img, target_ix):
        """Draw target color border on an image (in-place)."""
        color = TARGET_COLORS[target_ix]
        B = int(2 * math.sqrt(self._camera_resolution / 64))
        border_color = (color * 255 * 0.7).astype(np.uint8)
        img[:, :B] = border_color
        img[:, -B:] = border_color
        img[:B, :] = border_color
        img[-B:, :] = border_color

    def close(self):
        """Clean up resources."""
        pass


# ---------------------------------------------------------------------------
# Gym environment registration helper
# ---------------------------------------------------------------------------

def register_genesis_envs():
    """Register Genesis-backed Memory Maze environments with gym."""
    if gym is None:
        return

    from functools import partial
    from gym.envs.registration import register

    for key, (maze_size, (n_targets, time_limit, _, _)) in {
        '9x9': (9, MAZE_CONFIGS[9]),
        '11x11': (11, MAZE_CONFIGS[11]),
        '13x13': (13, MAZE_CONFIGS[13]),
        '15x15': (15, MAZE_CONFIGS[15]),
    }.items():
        register(
            id=f'MemoryMaze-{key}-Genesis-v0',
            entry_point='memory_maze.genesis_backend:GenesisMemoryMazeEnv',
            kwargs=dict(maze_size=maze_size, n_targets=n_targets, time_limit=time_limit),
        )
