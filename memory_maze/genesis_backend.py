"""
Genesis physics backend for Memory Maze.

Replaces the MuJoCo/dm_control stack with Genesis for GPU-accelerated
physics and rendering. Implements the same gym.Env interface.
"""

import logging
import math
import os
import time as _time
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
WALKER_CAMERA_HEIGHT = 0.7   # above ball center (matching MuJoCo's 0.9m above ground)
WALKER_CAMERA_FORWARD_OFFSET = 0.0  # at ball center — 0.2m clearance to walls (prevents see-through in raycast)

# Actuator params — tuned to match MuJoCo's effective dynamics.
# MuJoCo uses torque on a rolling hinge (friction-coupled to ground);
# Genesis uses direct translational force on a free sphere.
# To compensate, we use low surface friction + translational DOF damping.
# Terminal velocity: v_ss = |ROLL_GEAR| / TRANS_DAMPING = 400/200 = 2.0 m/s (matches MuJoCo)
# Terminal steer rate: STEER_GEAR / STEER_DAMPING ≈ 1.28 rad/s (matches MuJoCo)
ROLL_GEAR = -400.0      # translational force magnitude (gear * roll_cmd)
STEER_GEAR = 30.0       # steer torque magnitude (gear * steer_cmd)
TRANS_DAMPING = 200.0   # translational damping on tx, ty (provides deceleration)
ROLL_DAMPING = 5.0      # rotational damping on rx, ry
STEER_DAMPING = 23.4    # rotational damping on rz (30/23.4 ≈ 1.28 rad/s)
WALKER_FRICTION = 0.01   # minimum allowed by Genesis; uses max(mu_a, mu_b) for contact pairs
FLOOR_FRICTION = 0.01    # minimum allowed; effective walker-floor friction = 0.01

# MuJoCo camera
CAMERA_FOV = 80  # fovy from XML

# Target detection
TARGET_RADIUS = 0.6
TARGET_ACTIVATION_GAP = WALKER_RADIUS + TARGET_RADIUS  # 0.8m center-to-center, matches MuJoCo contact detection
# Spheres receive light from all directions (curved surface catches ~3 of 6 cardinal lights),
# so they appear brighter than flat walls. Scale down to compensate.
TARGET_COLOR_SCALE = 0.5

# Timing
DEFAULT_CONTROL_FREQ = 4.0
DEFAULT_PHYSICS_TIMESTEP = 0.005
DEFAULT_CONTROL_TIMESTEP = 1.0 / DEFAULT_CONTROL_FREQ  # 0.25s

# Texture support
N_WALL_GROUPS = 9   # '0'-'8' spatial blocks from TextMazeVaryingWalls

# ---------------------------------------------------------------------------
# Lighting — cardinal directional lights for direction-independent walls.
#
# BatchRenderer (Madrona) has ambient hardcoded in shaders (raised to 0.3
# in our C++ fix; was 0.05).  We add 6 directional lights from axis directions
# so every wall face is lit regardless of heading.  The C++ clamp fix prevents
# uint8 overflow, so any intensity is safe.
#
# Rasterizer (pyrender) has MAX_N_LIGHTS=4, so it only gets the 4 horizontal
# lights.  Vertical lighting is handled by its ambient_light (set to 0.5).
# ---------------------------------------------------------------------------
_LIGHT_INTENSITY = 1.5
_LIGHT_DIRS_HORIZ = [
    (1, 0, 0), (-1, 0, 0),   # +X, -X  (N/S walls)
    (0, 1, 0), (0, -1, 0),   # +Y, -Y  (E/W walls)
]
_LIGHT_DIRS_VERT = [
    (0, 0, -1), (0, 0, 1),   # -Z, +Z  (floor / ceiling)
]
_LIGHT_DIRS_ALL = _LIGHT_DIRS_HORIZ + _LIGHT_DIRS_VERT
# VisOptions.lights format (for Rasterizer — max 4 directional lights)
_CARDINAL_LIGHTS_VIS = [
    {"type": "directional", "dir": d, "color": (1.0, 1.0, 1.0), "intensity": _LIGHT_INTENSITY}
    for d in _LIGHT_DIRS_HORIZ
]
# scene.add_light() kwargs (for BatchRenderer — no light limit)
# castshadow=False prevents shadow acne (dark speckle noise on walls at close range).
# The maze has uniform ambient + directional lighting — shadows add noise, not value.
_CARDINAL_LIGHTS_BATCH = [
    {"pos": (0, 0, 10), "dir": d, "directional": True,
     "intensity": _LIGHT_INTENSITY, "color": (1.0, 1.0, 1.0),
     "castshadow": False}
    for d in _LIGHT_DIRS_ALL
]


def _to_numpy(x):
    """Convert a Genesis tensor or numpy array to numpy ndarray."""
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)


def _compute_walls_per_group(maze_size):
    """Compute wall capacity per texture group for a given maze size.

    The maze grid (outer = maze_size+2) is divided into a 3x3 block grid by
    ``TextMazeVaryingWalls._block_variations()``.  Block spans are unequal
    (integer division remainder), so we allocate every group to the largest
    block area.  This keeps all groups equal-sized, which allows shuffling
    the texture-to-region mapping each episode (matching MuJoCo behaviour).
    """
    outer = maze_size + 2
    # 3x3 block spans — last block gets the remainder
    max_span = max((b + 1) * outer // 3 - b * outer // 3 for b in range(3))
    return max_span * max_span


def _max_walls(maze_size):
    """Total pre-allocated walls = per_group * N_WALL_GROUPS."""
    return _compute_walls_per_group(maze_size) * N_WALL_GROUPS
BOX_OBJ_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'textured_box.obj')

# Maze config per size (maze_size -> (n_targets, time_limit, max_rooms, room_max_size))
MAZE_CONFIGS = {
    9:  (3, 250, 6, 5),
    11: (4, 500, 6, 5),
    13: (5, 750, 6, 5),
    15: (6, 1000, 9, 3),
}

# ---------------------------------------------------------------------------
# Texture loading utilities
# ---------------------------------------------------------------------------

def _load_texture_rgb(path):
    """Load a PNG texture as an RGB numpy array (handles palette-mode PNGs)."""
    from PIL import Image as _PILImage
    img = _PILImage.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def _load_wall_textures():
    """Load all style_01 wall textures as {name: rgb_array} dict."""
    import labmaze.assets as labmaze_assets
    paths = labmaze_assets.get_wall_texture_paths('style_01')
    return {name: _load_texture_rgb(p) for name, p in paths.items()}


def _load_floor_texture():
    """Load the style_01 blue floor texture as an RGB array."""
    import labmaze.assets as labmaze_assets
    paths = labmaze_assets.get_floor_texture_paths('style_01')
    return _load_texture_rgb(paths['blue'])


def _apply_block_variations(maze):
    """Replace '*' wall chars with '0'-'8' in a 3x3 block pattern.

    Same logic as TextMazeVaryingWalls._block_variations() from maze.py.
    """
    nblocks = 3
    n = maze.entity_layer.shape[0]
    ivar = 0
    for i in range(nblocks):
        for j in range(nblocks):
            i_from = i * n // nblocks
            i_to = (i + 1) * n // nblocks
            j_from = j * n // nblocks
            j_to = (j + 1) * n // nblocks
            grid = maze.entity_layer
            rows, cols = np.where(grid[i_from:i_to, j_from:j_to] == '*')
            grid[rows + i_from, cols + j_from] = str(ivar)
            ivar += 1


# ---------------------------------------------------------------------------
# Wall segment extraction (uses dm_control.locomotion.arenas.covering)
# ---------------------------------------------------------------------------

WallSegment = namedtuple('WallSegment', ['pos', 'half_size', 'wall_char'])


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
            segments.append(WallSegment(pos=pos, half_size=half_size, wall_char=wc))

    return segments


def extract_wall_cells(maze, xy_scale=2.0, z_height=1.5):
    """Extract one WallSegment per wall cell from the maze grid.

    Unlike ``extract_wall_segments`` which merges adjacent wall cells into
    larger boxes (variable sizes), this returns one identically-sized box per
    wall cell.  All boxes have half_size = (xy_scale/2, xy_scale/2, z_height/2)
    so they can share a single pre-allocated Genesis entity size.
    """
    x_offset = (maze.width - 1) / 2.0
    y_offset = (maze.height - 1) / 2.0
    half = np.array([xy_scale / 2.0, xy_scale / 2.0, z_height / 2.0])

    cells = []
    for row in range(maze.height):
        for col in range(maze.width):
            c = maze.entity_layer[row, col]
            if c not in (' ', 'P', 'G'):
                # Wall cell — compute world position (same coord transform as extract_wall_segments)
                pos = np.array([
                    (col - x_offset) * xy_scale,
                    -(row - y_offset) * xy_scale,
                    z_height / 2.0,
                ])
                cells.append(WallSegment(pos=pos, half_size=half, wall_char=c))
    return cells


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
# Shared base class for single-env and batch-env scenes
# ---------------------------------------------------------------------------

# Unified hidden depth for both single and batch modes
HIDDEN_Z = -100.0


class _BaseMazeScene:
    """Shared entity construction and configuration for single/batch scenes.

    Subclasses override ``_setup_camera()`` to handle renderer-specific
    camera creation.  The ``n_envs`` parameter controls batch mode:
    n_envs=0 for single-env (no batch dimension), n_envs>0 for batch.
    """

    def __init__(
        self,
        *,
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
        use_textures=True,
        texture_seed=None,
        n_envs=0,
        max_collision_pairs=None,
    ):
        if gs is None:
            raise ImportError("Genesis is not installed. Install with: pip install genesis-world")

        self._n_envs = n_envs
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
        self.use_textures = use_textures

        self.n_substeps = max(1, int(round(control_timestep / physics_timestep)))
        self.outer_size = maze_size + 2

        _log = logging.getLogger("genesis_backend")
        _t0 = _time.monotonic()
        def _elapsed():
            return f"{_time.monotonic() - _t0:.1f}s"

        # --- Renderer ---
        if _use_batch_renderer():
            renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
        else:
            renderer = gs.renderers.Rasterizer()
        _log.info("[%s] Creating scene (renderer=%s, n_envs=%d)",
                  _elapsed(), type(renderer).__name__, n_envs)

        # --- Scene options ---
        rigid_opts = dict(enable_collision=True, enable_joint_limit=True)
        if max_collision_pairs is not None:
            rigid_opts['max_collision_pairs'] = max_collision_pairs

        vis_opts = dict(
            show_world_frame=False,
            ambient_light=(0.5, 0.5, 0.5),
            lights=_CARDINAL_LIGHTS_VIS,
        )
        if n_envs > 0:
            vis_opts['env_separate_rigid'] = not _use_batch_renderer()

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=physics_timestep,
                substeps=1,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(**rigid_opts),
            vis_options=gs.options.VisOptions(**vis_opts),
            renderer=renderer,
        )

        # BatchRenderer requires explicit lights (Rasterizer uses VisOptions.lights)
        if _use_batch_renderer():
            for ldef in _CARDINAL_LIGHTS_BATCH:
                self.scene.add_light(**ldef)

        # --- Floor ---
        if use_textures:
            floor_tex_array = _load_floor_texture()
            floor_surface = gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(image_array=floor_tex_array),
            )
        else:
            floor_surface = gs.surfaces.Default(color=(0.4, 0.5, 0.6, 1.0))
        self.floor = self.scene.add_entity(
            morph=gs.morphs.Plane(pos=(0, 0, 0)),
            material=gs.materials.Rigid(friction=FLOOR_FRICTION),
            surface=floor_surface,
        )
        _log.info("[%s] Floor entity added", _elapsed())

        # --- Pre-allocate wall entities ---
        walls_per_group = _compute_walls_per_group(maze_size)
        max_walls = walls_per_group * N_WALL_GROUPS
        _batched = n_envs > 0
        _log.info("[%s] Adding %d wall entities (%d groups x %d, textures=%s)",
                  _elapsed(), max_walls, N_WALL_GROUPS, walls_per_group, use_textures)
        if use_textures:
            all_wall_textures = _load_wall_textures()
            texture_names = list(all_wall_textures.keys())
            tex_rng = np.random.RandomState(texture_seed)

            self._wall_groups = {}
            for group_idx in range(N_WALL_GROUPS):
                char = str(group_idx)
                tex_name = tex_rng.choice(texture_names)
                tex_array = all_wall_textures[tex_name]
                surface = gs.surfaces.Default(
                    diffuse_texture=gs.textures.ImageTexture(image_array=tex_array),
                )
                group_entities = []
                mesh_kwargs = dict(
                    file=BOX_OBJ_PATH,
                    pos=(0, 0, HIDDEN_Z),
                    scale=(xy_scale, xy_scale, z_height),
                    fixed=True,
                    convexify=False,
                    decimate=False,
                )
                if _batched:
                    mesh_kwargs['batch_fixed_verts'] = True
                for _ in range(walls_per_group):
                    wall = self.scene.add_entity(
                        morph=gs.morphs.Mesh(**mesh_kwargs),
                        material=gs.materials.Rigid(friction=0.5),
                        surface=surface,
                    )
                    group_entities.append(wall)
                self._wall_groups[char] = group_entities
            self.wall_entities = [e for g in self._wall_groups.values() for e in g]
        else:
            self._wall_groups = None
            self.wall_entities = []
            for _ in range(max_walls):
                wall = self.scene.add_entity(
                    morph=gs.morphs.Box(
                        pos=(0, 0, HIDDEN_Z),
                        size=(xy_scale, xy_scale, z_height),
                        fixed=True,
                    ),
                    material=gs.materials.Rigid(friction=0.5),
                    surface=gs.surfaces.Default(color=(0.8, 0.7, 0.3, 1.0)),
                )
                self.wall_entities.append(wall)

        _log.info("[%s] All %d wall entities added", _elapsed(), len(self.wall_entities))

        # --- Walker (rolling ball) ---
        # visualization=False: egocentric camera sits at ball center (offset=0.0),
        # so the sphere must be invisible to avoid raycast self-intersection.
        # Physics/collision still work — only visual geometry is skipped.
        self.walker = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0, 0, WALKER_RADIUS),
                radius=WALKER_RADIUS,
                fixed=False,
                visualization=False,
            ),
            material=gs.materials.Rigid(
                friction=WALKER_FRICTION,
                rho=WALKER_TOTAL_MASS / ((4.0 / 3.0) * math.pi * WALKER_RADIUS ** 3),
            ),
            surface=gs.surfaces.Default(color=(0.757, 0.757, 0.757, 1.0)),
        )
        _log.info("[%s] Walker entity added", _elapsed())

        # --- Targets (non-colliding colored spheres) ---
        self.target_entities = []
        self.target_colors = list(TARGET_COLORS)
        for i in range(n_targets):
            color = self.target_colors[i]
            target = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    pos=(0, 0, HIDDEN_Z),
                    radius=TARGET_RADIUS,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Default(
                    color=(float(color[0] * TARGET_COLOR_SCALE), float(color[1] * TARGET_COLOR_SCALE), float(color[2] * TARGET_COLOR_SCALE), 1.0),
                ),
            )
            self.target_entities.append(target)
        _log.info("[%s] %d target entities added", _elapsed(), n_targets)

        # --- Camera (subclass hook) ---
        self._setup_camera()

        self._built = False
        _log.info("[%s] Cameras added. Scene __init__ complete, ready to build.", _elapsed())

    @property
    def n_envs(self):
        """Number of parallel environments (0 for single-env mode)."""
        return self._n_envs

    def _setup_camera(self):
        """Subclass hook: create camera(s) appropriate for the rendering mode."""
        raise NotImplementedError

    def build(self):
        """Build the Genesis scene. Must be called once before stepping."""
        _log = logging.getLogger("genesis_backend")
        _t0 = _time.monotonic()
        if self._n_envs > 0:
            _log.info("[0.0s] scene.build(n_envs=%d) starting...", self._n_envs)
            self.scene.build(n_envs=self._n_envs)
        else:
            self.scene.build()
        _log.info("[%.1fs] scene.build() complete", _time.monotonic() - _t0)
        self._built = True

        # Configure walker DOF damping after build
        # DOFs: [tx, ty, tz, rx, ry, rz]
        # v_ss = |ROLL_GEAR| / TRANS_DAMPING = 400/200 = 2.0 m/s (matches MuJoCo)
        self.walker.set_dofs_damping(np.array([
            TRANS_DAMPING, TRANS_DAMPING, 0.0,
            ROLL_DAMPING, ROLL_DAMPING, STEER_DAMPING,
        ]))
        _log.info("[%.1fs] Walker damping configured. Build done.", _time.monotonic() - _t0)

    def shuffled_wall_groups(self, rng):
        """Return a shuffled copy of wall_groups (does not mutate shared state)."""
        if self._wall_groups is None:
            return None
        keys = sorted(self._wall_groups.keys())
        groups = [self._wall_groups[k] for k in keys]
        rng.shuffle(groups)
        return dict(zip(keys, groups))

    def _configure_walls_for_env(self, env_idx, wall_segments, wall_groups=None):
        """Position the wall pool for a single environment (or the only env).

        Uses envs_idx when n_envs > 0, plain set_pos when n_envs == 0.
        """
        _batched = self._n_envs > 0
        idx_tensor = torch.tensor([env_idx], dtype=torch.int32) if _batched else None
        hidden_pos = np.array([0.0, 0.0, HIDDEN_Z], dtype=np.float32)
        groups = wall_groups or self._wall_groups

        def _set_wall_pos(entity, pos_array):
            if _batched:
                entity.set_pos(np.array(pos_array, dtype=np.float32), envs_idx=idx_tensor)
            else:
                entity.set_pos(np.array(pos_array, dtype=np.float32))

        if groups is not None:
            group_usage = {char: 0 for char in groups}
            for seg in wall_segments:
                char = seg.wall_char
                if char not in groups:
                    continue
                group = groups[char]
                idx = group_usage[char]
                if idx < len(group):
                    _set_wall_pos(group[idx], seg.pos)
                    group_usage[char] += 1
            for char, group in groups.items():
                for i in range(group_usage[char], len(group)):
                    _set_wall_pos(group[i], hidden_pos)
        else:
            for i, wall_entity in enumerate(self.wall_entities):
                if i < len(wall_segments):
                    _set_wall_pos(wall_entity, wall_segments[i].pos)
                else:
                    _set_wall_pos(wall_entity, hidden_pos)

    def _hide_target(self, env_idx, target_idx):
        """Move a target underground (invisible)."""
        pos = np.array([0.0, 0.0, HIDDEN_Z], dtype=np.float32)
        if self._n_envs > 0:
            idx_tensor = torch.tensor([env_idx], dtype=torch.int32)
            self.target_entities[target_idx].set_pos(pos, envs_idx=idx_tensor)
        else:
            self.target_entities[target_idx].set_pos(pos)

    def _show_target(self, env_idx, target_idx, position):
        """Move a target to given position (visible)."""
        pos = np.array(position, dtype=np.float32)
        if self._n_envs > 0:
            idx_tensor = torch.tensor([env_idx], dtype=torch.int32)
            self.target_entities[target_idx].set_pos(pos, envs_idx=idx_tensor)
        else:
            self.target_entities[target_idx].set_pos(pos)


# ---------------------------------------------------------------------------
# Phase 1: Single-env Scene (inherits from _BaseMazeScene)
# ---------------------------------------------------------------------------

class GenesisMazeScene(_BaseMazeScene):
    """Manages the Genesis scene with walls, floor, walker, targets, and camera.

    Single-env mode: n_envs=0, no batch dimension on tensors.
    """

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
        use_textures=True,
        texture_seed=None,
    ):
        super().__init__(
            maze_size=maze_size, n_targets=n_targets, xy_scale=xy_scale,
            z_height=z_height, camera_resolution=camera_resolution,
            control_timestep=control_timestep, physics_timestep=physics_timestep,
            max_rooms=max_rooms, room_min_size=room_min_size,
            room_max_size=room_max_size,
            target_height_above_ground=target_height_above_ground,
            use_textures=use_textures, texture_seed=texture_seed,
            n_envs=0,
        )
        # State tracking (single-env specific)
        self._maze = None
        self._target_world_positions = []
        self._walker_heading = 0.0

    def _setup_camera(self):
        """Single-env: one camera, no env_idx."""
        self.camera = self.scene.add_camera(
            res=(self.camera_resolution, self.camera_resolution),
            pos=(0, 0, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
            lookat=(0, 1, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
            fov=CAMERA_FOV,
            near=0.05,
            far=50.0,
        )

    def reset(self, rng):
        """Reset the scene: regenerate maze, place walker and targets."""
        assert self._built, "Scene must be built before reset"

        # Create maze once, then regenerate on each reset (matches MuJoCo lifecycle)
        if self._maze is None:
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
        self._maze.regenerate()
        if self.use_textures:
            _apply_block_variations(self._maze)
            shuffled = self.shuffled_wall_groups(rng)
        else:
            shuffled = None

        # Configure walls (one entity per maze wall cell, uniform size)
        wall_segments = extract_wall_cells(self._maze, self.xy_scale, self.z_height)
        self._configure_walls_for_env(0, wall_segments, wall_groups=shuffled)

        # Place walker at random spawn
        spawn_positions = extract_positions(self._maze, 'P', self.xy_scale)
        if spawn_positions:
            spawn_idx = rng.randint(len(spawn_positions))
            spawn_pos = spawn_positions[spawn_idx]
        else:
            spawn_pos = np.array([0.0, 0.0, 0.0])

        self._walker_heading = rng.uniform(0, 2 * math.pi)
        self.walker.set_pos(np.array([spawn_pos[0], spawn_pos[1], WALKER_RADIUS]))
        qw = math.cos(self._walker_heading / 2)
        qz = math.sin(self._walker_heading / 2)
        self.walker.set_quat(np.array([qw, 0.0, 0.0, qz]))
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
                self._show_target(0, i, world_pos)
                self._target_world_positions.append(world_pos)
            else:
                self._hide_target(0, i)
                self._target_world_positions.append(np.array([0.0, 0.0, HIDDEN_Z]))

        # Update camera to walker position
        self._update_camera()

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
        # Update heading from z-axis angular velocity (DOF index 5 = rz).
        # This avoids quaternion yaw extraction which is unreliable for a
        # rolling sphere (roll/pitch rotations corrupt the yaw component).
        # Negated because MuJoCo's steer joint axis is (0,0,-1) while
        # Genesis rz DOF uses (0,0,+1), so omega_z has opposite sign.
        vel = self.walker.get_dofs_velocity()
        v = _to_numpy(vel)
        omega_z = float(v[5])
        self._walker_heading -= omega_z * self.control_timestep
        self._update_camera()

    def _get_walker_heading(self):
        """Get walker heading from tracked state.

        We track heading via z-axis angular velocity integration rather than
        extracting yaw from the sphere's quaternion.  A rolling ball accumulates
        rotation around its roll/pitch axes as it moves, so quaternion-based yaw
        is unreliable and can flip 180° after enough forward rolling.
        """
        return self._walker_heading

    def _update_camera(self):
        """Update camera position/orientation to follow walker."""
        pos = self.walker.get_pos()
        walker_pos = _to_numpy(pos)
        heading = self._get_walker_heading()

        # Camera above and slightly forward of ball, matching MuJoCo
        cam_x = walker_pos[0] + WALKER_CAMERA_FORWARD_OFFSET * math.cos(heading)
        cam_y = walker_pos[1] + WALKER_CAMERA_FORWARD_OFFSET * math.sin(heading)
        cam_z = walker_pos[2] + WALKER_CAMERA_HEIGHT
        cam_pos = np.array([cam_x, cam_y, cam_z])
        # Look direction: forward along heading, slightly downward
        look_dist = 1.0
        lookat = np.array([
            cam_x + look_dist * math.cos(heading),
            cam_y + look_dist * math.sin(heading),
            cam_z - 0.1,
        ])
        self.camera.set_pose(pos=cam_pos, lookat=lookat, up=(0, 0, 1))

    def get_walker_position(self):
        """Get walker [x, y, z] position as numpy array."""
        return _to_numpy(self.walker.get_pos())

    def render_egocentric(self):
        """Render egocentric camera view. Returns uint8 numpy [H, W, 3]."""
        result = self.camera.render(rgb=True, depth=False, segmentation=False, force_render=True)
        # render() returns a tuple: (rgb, depth, segmentation, normal)
        return np.asarray(_to_numpy(result[0]), dtype=np.uint8)

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
        self._hide_target(0, index)
        self._target_world_positions[index] = np.array([0.0, 0.0, HIDDEN_Z])

    def show_target(self, index, position):
        """Move target to given position (visible)."""
        self._show_target(0, index, position)
        self._target_world_positions[index] = position.copy()


# ---------------------------------------------------------------------------
# Shared target picking (free function)
# ---------------------------------------------------------------------------

def _pick_new_target(rng, current_ix, target_positions, walker_pos,
                     n_targets, activation_gap=TARGET_ACTIVATION_GAP,
                     max_attempts=100):
    """Pick next target index with bounded iteration.

    Shared by both single-env and batch-env wrappers. Returns a new target
    index that is different from current_ix and far enough from walker_pos.
    Falls back to ``(current_ix + 1) % n_targets`` after max_attempts.
    """
    for _ in range(max_attempts):
        candidate = rng.randint(0, n_targets)
        if candidate == current_ix:
            continue
        dist = np.linalg.norm(walker_pos[:2] - target_positions[candidate][:2])
        if dist >= activation_gap:
            return candidate
    fallback = (current_ix + 1) % n_targets
    logging.warning("_pick_new_target: all targets within activation gap, using fallback")
    return fallback


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
        physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
        use_textures=True,
        use_batch_renderer=None,
    ):
        super().__init__()
        if gs is None:
            raise ImportError("Genesis not installed. Install with: pip install genesis-world")
        if gym is None:
            raise ImportError("gym not installed. Install with: pip install 'gym>=0.21,<1.0'")


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
        self._use_batch_renderer = use_batch_renderer

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

        # Initialize Genesis (skip if already initialized).
        # Default: CPU backend (safe for forked actor subprocesses).
        # use_batch_renderer=True requires CUDA — caller must ensure
        # gs.init(backend=gs.cuda) was called before constructing this env.
        if not gs._initialized:
            if use_batch_renderer:
                gs.init(backend=gs.cuda, logging_level='warning')
            else:
                gs.init(backend=gs.cpu, logging_level='warning')

        # Build scene
        self._scene = GenesisMazeScene(
            maze_size=maze_size,
            n_targets=n_targets,
            z_height=z_height,
            camera_resolution=camera_resolution,
            control_timestep=control_timestep,
            physics_timestep=physics_timestep,
            max_rooms=max_rooms,
            room_max_size=room_max_size,
            target_height_above_ground=target_height,
            use_textures=use_textures,
            texture_seed=seed,
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
        """Pick a new random target that is not within activation distance of the walker."""
        walker_pos = self._scene.get_walker_position()
        self._current_target_ix = _pick_new_target(
            self._rng, self._current_target_ix, self._target_world_positions,
            walker_pos, self._n_targets,
        )

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
# Phase 5: Batched Scene Builder (inherits from _BaseMazeScene)
# ---------------------------------------------------------------------------

# Backward-compatible alias — batch code and tests reference this.
BATCH_HIDDEN_Z = HIDDEN_Z


class BatchGenesisMazeScene(_BaseMazeScene):
    """Manages N parallel Genesis mazes in a single scene.

    Uses ``scene.build(n_envs=N)`` so all environments share the same entity
    pool but have independent state (positions, velocities, etc.) indexed by
    ``envs_idx``.
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
        use_textures=True,
        texture_seed=None,
    ):
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        super().__init__(
            maze_size=maze_size, n_targets=n_targets, xy_scale=xy_scale,
            z_height=z_height, camera_resolution=camera_resolution,
            control_timestep=control_timestep, physics_timestep=physics_timestep,
            max_rooms=max_rooms, room_min_size=room_min_size,
            room_max_size=room_max_size,
            target_height_above_ground=target_height_above_ground,
            use_textures=use_textures, texture_seed=texture_seed,
            n_envs=n_envs, max_collision_pairs=max_collision_pairs,
        )

    def _setup_camera(self):
        """Batch-env: BatchRenderer gets one camera, Rasterizer gets N cameras."""
        if _use_batch_renderer():
            self.camera = self.scene.add_camera(
                res=(self.camera_resolution, self.camera_resolution),
                pos=(0, 0, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
                lookat=(0, 1, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
                fov=CAMERA_FOV, near=0.05, far=50.0,
            )
            self.cameras = None
        else:
            self.camera = None
            self.cameras = []
            for i in range(self.n_envs):
                cam = self.scene.add_camera(
                    res=(self.camera_resolution, self.camera_resolution),
                    pos=(0, 0, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
                    lookat=(0, 1, WALKER_CAMERA_HEIGHT + WALKER_RADIUS),
                    fov=CAMERA_FOV, near=0.05, far=50.0,
                    env_idx=i,
                )
                self.cameras.append(cam)

    def configure_walls_for_env(self, env_idx, wall_segments, wall_groups=None):
        """Position the wall pool for a single environment.

        Public alias for ``_configure_walls_for_env`` (backward compat).
        """
        self._configure_walls_for_env(env_idx, wall_segments, wall_groups=wall_groups)

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
        return _to_numpy(self.walker.get_pos())

    def get_walker_quats(self):
        """Get walker quaternions for all environments.

        Returns
        -------
        np.ndarray, shape (n_envs, 4)
        """
        return _to_numpy(self.walker.get_quat())

    def update_cameras(self, positions, headings):
        """Update all per-env cameras to follow their walkers.

        Parameters
        ----------
        positions : np.ndarray, shape (n_envs, 3)
        headings : np.ndarray, shape (n_envs,)
        """
        cam_x = positions[:, 0] + WALKER_CAMERA_FORWARD_OFFSET * np.cos(headings)
        cam_y = positions[:, 1] + WALKER_CAMERA_FORWARD_OFFSET * np.sin(headings)
        cam_z = positions[:, 2] + WALKER_CAMERA_HEIGHT
        cam_positions = np.stack([cam_x, cam_y, cam_z], axis=-1)  # (n_envs, 3)

        look_dist = 1.0
        looktats = np.stack([
            cam_x + look_dist * np.cos(headings),
            cam_y + look_dist * np.sin(headings),
            cam_z - 0.1,
        ], axis=-1)  # (n_envs, 3)

        up = np.zeros_like(cam_positions)
        up[:, 2] = 1.0
        if self.camera is not None:
            # BatchRenderer: single vectorized call
            self.camera.set_pose(pos=cam_positions, lookat=looktats, up=up)
        else:
            # Rasterizer: per-env loop
            for i in range(self.n_envs):
                self.cameras[i].set_pose(pos=cam_positions[i], lookat=looktats[i], up=up[i])

    def render_all(self):
        """Render egocentric views for all environments.

        Returns
        -------
        np.ndarray, shape (n_envs, H, W, 3), dtype uint8
        """
        if self.camera is not None:
            # BatchRenderer: one call returns (n_envs, H, W, 3) CUDA tensor
            rgb = self.camera.render(rgb=True, depth=False, segmentation=False, force_render=True)[0]
            return np.asarray(_to_numpy(rgb), dtype=np.uint8)
        else:
            # Rasterizer: sequential per-env loop
            res = self.camera_resolution
            images = np.empty((self.n_envs, res, res, 3), dtype=np.uint8)
            for i in range(self.n_envs):
                result = self.cameras[i].render(rgb=True, depth=False, segmentation=False, force_render=True)
                images[i] = np.asarray(_to_numpy(result[0]), dtype=np.uint8)
            return images

    def render_single(self, env_idx):
        """Render egocentric view for a single environment.

        Returns
        -------
        np.ndarray, shape (H, W, 3), dtype uint8
        """
        if self.camera is not None:
            # BatchRenderer: render all, extract one
            rgb = self.camera.render(rgb=True, depth=False, segmentation=False, force_render=True)[0]
            return np.asarray(_to_numpy(rgb[env_idx]), dtype=np.uint8)
        else:
            # Rasterizer: render specific camera
            result = self.cameras[env_idx].render(rgb=True, depth=False, segmentation=False, force_render=True)
            return np.asarray(_to_numpy(result[0]), dtype=np.uint8)

    def reset_env(self, env_idx):
        """Reset a single environment to its build-time state.

        Preserves the global simulation time counter ``scene._t`` so that
        resetting one env doesn't rewind the clock for all other envs.
        """
        idx_tensor = torch.tensor([env_idx], dtype=torch.int32)
        saved_t = self.scene._t
        self.scene.reset(envs_idx=idx_tensor)
        self.scene._t = saved_t


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
        use_textures=True,
        physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
    ):
        if gs is None:
            raise ImportError("Genesis not installed. Install with: pip install genesis-world")

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

        _log = logging.getLogger("genesis_backend")

        # Initialize Genesis if needed
        if not gs._initialized:
            backend = gs.cuda if _BATCH_RENDERER_AVAILABLE else gs.cpu
            _log.info("gs.init(backend=%s) ...", backend)
            gs.init(backend=backend, logging_level='warning')
            _log.info("gs.init done, device=%s", gs.device)

        # Build batched scene
        _log.info("Creating BatchGenesisMazeScene(n_envs=%d, maze=%d, textures=%s) ...",
                  n_envs, maze_size, use_textures)
        self._scene = BatchGenesisMazeScene(
            n_envs=n_envs,
            maze_size=maze_size,
            n_targets=n_targets,
            xy_scale=self._xy_scale,
            z_height=z_height,
            camera_resolution=camera_resolution,
            control_timestep=control_timestep,
            physics_timestep=physics_timestep,
            max_rooms=max_rooms,
            room_max_size=room_max_size,
            target_height_above_ground=target_height,
            use_textures=use_textures,
            texture_seed=seed,
        )
        _log.info("Scene __init__ done, calling build() ...")
        self._scene.build()
        _log.info("Scene build complete!")

        # Per-env persistent maze objects (created on first reset, then regenerated)
        self._mazes = [None] * n_envs

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

        # 4. Update headings from z-axis angular velocity (avoids quaternion
        #    yaw extraction which is unreliable for rolling spheres).
        #    Negated: MuJoCo steer axis is (0,0,-1), Genesis rz is (0,0,+1).
        vel = self._scene.walker.get_dofs_velocity()  # (n_envs, 6)
        vel_np = _to_numpy(vel)
        omega_z = vel_np[:, 5]  # rz angular velocity per env
        self._walker_headings -= omega_z * self._scene.control_timestep

        # 5. Check target contacts (vectorized distance computation)
        positions = self._scene.get_walker_positions()  # (n_envs, 3)
        rewards = np.zeros(self._n_envs, dtype=np.float32)
        # Broadcast: (n_envs, 1, 2) - (n_envs, n_targets, 2) -> (n_envs, n_targets)
        dists = np.linalg.norm(
            positions[:, np.newaxis, :2] - self._target_positions[:, :, :2],
            axis=2,
        )
        visible = self._target_positions[:, :, 2] > -5
        is_current = (np.arange(self._n_targets)[np.newaxis, :]
                      == self._current_target_ix[:, np.newaxis])
        hit = (dists < TARGET_ACTIVATION_GAP) & visible & is_current
        for i, t in zip(*np.where(hit)):
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
        if dones.any():
            if not dones.all():
                raise RuntimeError(
                    f"Batched envs desynchronized: {dones.sum()}/{self._n_envs} done. "
                    "All envs must reset together."
                )
            for i in range(self._n_envs):
                infos[i]['TimeLimit.truncated'] = True
                infos[i]['targets_obtained'] = int(self._targets_obtained[i])
                self._reset_single_env(i)

            # Batch re-render all envs once after all resets
            wp = self._scene.get_walker_positions()
            self._scene.update_cameras(wp, self._walker_headings)
            images = self._scene.render_all()
            for i in range(self._n_envs):
                self._draw_border(images[i], self._current_target_ix[i])

        return images, rewards, dones, infos

    def _reset_single_env(self, env_idx):
        """Reset a single environment: new maze, walker, targets."""
        rng = self._rngs[env_idx]

        # Reset Genesis state for this env
        self._scene.reset_env(env_idx)

        # Create maze once per env, then regenerate (matches MuJoCo lifecycle)
        if self._mazes[env_idx] is None:
            seed = rng.randint(2147483648)
            self._mazes[env_idx] = labmaze.RandomMaze(
                height=self._scene.outer_size,
                width=self._scene.outer_size,
                max_rooms=self._max_rooms,
                room_min_size=3,
                room_max_size=self._room_max_size,
                spawns_per_room=1,
                objects_per_room=1,
                random_seed=seed,
            )
        maze = self._mazes[env_idx]
        maze.regenerate()
        if self._scene.use_textures:
            _apply_block_variations(maze)
            # Shuffle texture-to-region mapping per env (doesn't mutate shared state)
            shuffled = self._scene.shuffled_wall_groups(rng)
        else:
            shuffled = None

        # Configure walls (one entity per maze wall cell, uniform size)
        wall_segments = extract_wall_cells(maze, self._xy_scale, self._z_height)
        self._scene.configure_walls_for_env(env_idx, wall_segments, wall_groups=shuffled)

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
        """Pick a new random target for an environment (not within activation distance)."""
        walker_pos = self._scene.get_walker_positions()[env_idx]  # (3,)
        self._current_target_ix[env_idx] = _pick_new_target(
            self._rngs[env_idx], int(self._current_target_ix[env_idx]),
            self._target_positions[env_idx], walker_pos, self._n_targets,
        )

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
