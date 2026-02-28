# Architecture Report: Modular Memory Maze Decomposition

**Date:** February 28, 2026
**Purpose:** Blueprint for decoupling Memory Maze from dm_control/MuJoCo to enable alternative physics engine backends (Genesis, MJX, Madrona).

---

## 1. Current Architecture Analysis

### 1.1 File Inventory

The codebase consists of 7 files totaling ~1,350 lines in `memory_maze/`:

| File | Lines | Role | Coupling Level |
|------|-------|------|----------------|
| `maze.py` | 393 | Core monolith: walker, arena, task, textures, maze gen | **Tightly coupled** |
| `tasks.py` | 153 | Factory functions, environment assembly, wrapper chain | **Boundary** (imports coupled code, applies agnostic wrappers) |
| `__init__.py` | 64 | Gym registration, `MUJOCO_GL` env var | **Boundary** (engine-specific setup + generic registration) |
| `wrappers.py` | 232 | dm_env observation/action wrappers | **Engine-agnostic** (pure numpy on dm_env.TimeStep) |
| `gym_wrappers.py` | 59 | dm_env → Gym adapter | **Engine-agnostic** (operates on generic dm_env interface) |
| `oracle.py` | 112 | Oracle wrappers: BFS pathfinding, minimap overlay | **Engine-agnostic** (pure numpy/PIL) |
| `helpers.py` | 11 | Spec sampling utility | **Engine-agnostic** (uses dm_env.specs only) |

**Key insight:** 414/1,024 lines (40%) are engine-agnostic. The remaining 610 lines in `maze.py` + `tasks.py` + `__init__.py` contain all coupling points.

### 1.2 Class Hierarchy

Memory Maze inherits from 6 dm_control base classes:

```
dm_control.composer.Entity
  └── dm_control.locomotion.walkers.legacy_base.Walker
        └── dm_control.locomotion.walkers.base.Walker
              └── dm_control.locomotion.walkers.jumping_ball.JumpingBallWithHead
                    └── dm_control.locomotion.walkers.jumping_ball.RollingBallWithHead
                          └── RollingBallWithFriction                          [maze.py:29-36]

dm_control.composer.Arena
  └── dm_control.locomotion.arenas.mazes.MazeWithTargets
        └── MazeWithTargetsArena                                               [maze.py:236-365]

dm_control.composer.Task
  └── dm_control.locomotion.tasks.random_goal_maze.NullGoalMaze
        └── MemoryMazeTask                                                     [maze.py:39-200]

labmaze.RandomMaze
  └── TextMazeVaryingWalls                                                     [maze.py:367-393]

dm_control.locomotion.arenas.labmaze_textures.WallTextures
  └── FixedWallTexture                                                         [maze.py:203-216]

dm_control.locomotion.arenas.labmaze_textures.FloorTextures
  └── FixedFloorTexture                                                        [maze.py:218-233]
```

Additionally, `TargetSphere` from `dm_control.locomotion.props.target_sphere` is used as-is (not subclassed).

### 1.3 Coupling Inventory

All coupling points are in `maze.py`, `tasks.py`, and `__init__.py`. They fall into 5 categories:

#### Category 1: MJCF Structural (scene graph construction)

These calls build the MuJoCo scene graph programmatically:

| Location | Code | Purpose |
|----------|------|---------|
| `maze.py:7` | `from dm_control import mjcf` | MJCF module import |
| `maze.py:35` | `self._mjcf_root.find('joint', 'roll').damping = roll_damping` | Walker joint damping override |
| `maze.py:36` | `self._mjcf_root.find('joint', 'steer').damping = steer_damping` | Walker joint damping override |
| `maze.py:175-180` | `target_sphere.TargetSphere(radius=..., rgb1=..., rgb2=...)` | Target prop creation |
| `maze.py:182` | `self._maze_arena.attach(target)` | Composer entity attachment |
| `maze.py:191` | `mjcf.get_attachment_frame(target.mjcf_model).pos = pos` | Target positioning via MJCF frame |
| `maze.py:208` | `self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)` | Texture MJCF root (wall) |
| `maze.py:213-215` | `self._mjcf_root.asset.add('texture', ...)` | Texture asset registration |
| `maze.py:222` | `self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)` | Texture MJCF root (floor) |
| `maze.py:231-233` | `self._mjcf_root.asset.add('texture', ...)` | Floor texture asset registration |
| `maze.py:258-276` | `super()._build(maze=..., xy_scale=..., wall_textures=..., ...)` | Arena construction delegated to `MazeWithTargets._build()` |
| `maze.py:290-310` | `del self._mjcf_root.worldbody.geom[...]`, `self._maze_body.geom.clear()` | Maze regeneration: remove old geometry |
| `maze.py:359-364` | `self._mjcf_root.asset.add('material', ...)`, `self._mjcf_root.worldbody.add('geom', ...)` | Floor tile construction |
| `tasks.py:70` | `RollingBallWithFriction(camera_height=0.3, add_ears=top_camera)` | Walker instantiation |
| `tasks.py:71-88` | `MazeWithTargetsArena(x_cells=..., wall_textures=..., ...)` | Arena instantiation |
| `tasks.py:81-85` | `FixedFloorTexture(...)`, `FixedWallTexture(...)`, `labmaze_textures.WallTextures(...)` | Texture object creation |

#### Category 2: Physics State Access

These calls read or write simulation state:

| Location | Code | Purpose |
|----------|------|---------|
| `maze.py:80` | `phys.bind(walker.root_body).xpos` | Walker position (for egocentric transform origin) |
| `maze.py:83` | `physics.bind(targets[index].geom).xpos` | Target position (for egocentric vectors) |
| `maze.py:138` | `super().initialize_episode(physics, rng)` | Physics initialization (sets walker pose, resets contacts) |
| `maze.py:143` | `super().after_step(physics, rng)` | Post-step processing (updates observables, handles contacts) |
| `maze.py:146` | `target.activated` | Contact-based activation (via `TargetSphere.after_substep` checking `physics.data.contact`) |
| `maze.py:151` | `target.reset(physics)` | Reset target activation state + material alpha |
| `maze.py:153-154` | `super().should_terminate_episode(physics)` | Termination check (time limit via composer) |
| `tasks.py:105-109` | `composer.Environment(time_limit=..., task=..., random_state=..., strip_singleton_obs_buffer_dim=True)` | Top-level environment wrapping physics + rendering |

#### Category 3: Rendering

These calls configure visual output:

| Location | Code | Purpose |
|----------|------|---------|
| `maze.py:110` | `self._walker.observables.egocentric_camera.height = camera_resolution` | Camera resolution (egocentric) |
| `maze.py:111` | `self._walker.observables.egocentric_camera.width = camera_resolution` | Camera resolution (egocentric) |
| `maze.py:112` | `self._maze_arena.observables.top_camera.height = camera_resolution` | Camera resolution (top-down) |
| `maze.py:113` | `self._maze_arena.observables.top_camera.width = camera_resolution` | Camera resolution (top-down) |
| `tasks.py:103` | `task.observables['top_camera'].enabled = True` | Enable top-down camera |
| `tasks.py:112` | `'walker/egocentric_camera'` / `'top_camera'` | Observable key for image observation |
| `__init__.py:5-6` | `os.environ['MUJOCO_GL'] = 'egl'` | MuJoCo-specific rendering backend selection |

#### Category 4: Observation Pipeline

These calls create observables within the dm_control composer framework:

| Location | Code | Purpose |
|----------|------|---------|
| `maze.py:8` | `from dm_control.composer.observation import observable as observable_lib` | Observable library import |
| `maze.py:87-95` | `walker.observables.add_observable(...)`, `walker.observables.add_egocentric_vector(...)` | Per-target absolute and egocentric position observables |
| `maze.py:89,94` | `observable_lib.Generic(functools.partial(_target_pos, ...))` | Custom callable observables for target positions |
| `maze.py:92-95` | `walker.observables.add_egocentric_vector(...)` with `origin_callable` | Egocentric frame transform (body-relative vectors) |
| `maze.py:97` | `self._task_observables = super().task_observables` | Inherit base class observables (maze_layout, absolute_position, etc.) |
| `maze.py:105-108` | `self._task_observables['target_index'] = observable_lib.Generic(...)` | Custom task observables |

#### Category 5: Composer Lifecycle

These are lifecycle hooks called by the dm_control composer framework:

| Location | Code | Purpose |
|----------|------|---------|
| `maze.py:123` | `def initialize_episode_mjcf(self, rng)` | Pre-compilation episode setup: regenerate maze, place targets |
| `maze.py:124` | `self._maze_arena.regenerate(rng)` | Trigger maze layout regeneration + geometry rebuild |
| `maze.py:128` | `self._create_targets(clear_existing=True, ...)` | Recreate target entities (MJCF structural change) |
| `maze.py:137` | `def initialize_episode(self, physics, rng)` | Post-compilation: walker spawn, state reset |
| `maze.py:142` | `def after_step(self, physics, rng)` | Per-step: check target activation, award reward |
| `maze.py:153` | `def should_terminate_episode(self, physics)` | Termination condition check |
| `maze.py:156` | `def get_reward(self, physics)` | Reward computation |
| `maze.py:161-182` | `_create_targets()` | Create target entities and attach to arena |
| `maze.py:165` | `target.detach()` | Remove entity from scene graph |
| `maze.py:278-310` | `regenerate(self, random_state)` | Full maze geometry rebuild (walls, floors, textures) |

### 1.4 Dependency Graph

```
__init__.py
  ├── imports tasks.py
  ├── imports gym_wrappers.py
  └── sets MUJOCO_GL env var

tasks.py
  ├── imports maze.py (all classes via *)
  ├── imports wrappers.py (all wrappers via *)
  ├── imports oracle.py (DrawMinimapWrapper, PathToTargetWrapper)
  ├── imports dm_control.composer
  └── imports dm_control.locomotion.arenas.labmaze_textures

maze.py
  ├── imports dm_control.mjcf
  ├── imports dm_control.composer.observation.observable
  ├── imports dm_control.locomotion.arenas (covering, labmaze_textures, mazes)
  ├── imports dm_control.locomotion.props.target_sphere
  ├── imports dm_control.locomotion.tasks.random_goal_maze
  ├── imports dm_control.locomotion.walkers.jumping_ball
  ├── imports labmaze
  └── imports labmaze.assets

wrappers.py
  └── imports dm_env (interface only — no engine dependency)

gym_wrappers.py
  ├── imports dm_env
  └── imports gym

oracle.py
  └── imports memory_maze.wrappers.ObservationWrapper

helpers.py
  └── imports dm_env.specs
```

---

## 2. Modular Architecture Design

### 2.1 Four-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 4: PUBLIC API                                                │
│  __init__.py — Gym environment registration                         │
│  gym_wrappers.py — dm_env → Gym adapter                             │
│  (unchanged, engine-agnostic)                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: WRAPPERS                                                  │
│  wrappers.py — observation/action wrappers (pure numpy)             │
│  oracle.py — pathfinding, minimap (pure numpy/PIL)                  │
│  helpers.py — spec sampling                                         │
│  (unchanged, engine-agnostic)                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: TASK LOGIC                                                │
│  task.py — MemoryMazeEnv (game logic, calls PhysicsBackend)         │
│  factory.py — factory functions + backend registry                  │
│  (NEW, engine-agnostic)                                             │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1: BACKEND                                                   │
│  backends/mujoco/ — MuJoCoBackend (PhysicsBackend implementation)   │
│  backends/genesis/ — FUTURE                                         │
│  backends/mjx/ — FUTURE                                             │
│  (engine-specific, isolated)                                        │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 0: MAZE GENERATION                                           │
│  maze_gen.py — TextMazeVaryingWalls + MazeLayout                    │
│  (pure labmaze, engine-agnostic)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Principles

1. **No dm_control inheritance above Layer 1.** Task logic and public API never import dm_control.
2. **Backend isolation.** Each backend is a self-contained package. Importing `backends.genesis` must not import MuJoCo, and vice versa.
3. **Lazy imports.** Backend packages are imported only when selected, so users only need dependencies for their chosen backend.
4. **Backward compatibility.** Old import paths (`from memory_maze.maze import MemoryMazeTask`) continue to work via deprecated shims.
5. **dm_env as interface contract.** The task layer produces `dm_env.TimeStep` objects. Wrappers and Gym adapter consume them unchanged.

---

## 3. Abstract Interfaces (`interfaces.py`)

### 3.1 Design Decision: Flat PhysicsBackend

The interface uses a single flat `PhysicsBackend` protocol rather than separate Arena/Walker/Target ABCs. Rationale:

- The walker, arena, and targets share a **single physics world** and **contact graph**. Splitting them into separate abstractions forces every backend to coordinate shared state (e.g., "did the walker contact target #3?" requires knowledge of both).
- The arena cannot be independently swapped — wall geometry determines collision behavior that affects walker dynamics.
- Target activation depends on contact detection between walker geoms and target geoms within the same simulation step.
- A flat interface with ~12 methods is simpler to implement correctly for a new backend than 3 interacting ABCs.

### 3.2 Data Classes

```python
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

@dataclass(frozen=True)
class WalkerConfig:
    """Configuration for the rolling ball walker."""
    camera_height: float = 0.3
    camera_fov: float = 80.0
    ball_radius: float = 0.2
    roll_damping: float = 5.0
    steer_damping: float = 20.0
    add_ears: bool = False  # visual markers for top-down camera

@dataclass(frozen=True)
class TargetConfig:
    """Configuration for a single target sphere."""
    radius: float = 0.6
    height_above_ground: float = -0.6
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

@dataclass(frozen=True)
class MazeConfig:
    """Configuration for maze construction."""
    xy_scale: float = 2.0
    z_height: float = 1.5
    physics_timestep: float = 0.005
    control_timestep: float = 0.25
    camera_resolution: int = 64

@dataclass
class MazeLayout:
    """Output of maze generation — engine-agnostic maze description."""
    entity_layer: np.ndarray     # 2D char array: '*'=wall, ' '=open, 'P'=spawn, 'G'=target
    variations_layer: np.ndarray # 2D char array: '.'=default, 'A'-'Z'=room variations
    width: int                   # outer grid width (includes exterior walls)
    height: int                  # outer grid height
    spawn_positions: list        # list of (x, y) world-coordinate spawn positions
    target_positions: list       # list of (x, y) world-coordinate target positions
    wall_segments: list          # aggregated wall box descriptions for geometry creation
```

### 3.3 PhysicsBackend Protocol

```python
from typing import Protocol, Optional, Tuple
import numpy as np

class PhysicsBackend(Protocol):
    """Abstract interface for physics engine backends.

    A backend encapsulates all engine-specific operations:
    scene construction, physics stepping, state access, and rendering.
    """

    # --- Scene lifecycle ---

    def compile_scene(
        self,
        maze_layout: MazeLayout,
        walker_config: WalkerConfig,
        targets: list[TargetConfig],
        maze_config: MazeConfig,
    ) -> None:
        """Build or rebuild the physics scene from a maze layout.

        Called once per episode (or once at startup if the backend
        supports geometry reconfiguration without full recompilation).
        After this call, the scene is ready for stepping.
        """
        ...

    def close(self) -> None:
        """Release engine resources."""
        ...

    # --- Simulation ---

    def step(self, action: np.ndarray) -> None:
        """Apply a 2D continuous action [roll, steer] and advance physics.

        The backend handles internal sub-stepping (e.g., 50 physics steps
        per control step at 200Hz/4Hz).
        """
        ...

    def get_time(self) -> float:
        """Return current simulation time in seconds."""
        ...

    # --- Walker state ---

    def get_walker_position(self) -> np.ndarray:
        """Return walker world position as (3,) float array [x, y, z]."""
        ...

    def get_walker_orientation(self) -> np.ndarray:
        """Return walker orientation as (3, 3) rotation matrix."""
        ...

    def set_walker_pose(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """Set walker position and optionally orientation.

        Used during episode initialization to place walker at spawn point.
        """
        ...

    # --- Target state ---

    def get_target_position(self, index: int) -> np.ndarray:
        """Return target world position as (3,) float array."""
        ...

    def set_target_position(self, index: int, position: np.ndarray) -> None:
        """Move target to a new world position."""
        ...

    def check_target_contact(self, index: int) -> bool:
        """Check if walker is in contact/proximity with target.

        Equivalent to MuJoCo's gap-based contact detection: returns True
        if the walker geom overlaps the target's activation radius.
        """
        ...

    def set_target_visible(self, index: int, visible: bool) -> None:
        """Show or hide a target (e.g., after activation).

        In MuJoCo, this toggles material alpha. Backends may use
        position displacement, alpha, or removal.
        """
        ...

    # --- Rendering ---

    def render_egocentric(self) -> np.ndarray:
        """Render first-person RGB image from walker's camera.

        Returns (H, W, 3) uint8 array at the resolution specified
        in MazeConfig.camera_resolution.
        """
        ...

    def render_top_down(self) -> np.ndarray:
        """Render top-down RGB image of the full maze.

        Returns (H, W, 3) uint8 array. Used for visualization and
        oracle mode.
        """
        ...
```

### 3.4 Interface Verification

Every coupling point from Section 1.3 maps to a `PhysicsBackend` method:

| Coupling Category | Coupling Point | Interface Method |
|-------------------|---------------|------------------|
| MJCF Structural | Arena/walker/target construction | `compile_scene()` |
| MJCF Structural | `mjcf.get_attachment_frame().pos = pos` | `set_target_position()` |
| MJCF Structural | `target.detach()` / `arena.attach(target)` | Handled internally by `compile_scene()` |
| MJCF Structural | Texture creation | Handled internally by `compile_scene()` |
| Physics State | `physics.bind(walker.root_body).xpos` | `get_walker_position()` |
| Physics State | `physics.bind(targets[i].geom).xpos` | `get_target_position()` |
| Physics State | `target.activated` (contact detection) | `check_target_contact()` |
| Physics State | `target.reset(physics)` (visibility toggle) | `set_target_visible()` |
| Physics State | `initialize_episode(physics, rng)` (walker spawn) | `set_walker_pose()` |
| Rendering | `egocentric_camera` observable | `render_egocentric()` |
| Rendering | `top_camera` observable | `render_top_down()` |
| Rendering | Camera resolution configuration | `MazeConfig.camera_resolution` (passed to `compile_scene()`) |
| Observation Pipeline | `add_egocentric_vector()` | Replaced by pure linear algebra in `task.py` (see Section 4) |
| Observation Pipeline | `observable_lib.Generic()` | Replaced by direct calls to `get_walker_position()` / `get_target_position()` |
| Observation Pipeline | `task_observables` dict | Replaced by `task.py._get_observation()` assembling dict directly |
| Composer Lifecycle | `initialize_episode_mjcf()` | `compile_scene()` (called from `task.py.reset()`) |
| Composer Lifecycle | `initialize_episode()` | `set_walker_pose()` (called from `task.py.reset()`) |
| Composer Lifecycle | `after_step()` | `check_target_contact()` (called from `task.py.step()`) |
| Composer Lifecycle | `should_terminate_episode()` | Time check via `get_time()` in `task.py` |
| Composer Lifecycle | `get_reward()` | Pure logic in `task.py` (no backend call needed) |

**No coupling point is left unaddressed.**

---

## 4. Engine-Agnostic Task (`task.py`)

### 4.1 MemoryMazeEnv Class

`MemoryMazeEnv` implements the full environment logic without inheriting from any dm_control class. It produces `dm_env.TimeStep` objects so existing wrappers work unchanged.

```python
class MemoryMazeEnv(dm_env.Environment):
    """Memory Maze environment using a pluggable PhysicsBackend.

    Does NOT inherit from dm_control composer. Implements reset()/step()
    directly by calling PhysicsBackend methods and managing game state.
    """

    def __init__(
        self,
        backend: PhysicsBackend,
        maze_gen: MazeGenerator,          # from maze_gen.py
        walker_config: WalkerConfig,
        target_configs: list[TargetConfig],
        maze_config: MazeConfig,
        time_limit: float,
        enable_global_observables: bool = False,
        enable_top_camera: bool = False,
        random_state: np.random.RandomState = None,
    ):
        self._backend = backend
        self._maze_gen = maze_gen
        self._walker_config = walker_config
        self._target_configs = target_configs
        self._maze_config = maze_config
        self._time_limit = time_limit
        self._enable_global_observables = enable_global_observables
        self._enable_top_camera = enable_top_camera
        self._rng = random_state or np.random.RandomState()

        # Game state
        self._current_target_ix = 0
        self._rewarded_this_step = False
        self._targets_obtained = 0
        self._current_layout: Optional[MazeLayout] = None
```

### 4.2 Episode Lifecycle

```
reset()
  ├── maze_gen.generate(rng)          → MazeLayout
  ├── backend.compile_scene(layout, walker_config, targets, maze_config)
  ├── _place_targets(rng)             → calls backend.set_target_position()
  ├── _spawn_walker(rng)              → calls backend.set_walker_pose()
  ├── _pick_new_target(rng)           → pure game logic
  └── _get_observation()              → calls backend.render_*(), get_*_position()
      └── return dm_env.TimeStep(FIRST, ...)

step(action)
  ├── backend.step(action)
  ├── _check_contacts()               → calls backend.check_target_contact(i) for each target
  │     ├── if current target contacted: reward=1.0, _pick_new_target()
  │     └── for any contacted target: backend.set_target_visible(i, False), then True
  ├── _check_termination()            → compares backend.get_time() vs time_limit
  ├── _get_observation()
  └── return dm_env.TimeStep(MID or LAST, ...)
```

### 4.3 Egocentric Vector Computation

The dm_control `add_egocentric_vector()` is replaced by pure linear algebra:

```python
def _compute_egocentric_vector(self, world_pos: np.ndarray) -> np.ndarray:
    """Compute body-relative vector from walker to a world position.

    Replaces dm_control's walker.observables.add_egocentric_vector().
    """
    walker_pos = self._backend.get_walker_position()
    walker_rot = self._backend.get_walker_orientation()  # (3, 3) rotation matrix

    # World-frame displacement
    delta = world_pos - walker_pos

    # Rotate into walker's body frame
    egocentric = walker_rot.T @ delta

    return egocentric
```

This produces the same output as the dm_control implementation, which internally does:
1. `delta = target_xpos - origin_xpos` (world frame)
2. `egocentric = xmat.T @ delta` (rotate by inverse of walker orientation)

### 4.4 Observation Assembly

```python
def _get_observation(self) -> dict:
    """Assemble observation dict from backend state.

    Replaces dm_control's observable framework with direct calls.
    """
    obs = {}

    # Primary image observation
    obs['walker/egocentric_camera'] = self._backend.render_egocentric()

    if self._enable_top_camera:
        obs['top_camera'] = self._backend.render_top_down()

    # Task observables (always computed, selectively exposed by wrappers)
    obs['target_color'] = self._target_configs[self._current_target_ix].color
    obs['target_index'] = self._current_target_ix

    if self._enable_global_observables:
        walker_pos = self._backend.get_walker_position()
        walker_rot = self._backend.get_walker_orientation()

        obs['absolute_position'] = walker_pos
        obs['absolute_orientation'] = walker_rot
        obs['maze_layout'] = self._current_layout.entity_layer.copy()

        for i in range(len(self._target_configs)):
            target_pos = self._backend.get_target_position(i)
            obs[f'walker/target_abs_{i}'] = target_pos
            obs[f'walker/target_rel_{i}'] = self._compute_egocentric_vector(target_pos)

    return obs
```

---

## 5. Concrete File Layout

### 5.1 Target Structure

```
memory_maze/
    __init__.py          — MODIFIED: remove MUJOCO_GL, use factory.py, keep Gym registration
    interfaces.py        — NEW: PhysicsBackend protocol, WalkerConfig, TargetConfig, MazeConfig, MazeLayout
    maze_gen.py          — NEW: TextMazeVaryingWalls + MazeGenerator (extracted from maze.py:367-393)
    task.py              — NEW: MemoryMazeEnv (from maze.py:39-200, rewritten engine-agnostic)
    factory.py           — NEW: factory functions + backend registry (replaces tasks.py)
    wrappers.py          — UNCHANGED
    gym_wrappers.py      — UNCHANGED
    oracle.py            — UNCHANGED
    helpers.py           — UNCHANGED
    maze.py              — DEPRECATED: thin shim re-exporting for backward compat
    tasks.py             — DEPRECATED: thin shim delegating to factory.py
    backends/
        __init__.py      — Backend registry + get_backend() function
        mujoco/
            __init__.py
            backend.py   — MuJoCoBackend implementing PhysicsBackend
            arena.py     — MazeWithTargetsArena (from maze.py:236-365)
            walker.py    — RollingBallWithFriction (from maze.py:29-37)
            textures.py  — FixedWallTexture, FixedFloorTexture (from maze.py:203-233)
        genesis/         — FUTURE: GenesisBackend
            __init__.py
            backend.py
        mjx/             — FUTURE: MJXBackend
            __init__.py
            backend.py
```

### 5.2 File Mapping: Current → New

Every file in the current codebase has a defined destination:

| Current File | Destination | Status |
|--------------|-------------|--------|
| `maze.py:29-37` (RollingBallWithFriction) | `backends/mujoco/walker.py` | **Moved** |
| `maze.py:39-200` (MemoryMazeTask) | `task.py` (rewritten) | **Rewritten** |
| `maze.py:203-216` (FixedWallTexture) | `backends/mujoco/textures.py` | **Moved** |
| `maze.py:218-233` (FixedFloorTexture) | `backends/mujoco/textures.py` | **Moved** |
| `maze.py:236-365` (MazeWithTargetsArena) | `backends/mujoco/arena.py` | **Moved** |
| `maze.py:367-393` (TextMazeVaryingWalls) | `maze_gen.py` | **Moved** |
| `maze.py:1-27` (imports, constants) | Split across `interfaces.py`, `task.py`, `maze_gen.py` | **Split** |
| `tasks.py:50-152` (_memory_maze factory) | `factory.py` | **Rewritten** |
| `tasks.py:14-47` (size-specific functions) | `factory.py` | **Rewritten** |
| `__init__.py` | `__init__.py` (modified) | **Modified** |
| `wrappers.py` | `wrappers.py` | **Unchanged** |
| `gym_wrappers.py` | `gym_wrappers.py` | **Unchanged** |
| `oracle.py` | `oracle.py` | **Unchanged** |
| `helpers.py` | `helpers.py` | **Unchanged** |

### 5.3 Deprecated Shims

`maze.py` becomes a backward-compatibility shim:

```python
"""Deprecated: import from memory_maze.task, memory_maze.interfaces,
or memory_maze.backends.mujoco instead."""

import warnings
warnings.warn(
    "memory_maze.maze is deprecated. Use memory_maze.task for MemoryMazeEnv, "
    "memory_maze.backends.mujoco for MuJoCo-specific classes.",
    DeprecationWarning, stacklevel=2
)

# Re-export for backward compatibility
from memory_maze.backends.mujoco.walker import RollingBallWithFriction
from memory_maze.backends.mujoco.arena import MazeWithTargetsArena
from memory_maze.backends.mujoco.textures import FixedWallTexture, FixedFloorTexture
from memory_maze.maze_gen import TextMazeVaryingWalls
from memory_maze.interfaces import TARGET_COLORS

# MemoryMazeTask is replaced by MemoryMazeEnv (different interface)
# Importing it here would be misleading, so we don't re-export it.
```

`tasks.py` becomes a shim delegating to `factory.py`:

```python
"""Deprecated: use memory_maze.factory instead."""

import warnings
warnings.warn(
    "memory_maze.tasks is deprecated. Use memory_maze.factory instead.",
    DeprecationWarning, stacklevel=2
)

from memory_maze.factory import (
    memory_maze_9x9,
    memory_maze_11x11,
    memory_maze_13x13,
    memory_maze_15x15,
)
```

---

## 6. Backend Registration

### 6.1 Three-Level Selection

Backend selection follows a precedence chain:

```
1. Explicit kwarg:   memory_maze_9x9(backend='genesis')
2. Environment var:  MEMORY_MAZE_BACKEND=genesis
3. Default:          'mujoco'
```

### 6.2 Registry Implementation (`backends/__init__.py`)

```python
"""Backend registry with lazy imports."""

from typing import Optional
import os

_BACKEND_REGISTRY = {
    'mujoco': 'memory_maze.backends.mujoco.backend.MuJoCoBackend',
    'genesis': 'memory_maze.backends.genesis.backend.GenesisBackend',
    'mjx': 'memory_maze.backends.mjx.backend.MJXBackend',
}

def get_backend(name: Optional[str] = None) -> type:
    """Resolve and return the PhysicsBackend class.

    Selection precedence:
    1. Explicit `name` argument
    2. MEMORY_MAZE_BACKEND environment variable
    3. Default: 'mujoco'

    Uses lazy imports so that backend dependencies (e.g., genesis, jax)
    are only loaded when that backend is selected.
    """
    if name is None:
        name = os.environ.get('MEMORY_MAZE_BACKEND', 'mujoco')

    if name not in _BACKEND_REGISTRY:
        available = ', '.join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {available}"
        )

    # Lazy import
    module_path, class_name = _BACKEND_REGISTRY[name].rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def register_backend(name: str, class_path: str) -> None:
    """Register a custom backend.

    Args:
        name: Backend identifier (used in MEMORY_MAZE_BACKEND env var)
        class_path: Fully-qualified class path, e.g.
            'my_package.backends.CustomBackend'
    """
    _BACKEND_REGISTRY[name] = class_path
```

### 6.3 Factory Integration (`factory.py`)

```python
def _memory_maze(
    maze_size,
    n_targets,
    time_limit,
    ...,
    backend=None,        # NEW: backend selection
    backend_kwargs=None,  # NEW: backend-specific options
):
    # Resolve backend class
    from memory_maze.backends import get_backend
    BackendClass = get_backend(backend)
    backend_instance = BackendClass(**(backend_kwargs or {}))

    # Build engine-agnostic components
    maze_gen = MazeGenerator(...)
    walker_config = WalkerConfig(camera_height=0.3, ...)
    target_configs = [TargetConfig(radius=0.6, color=TARGET_COLORS[i], ...) for i in range(n_targets)]
    maze_config = MazeConfig(camera_resolution=camera_resolution, ...)

    # Create engine-agnostic environment
    env = MemoryMazeEnv(
        backend=backend_instance,
        maze_gen=maze_gen,
        walker_config=walker_config,
        target_configs=target_configs,
        maze_config=maze_config,
        time_limit=time_limit,
        ...
    )

    # Apply wrapper chain (unchanged from current tasks.py)
    env = RemapObservationWrapper(env, obs_mapping)
    if target_color_in_image:
        env = TargetColorAsBorderWrapper(env)
    ...
    return env
```

---

## 7. Data Flow Diagrams

### 7.1 Observation Pipeline

```
PhysicsBackend                    MemoryMazeEnv                  Wrapper Chain                   Gym
┌─────────────┐            ┌──────────────────┐           ┌─────────────────┐          ┌──────────┐
│render_egocen-│───RGB───>  │_get_observation()│──dict──>  │RemapObservation │──dict──>  │GymWrapper│
│  tric()     │            │                  │           │  Wrapper        │          │          │
│render_top_  │───RGB───>  │ Assembles:       │           ├─────────────────┤          │ Converts │
│  down()     │            │  - image         │           │TargetColorAs-   │          │ dm_env   │
│get_walker_  │───pos───>  │  - target_color  │           │  BorderWrapper  │          │ TimeStep │
│  position() │            │  - target_index  │           ├─────────────────┤          │ to Gym   │
│get_walker_  │───rot───>  │  - abs_position  │           │ImageOnlyObs-    │          │ (obs,    │
│  orient...()│            │  - abs_orient    │           │  Wrapper        │          │  reward, │
│get_target_  │───pos───>  │  - maze_layout   │           ├─────────────────┤          │  done,   │
│  position() │            │  - target_rel/abs│           │DiscreteAction-  │          │  info)   │
└─────────────┘            │                  │           │  SetWrapper     │          │          │
                           │ Returns:         │           └─────────────────┘          └──────────┘
                           │  dm_env.TimeStep │
                           └──────────────────┘
```

### 7.2 Action Pipeline

```
Gym User                    DiscreteAction-           MemoryMazeEnv            PhysicsBackend
                            SetWrapper
┌─────────┐          ┌─────────────────┐         ┌──────────────┐         ┌───────────────┐
│ int ∈   │──int──>  │ Maps to 2D      │──2D──>  │ step(action) │──2D──>  │ step(action)  │
│ {0..5}  │          │ continuous      │ float   │              │ float   │               │
│         │          │ action vector   │ array   │ Checks       │ array   │ Physics sub-  │
└─────────┘          │                 │         │ contacts,    │         │ stepping      │
                     │ [0.0, 0.0] noop│         │ computes     │         │ (50 steps @   │
                     │ [-1, 0] forward│         │ reward,      │         │  200Hz)       │
                     │ [0, -1] left   │         │ assembles    │         │               │
                     │ [0, +1] right  │         │ observation  │         │               │
                     │ [-1, -1] fwd+l │         │              │         │               │
                     │ [-1, +1] fwd+r │         │              │         │               │
                     └─────────────────┘         └──────────────┘         └───────────────┘
```

### 7.3 Episode Lifecycle

```
factory.py                     MemoryMazeEnv                   PhysicsBackend
┌──────────────┐        ┌─────────────────────┐         ┌──────────────────────┐
│ Resolve      │        │                     │         │                      │
│ backend      │───────>│ __init__(backend,   │───────> │ (stored, not called  │
│ Create env   │        │   maze_gen, ...)    │         │  until reset)        │
└──────────────┘        │                     │         │                      │
                        │ reset()             │         │                      │
                        │  ├ maze_gen.gen()   │         │                      │
                        │  ├─────────────────────────>  │ compile_scene(       │
                        │  │                  │         │   layout, walker,    │
                        │  │                  │         │   targets, config)   │
                        │  ├─────────────────────────>  │ set_target_pos(i,p)  │
                        │  ├─────────────────────────>  │ set_walker_pose(p,r) │
                        │  ├ _pick_target()   │         │                      │
                        │  ├─────────────────────────>  │ render_egocentric()  │
                        │  └ return TimeStep  │         │                      │
                        │                     │         │                      │
                        │ step(action)        │         │                      │
                        │  ├─────────────────────────>  │ step(action)         │
                        │  ├─────────────────────────>  │ check_target_contact │
                        │  │  reward logic    │         │   (i) for each i     │
                        │  ├─────────────────────────>  │ get_time()           │
                        │  ├─────────────────────────>  │ render_egocentric()  │
                        │  └ return TimeStep  │         │                      │
                        │                     │         │                      │
                        │ close()             │         │                      │
                        │  └─────────────────────────>  │ close()              │
                        └─────────────────────┘         └──────────────────────┘
```

---

## 8. Migration Path

### Phase 1: Extract Engine-Agnostic Code

**Effort:** 1-2 days
**Files created:** `interfaces.py`, `maze_gen.py`
**Files modified:** None (additive only)

1. Create `interfaces.py` with `PhysicsBackend` protocol, `WalkerConfig`, `TargetConfig`, `MazeConfig`, `MazeLayout` data classes, and `TARGET_COLORS` constant.
2. Create `maze_gen.py` by extracting `TextMazeVaryingWalls` (maze.py:367-393) and adding `MazeGenerator` class that wraps labmaze and produces `MazeLayout` objects.
3. Create `backends/__init__.py` with the registry and `get_backend()`.

**Verification:** `maze_gen.py` should be independently testable — generate a layout and assert spawn/target positions exist.

### Phase 2: Create MuJoCo Backend

**Effort:** 2-3 days
**Files created:** `backends/mujoco/backend.py`, `backends/mujoco/arena.py`, `backends/mujoco/walker.py`, `backends/mujoco/textures.py`
**Files modified:** None (additive only)

1. Move `RollingBallWithFriction` → `backends/mujoco/walker.py` (unchanged).
2. Move `FixedWallTexture`, `FixedFloorTexture` → `backends/mujoco/textures.py` (unchanged).
3. Move `MazeWithTargetsArena` → `backends/mujoco/arena.py` (unchanged).
4. Create `backends/mujoco/backend.py` implementing `PhysicsBackend`:
   - `compile_scene()`: instantiates walker, arena, and target spheres; builds `composer.Environment`
   - `step()`: delegates to `composer.Environment.step()`
   - `get_walker_position()`: `physics.bind(walker.root_body).xpos`
   - `check_target_contact()`: checks `target.activated`
   - `render_egocentric()`: reads camera observable
   - etc.

**Verification:** `MuJoCoBackend` should pass all methods when called in the same sequence as the current `tasks._memory_maze()`.

### Phase 3: Create Engine-Agnostic Task

**Effort:** 2-3 days
**Files created:** `task.py`, `factory.py`
**Files modified:** None (additive only)

1. Create `task.py` with `MemoryMazeEnv(dm_env.Environment)`:
   - `reset()`: calls `maze_gen.generate()` → `backend.compile_scene()` → `set_walker_pose()` → `_get_observation()`
   - `step()`: calls `backend.step()` → `check_target_contact()` → `_get_observation()`
   - Egocentric vector computation via pure linear algebra
   - All game logic (target cycling, reward) in pure Python
2. Create `factory.py` with `memory_maze_9x9()`, `memory_maze_11x11()`, etc. that:
   - Resolve backend via `get_backend()`
   - Create `MemoryMazeEnv` with resolved backend
   - Apply wrapper chain (copied from current `tasks.py:111-151`)

**Verification:** `factory.memory_maze_9x9()` should produce an environment behaviorally identical to `tasks.memory_maze_9x9()`. Test by running both with the same seed and comparing observations/rewards for 100 steps.

### Phase 4: Rewire Public API

**Effort:** 1 day
**Files modified:** `__init__.py`, `maze.py` (→ shim), `tasks.py` (→ shim)

1. Update `__init__.py`:
   - Remove `MUJOCO_GL` env var setting (move to `backends/mujoco/__init__.py`)
   - Import from `factory` instead of `tasks`
   - Keep Gym registration unchanged
2. Convert `maze.py` to backward-compat shim (re-exports from new locations)
3. Convert `tasks.py` to backward-compat shim (re-exports from `factory`)

**Verification:** Existing import paths still work. Gym environment registration still works. Deprecation warnings are emitted for old imports.

### Phase 5: Implement Alternative Backends

**Effort:** 2-4 weeks per backend
**Files created:** `backends/genesis/backend.py`, `backends/mjx/backend.py`

1. **Genesis** (recommended first):
   - `compile_scene()`: `gs.morphs.Box` for walls, `gs.morphs.Sphere` for walker/targets, `gs.Camera` for egocentric view
   - `step()`: `scene.step()`
   - `check_target_contact()`: distance-based check in PyTorch (replaces MuJoCo `gap`)
   - `render_egocentric()`: `camera.render()` → numpy
   - Challenge: no dynamic scene rebuild → pre-generate maze pool or use superset geometry

2. **MJX** (if physics fidelity is critical):
   - `compile_scene()`: `mjx.put_model()` + Warp renderer context
   - `step()`: `mjx.step()` (JAX)
   - Challenge: JAX functional style requires rethinking state management

**Verification per backend:** Same seed comparison test as Phase 3. Observation images will differ (different renderer), but walker trajectories, reward timing, and episode lengths should match within tolerance.

---

## 9. Risk Assessment & Trade-offs

### 9.1 Rendering Fidelity

**Risk:** Different rendering backends produce visually different observations.

- MuJoCo uses OpenGL with specific texture mapping, lighting, and anti-aliasing.
- Genesis uses Madrona's batch renderer or its own rasterizer.
- MJX Warp uses NVIDIA's Warp renderer.

**Impact:** RL agents trained on one backend will not transfer zero-shot to another. This is expected and acceptable — the benchmark measures relative agent performance, not cross-engine transfer.

**Mitigation:** Retrain agents on each backend. Compare learning curves and asymptotic performance, not raw observation pixels.

### 9.2 Contact Detection Semantics

**Risk:** MuJoCo's `gap` parameter on `TargetSphere` activates contact at a distance (2× radius), without physical collision force. Alternative backends may not have an exact equivalent.

**Impact:** If contact detection is too sensitive or too loose, reward timing changes, affecting learning.

**Mitigation:** The `PhysicsBackend.check_target_contact()` interface allows each backend to implement distance-based detection. For Genesis/MJX, compute `||walker_pos - target_pos|| < activation_radius` explicitly. Tune the activation radius to match MuJoCo behavior.

### 9.3 Walker Spawn Simplification

**Risk:** The dm_control `NullGoalMaze` base class uses `mj_ray()` raycasting to bias walker spawn orientation toward open space. The `PhysicsBackend` interface does not include raycasting.

**Impact:** Minimal. Spawn orientation affects only the first few steps of an episode. With uniform random orientation, the walker occasionally starts facing a wall and must turn — a trivial difference for RL training.

**Mitigation:** Use uniform random rotation in `task.py._spawn_walker()`. If needed later, add an optional `raycast()` method to `PhysicsBackend`.

### 9.4 Scene Recompilation Cost

**Risk:** MuJoCo recompiles the MJCF model every episode because `initialize_episode_mjcf()` makes structural changes (new wall geometry). The `MuJoCoBackend.compile_scene()` will have the same cost.

**Impact:** This is the current behavior — no regression. For GPU backends (Genesis, MJX), the cost model is different: they require static scene structure after `build()`, so they use a pre-generated maze pool instead.

**Mitigation:** The `compile_scene()` interface is backend-specific. `MuJoCoBackend` recompiles per episode (matching current behavior). `GenesisBackend` would call `compile_scene()` once with a pool of layouts, then reconfigure geometry positions on reset.

### 9.5 Wrapper Compatibility

**Risk:** The existing wrappers (`TargetsPositionWrapper`, `AgentPositionWrapper`, `MazeLayoutWrapper`) access specific observation keys (`walker/target_rel_0`, `absolute_position`, `maze_layout`) that are currently generated by dm_control's observable framework.

**Impact:** If `task.py._get_observation()` doesn't produce the exact same key names and array shapes, wrappers will break.

**Mitigation:** The observation dict assembly in `task.py` (Section 4.4) produces the same keys. Verify with an integration test that checks all wrapper chains.

### 9.6 Proprioceptive Observations

**Risk:** The dm_control composer enables proprioceptive observations (`walker/joints_pos`, `walker/sensors_gyro`, etc.) by default. These are present in the underlying observation dict but filtered out by `RemapObservationWrapper` in standard configurations. They are available in `ExtraObs` variants.

**Impact:** The standard image-only observation mode is unaffected. `ExtraObs` variants that rely on proprioceptive keys would need those keys in the observation dict.

**Mitigation:** The initial implementation omits proprioceptive observations from `task.py._get_observation()` since they are not used in standard benchmarks. If needed, add `get_walker_joint_positions()`, `get_walker_sensors()` methods to `PhysicsBackend` and expose them in `ExtraObs` mode.

### 9.7 Target Color Randomization

**Risk:** The `target_randomize_colors` feature shuffles `TARGET_COLORS` and recreates target entities with new materials. In the modular architecture, `TargetConfig.color` is set at construction time.

**Impact:** Color randomization still works — `factory.py` shuffles colors before creating `TargetConfig` objects, then passes them to `MemoryMazeEnv`. On each `reset()`, the environment can re-shuffle and pass updated `TargetConfig` objects to `compile_scene()`.

**Mitigation:** Add a `randomize_target_colors: bool` flag to `MemoryMazeEnv.__init__()`. When enabled, `reset()` shuffles the color list and updates target configs before calling `compile_scene()`.

---

## Appendix A: Cross-Reference to Existing Analysis

This document was verified against:

- **`analysis_memory_maze_reference.md`** — All 13 MuJoCo features listed in Section 6.1 are covered by `PhysicsBackend` methods. The 7 "MUST-HAVE" porting requirements (Section 8.1) each map to specific interface methods:
  1. Rigid body simulation → `step()`, `compile_scene()`
  2. First-person camera rendering → `render_egocentric()`
  3. Programmatic scene construction → `compile_scene()`
  4. Textured rendering → `compile_scene()` (textures are backend-internal)
  5. Actuator/joint system → `step()` (action mapping is backend-internal)
  6. Collision filtering → `compile_scene()` (backend-internal)
  7. Deterministic simulation → Backend responsibility (same seed → same behavior)

- **`engine_comparison_report.md`** — The "Universal Challenge: Dynamic Maze Regeneration" (Section 6) is addressed by the `compile_scene()` interface, which allows each backend to choose its strategy (per-episode recompilation for MuJoCo, pre-generated pool for Genesis/MJX).

## Appendix B: Complete Coupling Point Traceability

Every coupling point from Section 1.3 → its resolution in the modular architecture:

| # | File:Line | Coupled Code | Resolution |
|---|-----------|-------------|------------|
| 1 | maze.py:7 | `from dm_control import mjcf` | Moved to `backends/mujoco/` |
| 2 | maze.py:8 | `from dm_control.composer.observation import observable as observable_lib` | Eliminated; `task.py` assembles observations directly |
| 3 | maze.py:9 | `from dm_control.locomotion.arenas import covering, labmaze_textures, mazes` | Moved to `backends/mujoco/arena.py` |
| 4 | maze.py:10 | `from dm_control.locomotion.props import target_sphere` | Moved to `backends/mujoco/backend.py` |
| 5 | maze.py:11 | `from dm_control.locomotion.tasks import random_goal_maze` | Eliminated; `task.py` doesn't inherit from dm_control |
| 6 | maze.py:12 | `from dm_control.locomotion.walkers import jumping_ball` | Moved to `backends/mujoco/walker.py` |
| 7 | maze.py:29-36 | `class RollingBallWithFriction(jumping_ball.RollingBallWithHead)` | Moved to `backends/mujoco/walker.py` |
| 8 | maze.py:35-36 | `self._mjcf_root.find('joint', ...).damping = ...` | Moved to `backends/mujoco/walker.py` |
| 9 | maze.py:39 | `class MemoryMazeTask(random_goal_maze.NullGoalMaze)` | Replaced by `task.py:MemoryMazeEnv(dm_env.Environment)` |
| 10 | maze.py:55-63 | `super().__init__(walker=walker, maze_arena=maze_arena, ...)` | Replaced by `MemoryMazeEnv.__init__(backend=...)` |
| 11 | maze.py:80 | `phys.bind(walker.root_body).xpos` | `backend.get_walker_position()` |
| 12 | maze.py:83 | `physics.bind(targets[index].geom).xpos` | `backend.get_target_position(index)` |
| 13 | maze.py:87-95 | `walker.observables.add_observable(...)`, `add_egocentric_vector(...)` | `task.py._compute_egocentric_vector()` + `_get_observation()` |
| 14 | maze.py:97 | `super().task_observables` | `task.py._get_observation()` assembles dict directly |
| 15 | maze.py:105-108 | `observable_lib.Generic(...)` for target_index, target_color | `task.py._get_observation()` sets these keys directly |
| 16 | maze.py:110-113 | Camera resolution configuration | `MazeConfig.camera_resolution` passed to `compile_scene()` |
| 17 | maze.py:123-134 | `initialize_episode_mjcf(rng)` lifecycle hook | `task.py.reset()` → `backend.compile_scene()` |
| 18 | maze.py:137-140 | `initialize_episode(physics, rng)` | `task.py.reset()` → `backend.set_walker_pose()` |
| 19 | maze.py:142-151 | `after_step(physics, rng)` with contact checking | `task.py.step()` → `backend.check_target_contact()` |
| 20 | maze.py:146 | `target.activated` | `backend.check_target_contact(i)` |
| 21 | maze.py:151 | `target.reset(physics)` | `backend.set_target_visible(i, True)` |
| 22 | maze.py:153-154 | `should_terminate_episode(physics)` | `backend.get_time() >= time_limit` in `task.py` |
| 23 | maze.py:165 | `target.detach()` | Internal to `backend.compile_scene()` |
| 24 | maze.py:175-180 | `target_sphere.TargetSphere(...)` | Internal to `backends/mujoco/backend.py` |
| 25 | maze.py:182 | `self._maze_arena.attach(target)` | Internal to `backends/mujoco/backend.py` |
| 26 | maze.py:191 | `mjcf.get_attachment_frame(target.mjcf_model).pos = pos` | `backend.set_target_position(i, pos)` |
| 27 | maze.py:203-233 | Texture classes inheriting from `labmaze_textures` | Moved to `backends/mujoco/textures.py` |
| 28 | maze.py:236-365 | `MazeWithTargetsArena(mazes.MazeWithTargets)` | Moved to `backends/mujoco/arena.py` |
| 29 | maze.py:367-393 | `TextMazeVaryingWalls(labmaze.RandomMaze)` | Moved to `maze_gen.py` (engine-agnostic) |
| 30 | tasks.py:2 | `from dm_control import composer` | Moved to `backends/mujoco/backend.py` |
| 31 | tasks.py:3 | `from dm_control.locomotion.arenas import labmaze_textures` | Moved to `backends/mujoco/backend.py` |
| 32 | tasks.py:70-100 | Walker, arena, task, texture instantiation | Split: engine-agnostic config in `factory.py`, engine-specific in `backends/mujoco/backend.py` |
| 33 | tasks.py:105-109 | `composer.Environment(...)` | Internal to `backends/mujoco/backend.py` |
| 34 | __init__.py:5-6 | `os.environ['MUJOCO_GL'] = 'egl'` | Moved to `backends/mujoco/__init__.py` |

---

*Report produced February 28, 2026. Based on source code analysis of Memory Maze, existing analysis documents (`analysis_memory_maze_reference.md`, `engine_comparison_report.md`), and the project CLAUDE.md.*
