# Genesis Expert Review: Architecture Report Assessment

**Reviewer:** Genesis Engine Expert
**Date:** February 28, 2026
**Document reviewed:** `memory-maze/architecture_report.md`
**Supporting sources:** `analysis_genesis.md`, Genesis source code (`Genesis/`)

---

## 1. MJCF Loading Compatibility

**Verdict: Not directly applicable, but the architecture report correctly addresses this.**

The architecture report correctly identifies that Genesis cannot directly load the Memory Maze MJCF because it is programmatically generated via `dm_control.mjcf` (an object model), not stored as a static `.xml` file. The proposed `PhysicsBackend.compile_scene()` approach bypasses MJCF entirely by constructing the scene from `MazeLayout` data using Genesis primitives — this is the right call.

However, there are nuances the report does not address:

- **Genesis's MJCF loader depends on MuJoCo itself.** Looking at `Genesis/genesis/utils/mjcf.py:8`, Genesis `import mujoco` and uses `mujoco.MjModel.from_xml_string()` internally to parse MJCF files. This means Genesis's MJCF "understanding" is actually MuJoCo doing the parsing — Genesis then extracts the compiled model structure. For the Memory Maze port, this is irrelevant since we bypass MJCF, but it's worth noting that Genesis does not have an independent MJCF parser.

- **Single-entity limitation is not a blocker.** The `MJCF` morph docstring (`Genesis/genesis/options/morphs.py:787-793`) confirms Genesis treats MJCF as a single entity with one kinematic chain. Since the Memory Maze port will use individual `gs.morphs.Box` / `gs.morphs.Sphere` calls via `compile_scene()`, this limitation is irrelevant.

**Recommendation:** The `compile_scene()` approach is correct. No MJCF loading is needed.

---

## 2. Dynamic Scene Rebuild (Critical Issue)

**Verdict: The architecture report identifies the problem but underspecifies the solution.**

The architecture report correctly notes that Genesis cannot add/remove entities after `scene.build()` (Section 5, Phase 5). It proposes several strategies in `analysis_genesis.md`:
1. Pre-generated maze pool
2. Entity repositioning
3. Multiple scenes
4. Superset geometry

Let me assess each against the Genesis source code:

### Strategy A: Pre-generated Maze Pool (Recommended)

Pre-generate N maze layouts, build one scene with `n_envs=N`, each environment containing a different maze layout. On reset, assign a new layout from the pool.

**Genesis support:** This requires per-environment geometry — different wall positions for each env. Genesis supports this via the **heterogeneous morphs** feature (`Genesis/genesis/engine/scene.py:362-373`):

```python
# Each env gets a different morph variant
morphs = [gs.morphs.Box(pos=wall_pos_env_i, size=wall_size) for i in range(n_envs)]
wall = scene.add_entity(morphs, material=gs.materials.Rigid())
```

However, heterogeneous morphs are limited to Primitive and Mesh types only (line 370-373) and support only single-link entities. This is fine for wall boxes and target spheres. But there's a critical constraint: each variant must be specified **before** `build()`. You cannot change which variant an environment uses after build.

**Problem:** If `n_envs=1024` and pool_size=100, you'd need 1024 variants (one per env), not 100. Or you'd need to build separate scenes per maze layout and batch them — which loses the single-batch advantage.

**Alternative within this strategy:** Use `batch_fixed_verts=True` on wall morphs (`Genesis/genesis/options/morphs.py:184`), which enables per-environment positioning of fixed entities via `entity.set_pos(pos, envs_idx=...)`. This way, you could build a "superset" of wall entities (enough for the largest possible maze), and for each environment, position walls according to its layout. Unused walls could be moved far below the floor.

This is the most practical approach. Let me verify `set_pos` works on fixed entities with `batch_fixed_verts=True`.

From `Genesis/genesis/engine/entities/rigid_entity/rigid_entity.py:2536-2537`:
```python
if self._is_attached:
    gs.raise_exception("Impossible to set position of an entity that has been attached.")
```

The check is on `_is_attached`, not on `is_fixed`. So `set_pos()` should work on fixed entities. However, for fixed entities without a free joint, `set_pos` calls `set_base_links_pos` which manipulates qpos — and fixed entities have 0 DOFs. This needs verification. If `set_pos` doesn't work on fixed entities, the workaround is to use `fixed=False` with very high damping or zero velocity to prevent movement.

**Concrete recommendation for the architecture report:** Add a "Superset Geometry with Per-Env Repositioning" strategy as the primary approach for Genesis:

1. Determine the maximum number of wall segments across all possible maze sizes (a 15x15 maze has ~100 wall segments)
2. Create that many `gs.morphs.Box(fixed=True, batch_fixed_verts=True)` entities
3. On per-environment reset, reposition walls to match the new layout using `entity.set_pos(new_pos, envs_idx=[env_i])`; unused walls go to `(0, 0, -100)`
4. Targets are already repositioned via `set_target_position()` in the interface

### Strategy B: Multiple Scenes

Create one scene per maze layout, each with `n_envs=batch_per_layout`. On reset, swap to a different scene.

**Genesis support:** This works but requires multiple `gs.init()` / `gs.destroy()` cycles or careful multi-scene management. Genesis supports multiple scenes (seen in `gs._scene_registry` at `scene.py:860-867`), but each scene has its own kernel compilation overhead. This is wasteful.

### Strategy C: Rebuild Scene Each Episode

Call `gs.destroy()` and rebuild the entire scene on reset.

**Genesis support:** Technically possible but disastrously slow. `scene.build()` triggers JIT compilation of physics kernels via Quadrants. Even with caching, this adds seconds per episode. Absolutely not viable for RL training.

### Missing from the report

The `compile_scene()` interface as designed (called once per episode) implicitly suggests scene recompilation. The report should explicitly state that **for Genesis, `compile_scene()` should only be called once at startup** (or very rarely), and per-episode maze variation should be handled by repositioning entities within a pre-built scene. The interface comment says "Called once per episode (or once at startup if the backend supports geometry reconfiguration without full recompilation)" — this is good, but the Genesis backend should clearly use the "once at startup" path.

**Risk level: HIGH.** This is the single most important architectural concern for Genesis. If the `PhysicsBackend` interface assumes `compile_scene()` is cheap (called every episode), the Genesis backend will be prohibitively slow.

---

## 3. Batch Simulation

**Verdict: The interface partially supports Genesis's batch sim capabilities but misses the key opportunity.**

The `PhysicsBackend` protocol is designed for **single-environment** operation:

```python
def step(self, action: np.ndarray) -> None:  # single 2D action
def get_walker_position(self) -> np.ndarray:  # single (3,) array
def check_target_contact(self, index: int) -> bool:  # single bool
def render_egocentric(self) -> np.ndarray:  # single (H, W, 3) image
```

Genesis's primary value proposition is batch simulation with `n_envs` parallelism. The current interface design forces a Genesis backend to either:

1. **Ignore batching:** Run single-env Genesis (wastes 99% of the value)
2. **Wrap batch internally:** Have `GenesisBackend` manage `n_envs` internally, but then `MemoryMazeEnv.step()` calls `check_target_contact()` N_targets times per step — all scalar Python calls looping over what should be a single batched GPU operation

This is a significant missed opportunity. The interface should have a batch-aware variant:

```python
class BatchPhysicsBackend(Protocol):
    def step(self, actions: np.ndarray) -> None:  # [n_envs, 2]
    def get_walker_positions(self) -> np.ndarray:  # [n_envs, 3]
    def check_target_contacts(self) -> np.ndarray:  # [n_envs, n_targets] bool
    def render_egocentric(self) -> np.ndarray:  # [n_envs, H, W, 3]
```

**Recommendation:** Either:
- Add a `BatchPhysicsBackend` protocol alongside the scalar one, letting `MemoryMazeEnv` detect and use it when available
- Or redesign `PhysicsBackend` to always operate in batch mode (batch dim of 1 for single-env)

Without this, Genesis's batch simulation is hobbled by Python-side scalar loops.

**Risk level: MEDIUM-HIGH.** The whole point of porting to Genesis is throughput. A scalar interface negates this.

---

## 4. Rendering Pipeline

**Verdict: Mostly well-handled, with important caveats.**

### Egocentric Camera Attachment

Genesis supports camera attachment to rigid links via `camera.attach(rigid_link, offset_T)` (`Genesis/genesis/vis/camera.py:203-225`). This maps well to the Memory Maze egocentric camera. The `move_to_attach()` method (`camera.py:237-260`) updates the camera pose from the link's current transform, and must be called each step before rendering.

**Important detail:** Camera attachment does NOT work with `env_separate_rigid=True` (line 218-219). For per-environment rendering with the Rasterizer, you need `env_separate_rigid=True` to get per-env images. This means attached cameras and per-env Rasterizer rendering are mutually exclusive.

For the BatchRenderer (Madrona), this limitation does not apply — BatchRenderer handles per-env rendering natively. But BatchRenderer is CUDA+Linux only (`Genesis/genesis/vis/batch_renderer.py:274-277`).

**Gap:** For non-Linux/non-CUDA development (e.g., macOS with Metal), the Genesis backend would need a workaround: render one environment at a time using a non-attached camera that manually computes its pose from the walker link's position. This is slower but functional.

### Top-down Camera

Genesis cameras support arbitrary positioning via `scene.add_camera(pos=..., lookat=...)`. The top-down camera is straightforward. For batch envs, the camera would need `env_idx` to render a specific environment.

### Rendering Output

Genesis camera `render()` returns numpy arrays by default (`Genesis/genesis/vis/camera.py:434-441`). RGB is `(H, W, 3)` uint8 — matches the `PhysicsBackend.render_egocentric()` return type. BatchRenderer returns stacked tensors `(n_envs, H, W, 3)` on GPU.

### Visual Fidelity

The architecture report correctly notes that rendering differences between MuJoCo OpenGL and Genesis's renderer(s) mean agents are not cross-transferable. This is acceptable.

**Risk level: LOW** for Linux/CUDA. **MEDIUM** for cross-platform development.

---

## 5. Quadrants Backend for Rolling Ball Physics

**Verdict: Massive overkill, but works fine.**

The Memory Maze physics are trivial: a rolling ball on a flat floor, bouncing off axis-aligned box walls. Genesis's Quadrants-based GPU physics pipeline (JIT-compiled kernels, articulated body dynamics, constraint solving) is vastly overengineered for this. The RigidSolver with its Newton-based constraint solver (`Genesis/genesis/engine/solvers/rigid/constraint/solver.py`) and broadphase+narrowphase collision detection (`Genesis/genesis/engine/solvers/rigid/collider/collider.py`) will handle it, but:

- **Kernel compilation overhead:** The first `scene.build()` will compile physics kernels. For a simple ball+walls scene, this should be relatively fast (tens of seconds, not minutes), but it's a fixed startup cost.
- **Physics accuracy:** Genesis's rigid body solver is tuned for complex articulated systems (robots). For a simple ball, it should work but may require tuning `dt` and `substeps` to match MuJoCo's behavior. The `enable_mujoco_compatibility` flag in `RigidOptions` helps but doesn't guarantee identical dynamics.
- **Performance:** For the simple physics, the bottleneck will be rendering, not physics. Genesis's physics step for a ball+walls scene will be microseconds per environment — the GPU overhead of launching the kernels may actually dominate.

**No concerns here.** The physics will work. It's just overkill.

---

## 6. Contact Detection: Distance-Based Check

**Verdict: The interface design is correct; implementation needs care.**

The `PhysicsBackend.check_target_contact()` interface wisely uses distance-based detection rather than physical contact. In the MuJoCo reference, targets use a `gap` parameter that triggers activation at a distance (2x radius, approximately) without physical collision forces.

In Genesis, this can be implemented two ways:

### Option A: Pure distance computation (Recommended)

```python
def check_target_contact(self, index: int) -> bool:
    walker_pos = self.walker.get_pos()  # torch tensor
    target_pos = self.targets[index].get_pos()
    distance = torch.norm(walker_pos - target_pos, dim=-1)
    return (distance < activation_radius).item()
```

This is simple, fast, and exactly matches the semantic intent. For batch mode, this would be:

```python
def check_target_contacts_batch(self) -> torch.Tensor:
    walker_pos = self.walker.get_pos()  # [n_envs, 3]
    target_pos = torch.stack([t.get_pos() for t in self.targets])  # [n_targets, n_envs, 3]
    # ... compute distances in batch
```

### Option B: Genesis contact system

Genesis does have `entity.get_contacts(with_entity=other)` (`rigid_entity.py:3118-3197`), which returns contact information including geom pairs and forces. This could be used, but:

- It requires the target spheres to have `collision=True` and actually generate contacts
- Contact detection is more expensive than simple distance checks
- The `gap` parameter behavior (activation at a distance) is not naturally represented by Genesis's contact system

**Recommendation:** Use Option A (distance-based). It's simpler, faster, and semantically correct.

---

## 7. Missing API Surface

The `PhysicsBackend` interface is well-designed for the current scope. However, several Genesis-specific needs are not addressed:

### 7.1 Entity Visibility Toggle

`set_target_visible(index, visible)` has no direct equivalent in Genesis. The Genesis `RigidEntity` has no `set_visible()`, `set_alpha()`, or `hide()` method. I searched the entire entity system and found no visibility controls on entities, links, or geoms.

**Workaround options:**
1. Move invisible targets far away (e.g., `(0, 0, -100)`) — crude but effective
2. Use `collision=False, visualization=False` on the morph at creation time — but this is static, set before `build()`
3. Modify the Genesis surface/material at runtime — not currently supported

**Recommendation:** The Genesis backend should implement `set_target_visible()` by moving targets below the floor plane. Document this as a workaround.

### 7.2 Simulation Time Tracking

`get_time()` is needed by the interface. Genesis tracks steps via `scene._t` (step counter, `scene.py:947`), but does not expose elapsed time directly. The backend would compute `time = scene._t * dt * n_substeps_per_control_step`.

### 7.3 Walker Orientation as Rotation Matrix

`get_walker_orientation()` returns a `(3, 3)` rotation matrix. Genesis provides quaternions via `entity.get_quat()`. The backend needs to convert: `R = quaternion_to_rotation_matrix(quat)`. Genesis has `genesis.utils.geom` utilities for this.

### 7.4 Reset Velocity Zeroing

`set_walker_pose()` needs to also zero the walker's velocity. Genesis's `set_pos()` already zeros velocity by default (`zero_velocity=True` at `rigid_entity.py:2518`), so this is handled automatically.

### 7.5 Walker Joint Damping

The Memory Maze walker has specific roll and steer damping values (5.0 and 20.0). In Genesis, joint damping is set via `entity.set_dofs_damping()` or specified in the morph parameters. Since the walker is constructed from primitives, damping must be set explicitly after entity creation.

---

## 8. Overall Fit Score

**Score: 7/10**

### Strengths (what earns the score)

1. **Batch simulation** is Genesis's killer feature for this use case. Running 1024+ maze environments in parallel on GPU with a single `scene.step()` call is exactly what's needed for high-throughput RL training.

2. **Primitive-based construction** (Box, Sphere, Plane) maps perfectly to Memory Maze geometry. No MJCF translation needed.

3. **Camera attachment** works well for egocentric views.

4. **PyTorch-native tensors** enable zero-copy integration with RL training loops.

5. **Per-environment reset** (`scene.reset(envs_idx=...)`) handles asynchronous episode termination.

6. **Contact system** (`get_contacts()`) exists as a fallback, and distance-based checks are trivially implementable in PyTorch.

### Weaknesses (what costs points)

1. **Static scene constraint (-1.5):** Cannot add/remove entities after `build()`. Dynamic maze regeneration requires the superset geometry workaround with per-env repositioning. This is the most significant engineering challenge.

2. **No entity visibility control (-0.5):** Must use position displacement as a workaround for hiding targets.

3. **BatchRenderer platform constraints (-0.5):** CUDA+Linux only. Development on macOS requires fallback to single-env Rasterizer, significantly complicating the development workflow.

4. **Interface mismatch with batch operations (-0.5):** The scalar `PhysicsBackend` interface doesn't expose Genesis's batch capabilities. Needs a batch-aware interface to realize performance gains.

---

## 9. Specific Code-Level Concerns and Suggestions

### 9.1 `compile_scene()` Semantics for Genesis

The current interface implies `compile_scene()` is called every episode:

```python
# task.py reset()
def reset(self):
    layout = self._maze_gen.generate(rng)
    self._backend.compile_scene(layout, ...)  # called every reset!
```

For Genesis, this must NOT trigger `scene.build()` every time. The Genesis backend should:

```python
class GenesisBackend:
    def __init__(self, n_envs, maze_pool_size):
        # Build scene ONCE with superset geometry
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(...)
        self._build_superset_scene(n_envs, max_walls=120)
        self.scene.build(n_envs=n_envs)

    def compile_scene(self, maze_layout, walker_config, targets, maze_config):
        # Just reposition entities — no rebuild
        self._reposition_walls(maze_layout)
        self._reposition_targets(targets)
```

### 9.2 `check_target_contact()` Should Be Batched

For RL training with 1024 environments, calling `check_target_contact(i)` in a Python loop for each target in each environment is O(n_envs * n_targets) Python calls. This should be a single batched GPU operation:

```python
# In task.py step():
contacts = self._backend.check_all_target_contacts()  # [n_envs, n_targets] bool tensor
```

### 9.3 Rendering Must Handle `move_to_attach()` Timing

The Genesis camera attached to a walker link requires `camera.move_to_attach()` to be called before each `camera.render()`. The `step()` method in the backend must ensure this ordering:

```python
def step(self, action):
    self.walker.control_dofs_velocity(action)
    self.scene.step()

def render_egocentric(self):
    self.ego_camera.move_to_attach()  # MUST come before render
    return self.ego_camera.render(rgb=True)
```

If `render_egocentric()` is called without `move_to_attach()`, the camera will render from the previous step's position.

### 9.4 Wall Entity Count Must Be Pre-Determined

The superset geometry approach requires knowing the maximum number of wall segments across all maze sizes. From the maze generation analysis:

| Maze Size | Approx. Max Wall Segments |
|-----------|--------------------------|
| 9x9       | ~40                       |
| 11x11     | ~60                       |
| 13x13     | ~85                       |
| 15x15     | ~110                      |

The Genesis backend should allocate `max_walls` box entities at `build()` time. Unused walls are moved below the floor.

### 9.5 `batch_fixed_verts=True` for Per-Environment Wall Positions

When creating wall entities, use `batch_fixed_verts=True` to enable per-environment positioning:

```python
for i in range(max_walls):
    wall = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, -100),  # default: hidden below floor
            size=(wall_thickness, wall_length, wall_height),
            fixed=True,
            batch_fixed_verts=True,  # CRITICAL: enables per-env set_pos
        ),
        material=gs.materials.Rigid(),
    )
    self.walls.append(wall)
```

### 9.6 Surface Textures

Memory Maze uses textured walls and floors for visual richness. Genesis's `gs.surfaces.Default()` provides basic rendering, but custom textures would need `gs.surfaces.Flat(color=...)` or `gs.surfaces.PBR(...)` with texture images. The architecture report should note that texture support needs verification — Genesis's texture pipeline is designed for meshes loaded from files, not programmatically generated textures.

### 9.7 Camera FOV Convention

Memory Maze uses an 80-degree FOV for the egocentric camera. Genesis cameras use vertical FOV (`fov` parameter in `Camera.__init__`, `camera.py:79`). MuJoCo also uses vertical FOV. These should match, but verify the convention (MuJoCo's `yfov` vs Genesis's `fov`).

---

## 10. Summary of Recommendations

| Priority | Recommendation | Impact |
|----------|---------------|--------|
| **P0** | Add batch-aware `PhysicsBackend` methods (or a `BatchPhysicsBackend` protocol) | Without this, Genesis's batch sim value is wasted |
| **P0** | Document that Genesis's `compile_scene()` should only build once; per-episode resets use entity repositioning | Prevents disastrous per-episode kernel recompilation |
| **P1** | Use superset geometry + `batch_fixed_verts=True` for dynamic maze layouts | Solves the static-scene limitation |
| **P1** | Implement `set_target_visible()` via position displacement (move below floor) | Only viable Genesis approach |
| **P2** | Verify `set_pos()` works on `fixed=True` entities with `batch_fixed_verts=True` | If it doesn't, use `fixed=False` with high damping |
| **P2** | Test camera attachment compatibility with `env_separate_rigid` and BatchRenderer | Affects cross-platform rendering strategy |
| **P2** | Add texture support verification for programmatic wall/floor textures | Affects visual fidelity |
| **P3** | Add fallback rendering path for non-Linux/non-CUDA development | Developer experience on macOS |

---

*Review produced February 28, 2026. Based on source code analysis of Genesis v0.3.x at `Genesis/` and the architecture report at `memory-maze/architecture_report.md`.*
