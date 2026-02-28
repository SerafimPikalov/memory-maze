# MuJoCo/MJX Expert Review: Architecture Report

**Reviewer:** MuJoCo/MJX domain expert
**Date:** February 28, 2026
**Document reviewed:** `memory-maze/architecture_report.md`

---

## 1. Overall Assessment

The architecture report proposes a clean, well-considered decomposition of Memory Maze into a backend-agnostic task layer with pluggable physics backends. From a MuJoCo/MJX perspective, the design is sound but has several important nuances that will affect implementation difficulty and performance.

**Fit scores:**
- **MuJoCo (CPU) backend: 9/10** — Near-trivial to implement since it wraps existing dm_control code. The only cost is the thin indirection layer.
- **MJX (GPU) backend: 5/10** — Fundamentally viable but faces significant architectural friction from the stateful `PhysicsBackend` Protocol, per-episode model recompilation, and rendering constraints.

---

## 2. MuJoCo Backend: dm_control/Composer Feature Coverage

### 2.1 What the Report Gets Right

The coupling inventory (Section 1.3) is thorough and accurately identifies all dm_control interaction points. The traceability matrix (Appendix B) correctly maps every coupling point to a `PhysicsBackend` method. Key observations:

- **Entity attachment system** (`arena.attach(walker)`, `arena.attach(target)`) is correctly absorbed into `compile_scene()` as an internal detail.
- **Observable framework elimination** is sound. The dm_control `observable.MJCFCamera`, `observable.MJCFFeature`, and `observable.Generic` classes add caching, buffering, and buffer-dimension management that the architecture report replaces with direct state reads. This is correct for Memory Maze because it uses `strip_singleton_obs_buffer_dim=True` (no temporal buffering needed).
- **Egocentric vector computation** (Section 4.3) correctly reimplements dm_control's body-frame transform as `walker_rot.T @ (target_pos - walker_pos)`. This is mathematically equivalent.

### 2.2 Subtle dm_control Features That Need Attention

**2.2.1 `after_substep` vs `after_step` for Contact Detection**

The report maps contact checking to `task.py.step()` calling `check_target_contact(i)` once per control step. However, in the reference implementation, `TargetSphere.after_substep()` is called after *every physics substep* (50 times per control step at 200Hz/4Hz). It checks `physics.data.contact` each substep to detect transient contacts.

This matters because a fast-moving ball could pass through a target's activation zone within a single substep but not be overlapping at the end of the control step. The `PhysicsBackend.check_target_contact()` as designed would miss this.

**Recommendation:** Either:
1. Document that `check_target_contact()` returns `True` if contact occurred at *any* substep during the last `step()` call (requiring the backend to accumulate contact events internally), or
2. Add an explicit comment that Memory Maze's low ball velocity (~2 m/s) and large target activation radius (1.2m = 2 × 0.6m gap) make single-check-per-step sufficient in practice.

Option (1) is more correct. The MuJoCo backend can implement this by checking `target.activated` which dm_control's `TargetSphere` already accumulates across substeps.

**2.2.2 Walker `create_root_joints()` Mechanism**

The reference walker uses `create_root_joints()` to dynamically add 3 slide joints (root_x, root_y, root_z) to the attachment frame when the walker is placed in the arena. This is part of dm_control's `Entity` lifecycle, not a simple MJCF property.

The `compile_scene()` abstraction correctly hides this, but implementers of the MuJoCo backend must remember that the walker XML alone does not include root translation joints -- they are added by the composer framework during attachment. The MuJoCo backend should either:
- Use the dm_control attachment mechanism as-is (preferred), or
- Pre-modify the walker MJCF to include root joints before compilation.

**2.2.3 `physics.bind()` Semantics**

The report correctly identifies `physics.bind(walker.root_body).xpos` as a coupling point resolved by `get_walker_position()`. However, `physics.bind()` returns a *binding* that reads from `mjData` arrays using the element's compiled index. The MuJoCo backend must either:
- Continue using `physics.bind()` internally (natural if wrapping dm_control), or
- Use `mujoco.mj_name2id()` + direct array indexing into `d.xpos` (if using raw MuJoCo without dm_control).

Both work. The report doesn't need to specify this -- it's correctly left as an internal implementation detail.

### 2.3 Missing: `NullGoalMaze` Base Class Behaviors

`MemoryMazeTask` inherits from `NullGoalMaze`, which inherits from `random_goal_maze.RepeatSingleGoalMaze`, which inherits from `composer.Task`. The base classes provide several behaviors that the architecture report handles but should document more explicitly:

1. **`initialize_episode()` in base classes**: Sets walker pose, clears contacts, resets activation states. The report's `set_walker_pose()` covers this.
2. **Maze observables from `NullGoalMaze`**: `maze_layout`, `absolute_position`, `absolute_orientation`. The report reimplements these in `_get_observation()`.
3. **Spawn orientation via `mj_ray()`**: The report acknowledges this (Section 9.3) and proposes uniform random rotation. This is acceptable.
4. **`n_sub_steps` configuration**: dm_control's `composer.Environment` derives `n_sub_steps = control_timestep / physics_timestep`. The backend's `step()` must handle this. The report says "The backend handles internal sub-stepping" -- correct.

---

## 3. MJX Backend Feasibility

### 3.1 Stateful Protocol vs JAX Functional Style

**This is the central architectural tension for MJX.**

The `PhysicsBackend` Protocol is inherently stateful:
```python
backend.compile_scene(...)  # mutates internal state
backend.step(action)        # mutates internal state
backend.get_walker_position()  # reads internal state
```

MJX is inherently functional:
```python
mjx_data = mjx.step(mjx_model, mjx_data)  # returns new data, no mutation
walker_pos = mjx_data.xpos[walker_body_id]  # reads from returned data
```

An MJX backend would need to maintain `self._model` and `self._data` as instance attributes, calling `mjx.step()` and storing the result:

```python
class MJXBackend:
    def step(self, action):
        self._data = self._data.replace(ctrl=action)
        self._data = mjx.step(self._model, self._data)

    def get_walker_position(self):
        return np.array(self._data.xpos[self._walker_body_id])
```

**This works for a single-environment MJX backend**, but it completely negates MJX's primary advantage: batch simulation via `jax.vmap`. The `PhysicsBackend` Protocol, as designed, is a single-environment interface. Running 1000 parallel environments would create 1000 `MJXBackend` instances, each holding a separate `mjx.Data` -- which is exactly how CPU MuJoCo works and provides zero GPU batching benefit.

**Recommendation for batch MJX:** The `PhysicsBackend` Protocol should have a batch variant or the MJX backend should internally manage a batch dimension:

```python
class BatchMJXBackend:
    """MJX backend that internally manages N parallel environments."""
    def __init__(self, n_envs: int, ...):
        self._n_envs = n_envs
        # self._data has batch dimension: [n_envs, ...]

    def step(self, actions: np.ndarray) -> None:
        """actions shape: [n_envs, 2]"""
        self._data = jax.vmap(mjx.step, in_axes=(None, 0))(self._model, self._data)
```

This is a significant design change that the report should address. Without batch support in the Protocol, MJX provides no advantage over CPU MuJoCo for Memory Maze.

### 3.2 `compile_scene()` Per-Episode: JIT Compilation Overhead

**This is the highest-risk issue for MJX.**

The report's lifecycle calls `compile_scene()` every episode because Memory Maze regenerates maze geometry. For MJX, `compile_scene()` involves:

1. `mujoco.MjModel.from_xml_string(xml)` -- compile MJCF to MjModel (CPU, ~10-50ms)
2. `mjx.put_model(mj_model)` -- transfer model to device, create JAX PyTree (~50-200ms)
3. `mjx.put_data(mj_model, mj_data)` -- transfer data to device (~10-50ms)
4. First call to `jax.jit(mjx.step)` -- **JIT compilation** (~5-30 seconds for a model of this complexity)
5. If using Warp renderer: `mjx.create_render_context(mj_model, nworld=...)` -- create Warp rendering context (~1-5 seconds)

**The JIT compilation (step 4) is the killer.** JAX traces and compiles the step function for a *specific model structure* (number of bodies, joints, geoms). When the model changes (new maze layout = different number of geom primitives), the JIT cache misses and recompilation occurs.

**Mitigation strategies:**

1. **Fixed-size superset model (RECOMMENDED):** Pre-allocate the maximum number of wall geoms, floor tiles, and targets. Each episode, reconfigure positions/sizes via `mjx.Data` fields (specifically `geom_xpos`, `geom_size`, `geom_rgba`). Hide unused geoms by setting `geom_rgba` alpha to 0 or moving them far off-screen.

   This is the approach mentioned in the existing `analysis_mujoco_mjx.md` (Section 6.3). The key insight: the *number* of geoms must be fixed, but their *positions, sizes, and visibility* can change between steps without recompilation.

   **Caveat:** `geom_xpos` and `geom_size` are in `mjx.Model`, not `mjx.Data`. In MJX, the model is the static component. To change geometry positions, you would need to either:
   - Use `mjx.Model.replace(geom_pos=new_positions)` (which creates a new Model PyTree but does NOT trigger JIT recompilation as long as array shapes are identical)
   - Or map wall geoms to bodies with slide joints, so their positions are controlled through `qpos`

   The `replace()` approach is correct and shape-stable, so it will NOT retrigger JIT. This is a viable path.

2. **Pre-compiled maze pool:** Generate K fixed maze layouts, compile all K models at startup (paying the JIT cost once per layout), and sample from the pool during training. This limits maze diversity but avoids per-episode JIT.

3. **Warp backend instead of JAX:** The Warp backend's `mjx.step()` doesn't use JAX JIT at all -- it uses Warp kernels that are compiled once per model structure and cached. Warp compilation is also slower on first run but the caching is more predictable. However, the same structural-change issue applies.

### 3.3 MJX Feature Support for Memory Maze

Memory Maze uses the following MuJoCo features. Here's their MJX support status:

| Feature | Used by Memory Maze | MJX JAX Support | MJX Warp Support |
|---------|-------------------|-----------------|-------------------|
| Sphere geom | Walker shell | Yes | Yes |
| Box geom | Walls | Yes | Yes |
| Plane geom | Ground, floor tiles, wall textures | Yes | Yes |
| Cylinder geom | Walker head/neck | Yes | Yes |
| Hinge joint | Walker roll/steer | Yes | Yes |
| Slide joint | Walker root_x/y/z | Yes | Yes |
| General actuator | Walker roll | Yes | Yes |
| Motor actuator | Walker steer | Yes | Yes |
| Camera rendering | Egocentric, top-down | **No (JAX)** | **Yes (Warp)** |
| Texture rendering | Walls, floors, targets | **No (JAX)** | **Yes (Warp >= 1.12)** |
| `gap` parameter | Target activation | **Yes** (primitive geoms) | **Yes** |
| `contype`/`conaffinity` | Visual-only geoms | Yes | Yes |
| Geom groups | Wall collision geoms (group 3) | N/A (physics only) | Yes (renderer) |
| Material alpha | Target visibility toggle | Partial (Model field) | Yes |
| Skybox texture | Background | **No (JAX)** | Likely yes |

**Critical finding:** MJX JAX cannot render at all. The Warp backend is the only viable path for GPU-accelerated Memory Maze with rendering. This aligns with the existing analysis.

### 3.4 Gap-Based Contact Detection in MJX

The architecture report identifies contact detection as a risk (Section 9.2). From source code analysis:

MJX's collision driver (`_src/collision_driver.py:287-318`) correctly handles the `gap` parameter for primitive geom pairs:
```python
gap = m.geom_gap[geom1] + m.geom_gap[geom2]
# ...
includemargin = margin - gap  # contact activated when dist < includemargin
```

For Memory Maze's target spheres, `gap = 2 * radius` (set on the target geom). When the walker sphere (gap=0 by default) interacts with a target sphere (gap=1.2 for radius=0.6):
- `total_gap = 0 + 1.2 = 1.2`
- `includemargin = margin - gap` → contact is detected when distance < `margin - 1.2`
- With default `margin=0`, this means contact activates at `dist < -1.2` which is wrong...

**Wait -- let me re-examine.** In MuJoCo, `gap` *expands* the contact inclusion distance. The `includemargin` in MJX is used as: contact is generated if `dist < includemargin`. The computation is `includemargin = margin - gap`. But MuJoCo's convention is that a positive gap *increases* the distance at which contacts are generated. Looking at the MuJoCo source, the contact inclusion test is `dist < margin - gap`, which means *smaller* `includemargin` = contacts generated at *greater* distances. A positive gap makes `includemargin` more negative, meaning contacts are detected even when geoms are further apart.

So for the target sphere: `includemargin = 0 - 1.2 = -1.2`, meaning contact is generated when `dist < -1.2`. Since `dist` is signed distance (negative when penetrating), `dist < -1.2` means "generate contact when geoms overlap by more than 1.2m." But the target sphere has radius 0.6, so with `gap = 2 * 0.6 = 1.2`, the effect is that contact is detected when the walker sphere's surface is within `1.2m` of the target sphere's surface -- i.e., the activation zone extends 1.2m beyond the visible sphere.

**This is correct and MJX handles it identically to CPU MuJoCo for primitive geom pairs.** The `check_target_contact()` method can read from `mjx.Data.contact` to detect activated targets.

**Important note:** MJX does NOT support gap for mesh or heightfield geom types (raises `NotImplementedError`). Memory Maze only uses primitive geoms, so this is not an issue.

### 3.5 Warp Rendering Fidelity vs MuJoCo OpenGL

The Warp renderer uses ray tracing (via `mujoco_warp/_src/ray.py`) rather than MuJoCo's OpenGL rasterization. Key differences:

1. **Rendering technique:** MuJoCo CPU uses OpenGL rasterization with fixed-function pipeline features (flat/smooth shading, shadow mapping). Warp uses ray casting/tracing on GPU.

2. **Texture support:** Warp requires `warp >= 1.12` for `wp.Texture2D`. Without this version, textures are disabled silently. Memory Maze relies heavily on textures (wall textures, floor textures, target checker patterns).

3. **Geom group filtering:** The Warp renderer defaults to `enabled_geom_groups=[0, 1, 2]`. Memory Maze puts wall collision boxes in group 3 (via dm_control's `covering.make_walls()`), which means they are correctly excluded from rendering. The visual texture planes are in default groups (0), so they render correctly.

4. **Lighting model:** Warp's ray-based renderer may produce different lighting/shading compared to OpenGL. For RL training, this is acceptable (agents are retrained), but it means **zero-shot policy transfer from CPU-rendered to Warp-rendered observations will not work**.

5. **Camera model:** Both use the same pinhole camera model derived from `mjModel` camera parameters (fovy, resolution). The projection should match.

6. **Anti-aliasing:** Differences in AA between OpenGL and Warp ray tracing will produce pixel-level differences, especially at 64x64 resolution where every pixel matters.

**Bottom line:** Visual fidelity will differ. This is expected and acceptable for the benchmark's purpose (measuring relative agent performance), but must be documented.

---

## 4. Specific Code-Level Concerns

### 4.1 `MazeLayout.wall_segments` Type

```python
@dataclass
class MazeLayout:
    wall_segments: list  # aggregated wall box descriptions for geometry creation
```

This should be more precisely typed. For the MuJoCo backend, wall segments come from `covering.make_walls()` which returns a list of `(position, size)` tuples. The interface should specify:
```python
wall_segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]]
# Each: ((x, y, z_center), (half_x, half_y, half_z))
```

Or better, define a `WallSegment` dataclass.

### 4.2 `set_target_visible()` Implementation for MJX

The report says: "In MuJoCo, this toggles material alpha. Backends may use position displacement, alpha, or removal."

In MJX, material alpha (`geom_rgba[..., 3]`) is a Model field, not a Data field. Changing it requires `model.replace(geom_rgba=new_rgba)`. This is fine for shape stability (no JIT recompilation), but it means the MJX backend needs to track the model-data pair together and update Model on visibility changes.

An alternative approach: use `geom_pos` displacement (move hidden targets to z=-1000). This only affects Data (via body qpos), which is cleaner for MJX.

### 4.3 `get_walker_orientation()` Return Type

The interface specifies `(3, 3) rotation matrix`. In MuJoCo, this is `d.xmat[body_id].reshape(3, 3)` (row-major). The report's egocentric computation `walker_rot.T @ delta` is correct for this representation. Good.

### 4.4 Missing `reset_data()` in the Protocol

After `compile_scene()`, the MuJoCo backend needs to call `mujoco.mj_resetData(m, d)` and `mujoco.mj_forward(m, d)` to initialize derived quantities. The report's `set_walker_pose()` implies this, but the sequence matters: `compile_scene()` → `set_walker_pose()` → `set_target_position()` must be followed by a `mj_forward()` call to update `xpos`, `xmat`, etc. before `render_egocentric()` can produce correct output.

This is an internal implementation detail, but backend implementers should be warned: **never render before calling `mj_forward()` after modifying state.**

### 4.5 Thread Safety Consideration

The report doesn't discuss threading. For CPU MuJoCo, `mjModel` is shared read-only, `mjData` is per-thread. Each `MuJoCoBackend` instance would have its own `mjData`. This is correct for multi-process training (each actor in a separate process).

For MJX, threading is handled by JAX's device placement -- multiple environments on one GPU are batched via `vmap`, not threads.

---

## 5. The Batch Simulation Gap

The architecture report designs a **single-environment interface**. This is correct for:
- CPU MuJoCo (one env per process, scale via multiprocessing)
- The reference Memory Maze implementation

But it misses the primary motivation for MJX/Genesis/Madrona: **batch simulation of thousands of environments on one GPU.**

The `PhysicsBackend` Protocol as-is works for single MJX environments but provides no GPU speedup. To realize MJX's potential, the architecture needs either:

1. **A `BatchPhysicsBackend` protocol** with vectorized methods:
   ```python
   def step(self, actions: np.ndarray) -> None:  # [n_envs, 2]
   def render_egocentric(self) -> np.ndarray:     # [n_envs, H, W, 3]
   def check_target_contact(self, index: int) -> np.ndarray:  # [n_envs] bool
   ```

2. **A vectorized `MemoryMazeEnv`** that manages N environments internally, calling a batch backend.

3. **Alternatively**: Use Gymnasium's `VectorEnv` API as the batching layer, with each sub-env using a single `MJXBackend`. But this eliminates GPU batching benefits.

**Recommendation:** The architecture report should acknowledge this gap and note that the current Protocol targets single-env correctness first, with batch extensions as a future concern. The MuJoCo backend doesn't need batching (multiprocessing works), but any GPU backend design should plan for it.

---

## 6. Risk Analysis

### 6.1 Low Risk

| Risk | Reason it's low |
|------|----------------|
| Physics fidelity (MuJoCo backend) | Same engine, identical physics |
| Gap-based contact detection (MJX) | Primitive geoms fully supported |
| Egocentric vector math | Correctly reimplemented |
| Wrapper compatibility | Same observation keys and shapes |
| `contype`/`conaffinity` filtering | Fully supported in both MJX backends |

### 6.2 Medium Risk

| Risk | Mitigation |
|------|-----------|
| Warp texture support requires `warp >= 1.12` | Pin dependency version |
| Visual fidelity differences between OpenGL and Warp ray tracing | Retrain agents; compare learning curves |
| `after_substep` contact accumulation | Document and implement substep-level tracking in backend |
| Material alpha changes require Model update in MJX | Use position displacement instead |

### 6.3 High Risk

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Per-episode JIT recompilation in MJX | 5-30 second delay per episode, training infeasible | Fixed-size superset model with position reconfiguration |
| No batch support in Protocol | MJX provides zero speedup over CPU MuJoCo | Design batch extension or batch wrapper |
| Warp renderer maturity (2026 codebase) | Unknown edge cases, limited community testing | Budget extra debugging time; fallback to CPU rendering for correctness validation |

---

## 7. Recommendations

### 7.1 For the MuJoCo Backend (Priority: Implement First)

1. Wrap the existing dm_control/composer code largely unchanged.
2. `compile_scene()` should create `composer.Environment` internally, managing the full dm_control lifecycle.
3. `check_target_contact()` should return `target.activated` (which accumulates across substeps).
4. Ensure `mj_forward()` is called after state modifications and before rendering.
5. **Estimated effort: 2-3 days.** This is the validation backend -- all tests should pass identically to the reference implementation.

### 7.2 For the MJX Backend (Priority: Plan Carefully)

1. **Use the Warp backend exclusively** (JAX backend cannot render).
2. **Design a fixed-size superset MJCF model** that accommodates the maximum maze size (15x15). Allocate the maximum number of wall geoms, floor tiles, and target spheres. Reconfigure via `Model.replace()` per episode.
3. **Plan for batch simulation from the start.** Even if the initial implementation is single-env, structure the code so that `vmap`/batching can be added later.
4. **Implement contact tracking at the substep level** -- MJX's `step()` can be decomposed into `step1`/`step2` if needed, but for simplicity, check `Data.contact` at the end of each substep within the step loop.
5. **Pin `warp-lang >= 1.12`** for texture support.
6. **Budget 3-4 weeks** for a working single-environment MJX Warp backend, plus 2 weeks for batch support.

### 7.3 For the Architecture Report

1. Add a note about the batch simulation gap and how it should be addressed in future iterations.
2. Clarify that `check_target_contact()` must report contacts from *any* substep during the last `step()` call, not just the final state.
3. Type the `wall_segments` field more precisely.
4. Add a brief note that `geom_rgba` alpha changes require `Model.replace()` in MJX and recommend position displacement as the portable alternative for target hiding.

---

## 8. Summary

| Dimension | MuJoCo Backend | MJX Backend |
|-----------|---------------|------------|
| **Fit score** | **9/10** | **5/10** |
| **Physics fidelity** | Identical (same engine) | Identical (same engine, GPU) |
| **Rendering** | OpenGL (proven, CPU) | Warp ray tracing (new, NVIDIA-only) |
| **Batch simulation** | No (multiprocessing) | Yes (but Protocol doesn't support it) |
| **Per-episode rebuild** | ~50ms (MJCF recompile) | 5-30s (JIT recompile) or instant (superset model) |
| **Implementation effort** | 2-3 days | 3-6 weeks |
| **Risk level** | Low | Medium-High |
| **Primary value** | Validation backend, backward compat | GPU speedup for training at scale |

The MuJoCo backend is the obvious first implementation target. The MJX backend is the most physics-faithful GPU option but requires careful architectural work to avoid the JIT recompilation trap and to realize batch simulation benefits. The Warp rendering path is the only viable GPU rendering option within the MuJoCo ecosystem and should be the focus of MJX backend development.

---

*Review produced February 28, 2026. Based on source code analysis of MuJoCo v3.5.1, MJX JAX/Warp backends, and the architecture report.*
