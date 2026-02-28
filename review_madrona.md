# Madrona Expert Review: Memory Maze Architecture Report

**Reviewer:** Madrona Engine Expert
**Date:** February 28, 2026
**Document Reviewed:** `memory-maze/architecture_report.md`
**Context Documents:** `analysis_madrona.md`, Madrona source at `madrona/`

---

## Executive Summary

The architecture report proposes a clean `PhysicsBackend` protocol with ~12 methods to decouple Memory Maze from dm_control/MuJoCo. From Madrona's perspective, this interface is **fundamentally incompatible** with how Madrona operates. Madrona is not a Python-callable physics engine with a step/render API — it is a C++ batch simulation framework where all logic runs inside a compiled task graph megakernel. A `MadronaBackend(PhysicsBackend)` class that implements `step()`, `render_egocentric()`, etc. in Python would negate Madrona's entire value proposition.

However, the architecture report correctly identifies the **madrona_mjx hybrid** as the most promising Madrona-related path, which sidesteps most of these incompatibilities. The review below evaluates both the direct integration path and the hybrid path.

**Overall Fit Score: 4/10** (as a `PhysicsBackend` implementation)
**madrona_mjx Hybrid Fit: 7/10** (bypasses the interface, uses Madrona for rendering only)

---

## 1. ECS Architecture vs. Flat PhysicsBackend Protocol

### The Fundamental Mismatch

The `PhysicsBackend` protocol (Section 3.3 of the report) assumes a **sequential, per-world, Python-callable** interface:

```python
backend.compile_scene(layout, walker_config, targets, maze_config)
backend.step(action)
pos = backend.get_walker_position()  # returns np.ndarray
img = backend.render_egocentric()     # returns np.ndarray
```

Madrona's ECS operates on entirely different principles:

1. **Data is stored in columnar SoA tables** (`include/madrona/table.hpp`), not as Python-accessible objects. Accessing a component requires knowing the archetype, entity, and column — there is no `get_walker_position()` equivalent without building a C++ system function.

2. **Logic runs inside system functions** (`ParallelForNode<Ctx, Fn, Components...>`) that are compiled into a GPU megakernel. You cannot call individual operations from Python mid-step.

3. **All worlds execute in lockstep.** A single `executor.run(launch_graph)` call processes ALL worlds simultaneously. There is no way to step world 42 independently, call `check_target_contact()` for that world, then decide what to do next — the entire task graph runs atomically.

4. **Data export is via `exportColumn()`** — you get a GPU tensor pointer to an entire column (e.g., all walker positions for all worlds). There is no per-entity Python getter.

### What a "MadronaBackend" Would Actually Look Like

To implement `PhysicsBackend` on top of Madrona, you would need a Python wrapper that:

- Calls `executor.run()` for `step()`
- Reads exported columns via `getExported()` for position/orientation getters
- Reads the batch renderer output buffer for `render_egocentric()`

But this wrapper would wrap the **entire batch** of worlds, not a single world. The `PhysicsBackend` protocol assumes single-world semantics: `get_walker_position()` returns one `(3,)` array. Madrona's exported columns return `[num_worlds, 3]` tensors.

**Verdict:** The `PhysicsBackend` protocol cannot express Madrona's batch semantics. A Madrona integration would need to bypass this protocol entirely and implement a batch-native `VectorizedBackend` interface that operates on tensors of shape `[num_worlds, ...]`.

### Suggested Interface Extension

If the project wants to support Madrona (or any GPU batch engine), the architecture should define a second protocol:

```python
class BatchPhysicsBackend(Protocol):
    """Vectorized backend for GPU batch simulation engines."""

    def compile_scenes(
        self,
        maze_layouts: list[MazeLayout],  # one per world
        walker_config: WalkerConfig,
        targets: list[TargetConfig],
        maze_config: MazeConfig,
    ) -> None: ...

    def step(self, actions: np.ndarray) -> None:
        """actions: [num_worlds, 2] float array"""
        ...

    def get_walker_positions(self) -> np.ndarray:
        """Returns [num_worlds, 3] float array"""
        ...

    def render_egocentric_batch(self) -> np.ndarray:
        """Returns [num_worlds, H, W, 3] uint8 array"""
        ...

    # etc.
```

This would allow a `MadronaBatchBackend` to efficiently wrap the executor and exported columns without per-world Python calls.

---

## 2. GPU Batch Rendering: Madrona's Core Advantage

### Rendering Architecture

The architecture report correctly identifies rendering as the bottleneck for Memory Maze training. Madrona's batch renderer is specifically designed for this scenario:

- **RenderManager** (`include/madrona/render/render_mgr.hpp`) supports configurable `agentViewWidth` and `agentViewHeight` — the Vulkan rasterizer path supports non-square output, which maps well to Memory Maze's 64x64 resolution.

- **CUDA raytracer** (`CudaBatchRenderConfig` in `include/madrona/mw_gpu.hpp`) only supports **square** output (`renderResolution x renderResolution`). This is a minor constraint since Memory Maze uses square 64x64 images.

- **Output is zero-copy GPU memory**: `batchRendererRGBOut()` returns `const uint8_t *` (GPU pointer), and `batchRendererDepthOut()` returns `const float *`. These can be wrapped as PyTorch tensors via nanobind without any CPU↔GPU transfer.

- **Per-world cameras** are configured via `RenderCamera` components (`include/madrona/render/ecs.hpp`): each world gets a camera entity with `fovScale`, `zNear`, and `cameraOffset`. This maps directly to Memory Maze's egocentric camera.

### Throughput Potential

Published benchmarks from the SIGGRAPH 2023 paper:

| Scene complexity | Resolution | RTX 4090 FPS |
|-----------------|-----------|-------------|
| Simple (Hide & Seek) | 64x64 | 300K+ |
| Complex (HSSD) | 64x64 | 30K+ |

Memory Maze's geometry is simpler than HSSD (axis-aligned boxes for walls, spheres for targets, ground plane), so we'd expect throughput closer to the "simple" category. At 300K+ frames/sec with 64x64 resolution, training throughput would be **orders of magnitude** higher than the current MuJoCo/OpenGL baseline.

### Rendering Concerns for Memory Maze

1. **Textured walls**: Memory Maze uses procedural textures for walls and floors (defined via `FixedWallTexture` / `FixedFloorTexture` classes). Madrona's renderer supports materials and textures (`loadObjects()` accepts `SourceMaterial` and `SourceTexture`), but the textures would need to be pre-baked as images rather than generated via MuJoCo's procedural texture model.

2. **Target sphere transparency**: Memory Maze toggles target visibility by changing material alpha (`set_target_visible()`). Madrona's `MaterialOverride` system (`include/madrona/render/ecs.hpp`, lines 121-129) supports `UseOverrideColor` with a packed `uint32_t color` (RGBA), but true alpha blending depends on the renderer implementation. An alternative is to move "invisible" targets off-screen (below the floor), which is simpler and universal.

3. **Top-down camera**: The `render_top_down()` method requires a second camera per world. Madrona's `maxViewsPerWorld` config parameter supports this — set it to 2 for egocentric + top-down views.

4. **Lighting**: Memory Maze uses default MuJoCo lighting. Madrona's `LightDesc` system supports both directional and spotlight types with configurable position, direction, cutoff, and intensity. Matching MuJoCo's exact lighting model would require some tuning but is feasible.

---

## 3. madrona_mjx Hybrid Approach

### Architecture

The `madrona_mjx` project (referenced in `madrona/README.md:52-53`) demonstrates:

```
MJX (JAX)  ──physics state──>  Madrona (CUDA)  ──rendered images──>  JAX/PyTorch
  physics                        batch renderer                       training
```

For Memory Maze, this would mean:
- MJX runs the maze physics (loads the existing MJCF models, handles walker dynamics, contacts)
- After each MJX step, walker/target transforms are copied to Madrona's ECS
- Madrona renders all worlds in a single batch call
- Rendered images go directly to the training pipeline as GPU tensors

### Feasibility Assessment

**Strengths:**
- Preserves physics fidelity — MJX uses MuJoCo's exact algorithms on GPU
- Reuses existing MJCF models without modification
- Gains Madrona's rendering throughput advantage
- JAX integration is production-quality (`JAXInterface::buildEntry<>()` in `include/madrona/py/bindings.hpp`)

**Concerns:**

1. **madrona_mjx is deprecated.** The `analysis_madrona.md` document notes: "Status: Deprecated in favor of MuJoCo Warp (MJWarp)." MJWarp provides its own GPU renderer via NVIDIA Warp, potentially making the Madrona rendering overlay unnecessary. The madrona_mjx codebase may not be maintained going forward.

2. **Scene geometry synchronization.** Memory Maze regenerates maze geometry every episode. In the hybrid approach, MJX physics runs in JAX (functional, immutable state) while Madrona's renderer needs mesh geometry on CUDA. Every time the maze changes, the new wall geometry must be transferred from MJX to Madrona's mesh data structures. This is not a standard operation — `madrona_mjx` was designed for fixed-geometry environments (Cartpole, Franka) where the mesh never changes at runtime.

3. **Dynamic geometry in Madrona's renderer.** The `RenderManager::loadObjects()` method loads mesh assets at initialization. The renderer's BVH structures are built once for the loaded geometry. Dynamically adding/removing wall segments per episode would require either:
   - Pre-loading a superset of all possible wall positions and toggling visibility
   - Rebuilding renderer state per episode (expensive, may negate throughput gains)

4. **JAX version constraint.** `madrona_mjx` requires JAX < 0.6.0. Current JAX (early 2026) is well past this. Compatibility issues are likely.

5. **Single renderer instance limitation.** Only one batch renderer can be active. If you need separate train and eval rendering (different resolutions, different world counts), this requires careful resource management.

### Verdict on Hybrid

The hybrid approach was the most promising Madrona path, but its foundation (`madrona_mjx`) being deprecated significantly weakens the case. If the team is considering MJX for physics, the MJWarp path (MJX physics + Warp GPU rendering) is now the recommended successor — it provides similar benefits without the Madrona rendering dependency. This is explicitly noted in the `engine_comparison_report.md` as the "safest" path.

---

## 4. C++ Development Overhead

### Quantifying the Cost

A full Madrona native port requires implementing all of the following in C++:

| Component | Estimated C++ Lines | Complexity |
|-----------|-------------------|-----------|
| ECS schema (archetypes, components, singletons) | 100-150 | Low |
| Procedural maze generation (Kruskal/Prim) | 200-300 | Medium |
| Wall geometry construction (boxes) | 150-200 | Medium |
| Agent movement system (velocity, collision) | 100-150 | Medium |
| Target contact detection system | 50-80 | Low |
| Reward/episode management systems | 100-150 | Low |
| Task graph setup | 80-120 | Medium |
| Python wrapper (nanobind) | 150-200 | Medium |
| CMake build system integration | 50-80 | Low |
| Asset loading (textures, meshes) | 100-150 | Medium |
| **Total** | **1,080-1,580** | **Medium-High** |

This is roughly 2-3x the current Python codebase (610 coupled lines), but in C++ — a language that is inherently more verbose and has a higher debugging cost, especially for GPU code.

### GPU Debugging Challenge

A critical concern the report understates: **GPU debugging is extremely difficult.** When a system function crashes inside the megakernel:
- No stack traces from device code (CUDA does not provide them in the megakernel model)
- `printf` debugging requires `MADRONA_MWGPU_FORCE_DEBUG=1` and is limited
- Must switch to CPU backend for proper debugging, which may mask GPU-specific bugs
- Memory errors (out-of-bounds in SoA tables) cause silent corruption, not clean crashes

For a research project with 1-2 developers, this overhead is substantial.

---

## 5. Batch Simulation Throughput Analysis

### The Real Throughput Story

The architecture report's throughput numbers are accurate — Madrona achieves millions of steps/sec for simple environments. But we need to decompose what "throughput" means for Memory Maze:

**Step time = Physics time + Rendering time + Data transfer time**

| Component | MuJoCo (current) | Madrona (native) | MJX + Madrona (hybrid) |
|-----------|------------------|-------------------|------------------------|
| Physics | CPU, sequential | GPU, batch | GPU (MJX), batch |
| Rendering | OpenGL EGL | CUDA/Vulkan batch | CUDA/Vulkan batch |
| Data transfer | CPU→GPU per step | Zero-copy (all GPU) | MJX→Madrona sync per step |
| Parallelism | 1 world at a time | 4K-64K worlds | 4K-64K worlds |

Memory Maze's physics is trivial (rolling ball + wall collisions), so **rendering dominates**. This means:
- A Madrona native port gains massively from batch rendering (100-1000x speedup)
- The hybrid approach gains similarly from batch rendering but adds MJX→Madrona sync overhead
- Even the rendering-only approach (Approach C in `analysis_madrona.md`) would provide significant gains

### Scaling Characteristics

From the embedded reference data:

| World Count | GPU Utilization | Memory Maze Suitability |
|-------------|----------------|------------------------|
| < 1K | Low | Debugging only |
| 4K-16K | High | Training sweet spot |
| 16K-64K | Very high | Maximum throughput if memory allows |

Memory Maze's per-world state is small (ball position, ~6 target positions, maze layout pointer), so memory should not be a bottleneck even at 64K worlds. The constraint is renderer memory: at 64x64 RGBD with 2 views/world, that's `64K * 2 * 64 * 64 * 5 bytes = ~2.6 GB` — well within modern GPU memory.

---

## 6. Dynamic Maze Regeneration in Madrona's Static-World Model

### The Core Tension

This is the most significant technical challenge for a Madrona integration, and the architecture report partially addresses it (Section 9.4) but underestimates its complexity.

**The problem:** Memory Maze generates a new random maze layout every episode. Madrona's ECS schema and renderer assets are designed to be static — all archetypes registered at startup, mesh assets loaded once.

**Madrona's constraints:**
1. `registerArchetype()` must be called before simulation starts (`include/madrona/registry.hpp`)
2. `RenderManager::loadObjects()` loads meshes at initialization
3. Entity creation/destruction is supported at runtime via `ctx.makeEntity<>()` / `ctx.destroyEntity()`
4. But the renderer's BVH is tied to loaded `ObjectID`s

**Possible solutions (ordered by feasibility):**

### Solution A: Pre-allocated Wall Pool (Recommended)

Pre-allocate the maximum number of wall entities across all maze sizes. On reset:
1. Destroy all current wall entities
2. Create new wall entities for the new maze layout
3. Set Position, Rotation, Scale for each wall to match the new layout
4. Use `ObjectID` referencing a pre-loaded unit-cube mesh, scaled per-wall

This works because:
- Entity creation/destruction is runtime-supported (`Context::makeEntity<>()`, `Context::destroyEntity()`)
- The mesh (unit cube) is loaded once; wall dimensions come from `Scale` components
- BVH rebuilds automatically (`BVH::rebuildOnUpdate()` in `include/madrona/broadphase.hpp:52`)

Constraints:
- Must set `maxInstancesPerWorld` in `RenderManager::Config` to the maximum possible wall count
- Must configure archetype capacity to handle maximum walls (e.g., 15x15 maze ≈ ~100 wall segments)

### Solution B: Superset Geometry with Visibility Toggle

Pre-create ALL possible wall positions for the largest maze (15x15) in every world. On reset:
- Make active walls visible (set position to correct location)
- Move inactive walls off-screen (position far below the floor)

This avoids entity creation/destruction but wastes memory proportional to the maximum maze size.

### Solution C: Per-Episode World Recycling

Instead of regenerating mazes, pre-generate a large pool of mazes at startup (e.g., 1000 layouts). Assign mazes to worlds round-robin. When an episode ends, recycle the world to a random layout from the pool.

This is the most efficient for Madrona but changes the environment semantics — the maze is no longer truly random per episode. For RL training, this is usually acceptable if the pool is large enough.

---

## 7. The "Highest Ceiling" Claim

### Evaluation

The `engine_comparison_report.md` describes Madrona as the option with the "highest ceiling" for throughput. Let me evaluate this claim against the evidence.

**Arguments in favor:**

1. **Megakernel execution** eliminates kernel launch overhead between systems. A full step (physics + rendering + reward computation) is a single `cudaGraphLaunch`. No other candidate engine achieves this.

2. **Zero-copy GPU pipeline.** Actions → ECS → Physics → Rendering → Observations → Training — the entire loop stays on GPU. MJX requires JAX→CUDA copies for Madrona rendering. Genesis requires Taichi→PyTorch copies.

3. **Published benchmarks** show 3.5M steps/sec for Overcooked (A40), 20M+ for Hanabi (RTX 4090). Memory Maze is simpler than Overcooked physics-wise, so similar or better throughput is plausible.

4. **The batch renderer is purpose-built** for generating small agent-view images at scale. This is exactly what Memory Maze needs.

**Arguments against:**

1. **Development cost is extreme.** The "ceiling" only exists if someone actually builds the simulator. A 4-8 week C++ development effort dwarfs the 1-2 week effort for Genesis or MJX Warp backends.

2. **The ceiling may not matter.** If training is limited by GPU memory, model computation, or sample efficiency rather than environment throughput, 10x faster environments provide no practical benefit over 3x faster.

3. **madrona_mjx deprecation.** The hybrid approach that would reduce development cost is no longer actively maintained.

4. **Research codebase risk.** API instability could invalidate development effort. The README warns: "missing features / documentation / bugs" and "breaking API changes."

### Verdict

The "highest ceiling" claim is **technically justified** but **practically misleading** for this project. Yes, a well-optimized Madrona native port would be the fastest option. But:
- The development cost is 3-5x higher than alternatives
- The performance gain over MJX Warp or Genesis may not translate to faster overall training time
- The risk of API breakage and debugging difficulty is significant

The claim should be qualified: "highest ceiling **if development cost is not a constraint and a C++ developer with GPU experience is available for 4-8 weeks.**"

---

## 8. Code-Level Concerns and Observations

### 8.1 Resolution Mismatch

The CUDA raytracer (`CudaBatchRenderConfig::renderResolution`) produces only **square** output. The comment at `include/madrona/mw_gpu.hpp:89-90` explicitly states:

> "The raytracer output is square so the resolution of the outputs would be renderResolution x renderResolution."

Memory Maze uses 64x64 (square), so this is not a problem for the standard benchmark. But the HD variant (256x256) may push the raytracer's memory budget. At 16K worlds with 256x256 RGBD, the output buffer alone would be `16K * 256 * 256 * 5 = ~20 GB`.

The Vulkan rasterizer path (`RenderManager::Config`) supports separate `agentViewWidth` and `agentViewHeight`, providing more flexibility if non-square resolutions are ever needed.

### 8.2 Contact Detection

The `PhysicsBackend.check_target_contact()` method checks whether the walker touches a target. In Madrona's physics system, this maps to `CollisionEvent` temporaries (`include/madrona/physics.hpp:95-100`):

```cpp
struct CollisionEvent {
    Entity a;
    Entity b;
};
struct CollisionEventTemporary : Archetype<CollisionEvent> {};
```

For Memory Maze, the targets are not physical objects — they use MuJoCo's `gap` parameter for proximity detection without collision forces. In Madrona, this is better implemented using `PhysicsSystem::findEntitiesWithinAABB()` (`include/madrona/physics.hpp:179-181`) or a simple distance check in a system function:

```cpp
void checkTargetProximity(MyContext &ctx, Position &walker_pos, ...) {
    // Distance-based check, no physics collision needed
    for (auto& target : targets) {
        float dist = (walker_pos - target.pos).length();
        if (dist < activation_radius) { /* activate */ }
    }
}
```

This is straightforward but requires implementing the target iteration pattern within a system function — not callable from Python.

### 8.3 Procedural Maze Generation on GPU

If running thousands of worlds, each needing a unique maze, the maze generation itself becomes a concern. Madrona provides:

- Per-world `RandKey` for RNG (`include/madrona/rand.hpp`)
- `tmpAlloc()` for per-step scratch memory (32 MB per world)

However, Memory Maze uses the `labmaze` library (written in C++/Python) for maze generation. This would need to be:
- **Pre-generated on CPU** and passed to each world via `WorldInit` structs (recommended)
- **Re-implemented in C++** for GPU-side generation (complex, rarely done)

The pre-generation approach is cleanest: generate N maze layouts on CPU, pack them into `WorldInit` data, and each world reads its assigned layout at startup. On episode reset, the world selects a new layout from a pre-generated pool stored in a Singleton or global buffer.

### 8.4 Navmesh Support

The architecture report's `analysis_madrona.md` mentions Madrona's navmesh support as a strength. Inspecting `include/madrona/navmesh.hpp`, it provides:

- Triangle-based navmesh with BFS and Dijkstra pathfinding
- Random point sampling on the navmesh surface
- Polygon adjacency traversal

This could be useful for the oracle controller's BFS pathfinding, but it's a GPU-side facility — not directly callable from the Python oracle wrapper. The oracle mode is for evaluation only, so this is not a critical concern.

### 8.5 JAX Integration Quality

The `JAXInterface::buildEntry<>()` template (`include/madrona/py/bindings.hpp`) is sophisticated:
- Supports XLA custom_call registration
- Token threading for ordered effects (prevents step elision by XLA)
- Checkpoint save/restore for PBT
- Separate CPU and GPU entry points

This is the most mature JAX integration of any batch simulation engine. If the project moves to JAX-based training (e.g., using PureJaxRL), Madrona's JAX interface would be a significant advantage over Genesis (PyTorch-centric).

---

## 9. Specific Recommendations

### 9.1 Do Not Implement `MadronaBackend(PhysicsBackend)`

The per-world, synchronous Python protocol is architecturally incompatible with Madrona. Forcing Madrona into this interface would:
- Execute one world at a time (negating batch parallelism)
- Require Python↔CUDA round-trips per method call
- Produce throughput **worse** than the current MuJoCo baseline

### 9.2 If Madrona is Desired, Use a Separate Integration Path

Rather than fitting Madrona into the `PhysicsBackend` protocol, build a standalone `MadronaMemoryMaze` project as a Madrona simulator (C++ submodule approach):

```
madrona_memory_maze/
    CMakeLists.txt          # Links against madrona submodule
    src/
        types.hpp           # ECS components, archetypes
        sim.hpp/cpp         # World data, system functions
        mgr.hpp/cpp         # Manager (Python-facing wrapper)
    scripts/
        train.py            # Direct training script (not via memory_maze package)
```

This bypasses the architecture report's modular structure entirely but delivers maximum performance. The training script would use Madrona's tensor export directly, not the `PhysicsBackend` → wrapper chain.

### 9.3 Recommended Path: Skip Madrona, Use MJX Warp

Given the trade-offs:
- **Development cost:** MJX Warp (3-6 weeks) vs. Madrona native (4-8 weeks) vs. Madrona hybrid (deprecated)
- **Physics fidelity:** MJX Warp = MuJoCo; Madrona native = approximation
- **Rendering throughput:** Both are GPU-accelerated; Madrona's batch renderer is faster, but MJX Warp's Warp renderer is also fast
- **Risk:** MJX Warp is actively maintained by DeepMind; Madrona is a research codebase

MJX Warp is the safer and more practical choice for this project. Madrona should only be considered if:
1. A C++ developer with GPU experience is available and willing to invest 4-8 weeks
2. Environment throughput is empirically proven to be the training bottleneck
3. The team is willing to maintain a C++ codebase alongside the Python training code

---

## 10. Summary Table

| Aspect | Assessment | Notes |
|--------|-----------|-------|
| PhysicsBackend compatibility | **Incompatible** | Per-world Python API vs. batch GPU execution |
| Rendering throughput | **Excellent** | 300K+ FPS at 64x64, purpose-built for this |
| madrona_mjx hybrid | **Deprecated** | MJWarp supersedes this approach |
| C++ development cost | **High** | 4-8 weeks, GPU debugging difficulty |
| Batch parallelism | **Excellent** | 4K-64K worlds on single GPU |
| Dynamic maze regeneration | **Solvable** | Pre-allocated wall pool or maze pool pattern |
| "Highest ceiling" claim | **Technically true** | But practically misleading given costs |
| Physics fidelity | **Poor** | XPBD is not MuJoCo; would need approximation |
| Platform support | **Linux-only (GPU)** | macOS/Windows = CPU backend only |
| Long-term maintenance risk | **High** | Research code, breaking API changes |
| **Overall fit score** | **4/10** | As a PhysicsBackend; 7/10 if hybrid were maintained |

---

*Review produced February 28, 2026. Based on source-level analysis of Madrona at `madrona/` (include headers, physics, rendering, ECS, Python bindings) and the architecture report at `memory-maze/architecture_report.md`.*
