# Engine Fit Synthesis: Which Engine for Memory Maze?

**Date:** February 28, 2026
**Based on:** Expert reviews from MuJoCo/MJX, Genesis, and Madrona specialists

---

## Executive Summary

Three engine experts independently reviewed the Memory Maze architecture report. Their findings converge on a clear ranking:

| Rank | Engine | Fit Score | Best For | Effort |
|------|--------|-----------|----------|--------|
| **1** | **MuJoCo CPU** | **9/10** | Validation backend, backward compat | 2-3 days |
| **2** | **Genesis** | **7/10** | Best ergonomics, Python-native batch sim | 2-4 weeks |
| **3** | **MJX Warp** | **5/10** | Physics-faithful GPU path | 3-6 weeks |
| **4** | **Madrona native** | **4/10** | Maximum throughput (if cost is no object) | 4-8 weeks |

**Recommendation:** Implement MuJoCo backend first (validation), then Genesis as the primary alternative backend.

---

## 1. Cross-Engine Comparison

### 1.1 Interface Compatibility

| Criterion | MuJoCo CPU | Genesis | MJX Warp | Madrona |
|-----------|-----------|---------|----------|---------|
| PhysicsBackend fit | Near-perfect | Good (with workarounds) | Awkward (stateful vs functional) | **Incompatible** |
| Batch support | N/A (multiprocessing) | Native but interface doesn't expose it | Native but interface doesn't expose it | Entire value proposition |
| compile_scene() per episode | ~50ms (same as current) | **Must NOT rebuild** — reposition entities only | 5-30s JIT miss (needs superset model) | Fundamentally wrong paradigm |
| Dynamic maze regeneration | Native (MJCF recompile) | Superset geometry + reposition | Superset MJCF + Model.replace() | Pre-allocated wall pool or maze pool |

**Key consensus finding:** All three GPU engine experts (Genesis, MJX, Madrona) independently identified the same two architectural gaps:

1. **The `PhysicsBackend` protocol is single-env only.** Every GPU engine's value comes from batch parallelism. The interface needs a `BatchPhysicsBackend` variant with tensor-shaped inputs/outputs `[n_envs, ...]`.

2. **`compile_scene()` must not mean "rebuild."** For GPU engines, scene compilation is a one-time startup cost. Per-episode variation must be handled by repositioning entities within a pre-built scene, not rebuilding.

### 1.2 Physics Fidelity

| Engine | Fidelity vs MuJoCo | Contact Detection | Notes |
|--------|-------------------|-------------------|-------|
| MuJoCo CPU | **Identical** | `gap`-based, accumulates across substeps | Same engine |
| MJX Warp | **Identical** | `gap` supported for primitive geoms | Same algorithms, GPU |
| Genesis | **Approximate** | Distance-based (`torch.norm`) | Different solver, needs tuning |
| Madrona | **Poor** | Distance-based (C++ system fn) | XPBD ≠ MuJoCo dynamics |

MuJoCo/MJX preserve exact physics. Genesis can approximate well (rolling ball is simple). Madrona would need a ground-up physics implementation with no guarantee of matching MuJoCo behavior.

### 1.3 Rendering

| Engine | Renderer | GPU Batch | Platform | Texture Support |
|--------|----------|-----------|----------|----------------|
| MuJoCo CPU | OpenGL (EGL) | No | All | Full |
| MJX Warp | Ray tracing (Warp) | Yes | NVIDIA only | Yes (warp >= 1.12) |
| Genesis | Rasterizer / BatchRenderer | BatchRenderer: CUDA+Linux | BatchRenderer: Linux; Rasterizer: all | Needs verification |
| Madrona | CUDA raytracer / Vulkan | Yes | GPU: Linux; CPU: all | Pre-baked images |

All experts agree: **rendering is the bottleneck**, not physics. Any GPU rendering path provides massive speedup over the current OpenGL baseline.

### 1.4 Development Risk

| Engine | Effort | Language | Debugging | Maintenance Risk |
|--------|--------|----------|-----------|-----------------|
| MuJoCo CPU | 2-3 days | Python | Easy | Low (stable API) |
| Genesis | 2-4 weeks | Python | Medium | Medium (active development) |
| MJX Warp | 3-6 weeks | Python/JAX | Medium-Hard | Medium (new renderer) |
| Madrona | 4-8 weeks | **C++** | **Very Hard** (GPU megakernel) | **High** (research code, breaking API) |

### 1.5 Target Visibility Workaround

A telling detail: **none of the alternative engines natively support hiding targets via alpha transparency**. All three experts converge on the same workaround — move "invisible" targets below the floor plane. This confirms the interface design is reasonable; the `set_target_visible()` method correctly abstracts over backend-specific tricks.

---

## 2. The Batch Interface Gap (Critical Finding)

The single most important finding across all reviews: **the `PhysicsBackend` protocol as designed captures single-env correctness but not GPU throughput.**

All three GPU engine experts independently proposed a `BatchPhysicsBackend`:

```python
class BatchPhysicsBackend(Protocol):
    def step(self, actions: np.ndarray) -> None:           # [n_envs, 2]
    def get_walker_positions(self) -> np.ndarray:           # [n_envs, 3]
    def check_target_contacts(self) -> np.ndarray:          # [n_envs, n_targets]
    def render_egocentric(self) -> np.ndarray:              # [n_envs, H, W, 3]
```

**Impact:** Without this, Genesis/MJX provide zero advantage over CPU MuJoCo. With it, 100-1000x throughput gains are achievable.

**Recommendation:** Add `BatchPhysicsBackend` to the architecture as a Phase 5 extension. The single-env `PhysicsBackend` remains correct for Phase 1-4 (MuJoCo validation). The batch protocol enables GPU backends to express their value.

---

## 3. Ranked Recommendations

### Rank 1: MuJoCo CPU Backend (Implement First)

**Why:** 9/10 fit. Near-trivial implementation wrapping existing dm_control code. Serves as the ground-truth validation backend. All tests must pass identically to the reference implementation before any alternative backend is attempted.

**Effort:** 2-3 days.

**Expert concerns addressed:**
- Ensure `check_target_contact()` accumulates across substeps (use `target.activated`)
- Call `mj_forward()` after state modifications, before rendering
- Type `MazeLayout.wall_segments` more precisely

### Rank 2: Genesis Backend (Primary Alternative)

**Why:** 7/10 fit. Best developer ergonomics — pure Python, PyTorch-native tensors, batch simulation built-in. The physics is overkill for a rolling ball but works fine. The main challenges (static scene, no visibility toggle) have known workarounds.

**Effort:** 2-4 weeks.

**Implementation strategy (from Genesis expert):**
1. Build superset scene once at startup with `max_walls` box entities (`batch_fixed_verts=True`)
2. Per-episode: reposition walls via `entity.set_pos(pos, envs_idx=...)`, unused walls below floor
3. Distance-based contact detection via `torch.norm()`
4. Camera attachment for egocentric view (verify `move_to_attach()` timing)
5. Target hiding via position displacement below floor

**Requires:** `BatchPhysicsBackend` interface to unlock batch simulation value.

**Platform constraint:** BatchRenderer (high throughput) is CUDA+Linux only. macOS development requires single-env Rasterizer fallback.

### Rank 3: MJX Warp Backend (If Physics Fidelity Is Critical)

**Why:** 5/10 fit for the current interface, but identical physics to MuJoCo. The Warp renderer provides GPU rendering with the exact same physics engine. Best choice if the research requires identical dynamics.

**Effort:** 3-6 weeks.

**Implementation strategy (from MuJoCo expert):**
1. Build fixed-size superset MJCF model (max geoms for 15x15 maze)
2. Per-episode: reconfigure via `Model.replace(geom_pos=new)` — no JIT recompilation since array shapes stay fixed
3. Warp backend exclusively (JAX backend cannot render)
4. Pin `warp-lang >= 1.12` for texture support
5. Design for batch from the start (`jax.vmap` over step function)

**Requires:** `BatchPhysicsBackend` interface. NVIDIA GPU mandatory. Warp renderer maturity risk.

### Rank 4: Madrona (Skip for Now)

**Why:** 4/10 fit. The PhysicsBackend protocol is fundamentally incompatible with Madrona's batch GPU megakernel execution model. The madrona_mjx hybrid that would have been the practical path is deprecated. Development requires 4-8 weeks of C++ from a GPU-experienced developer.

**When to reconsider:**
- Environment throughput is empirically proven to be the training bottleneck (not model compute or sample efficiency)
- A dedicated C++ developer with GPU experience is available
- The team is willing to maintain a separate C++ codebase

---

## 4. Suggested Architecture Amendments

Based on all three reviews, the architecture report should be amended with:

| Amendment | Priority | Source |
|-----------|----------|--------|
| Add `BatchPhysicsBackend` protocol for GPU engines | **P0** | All three experts |
| Document that GPU backends must NOT rebuild scene per episode | **P0** | Genesis + MJX experts |
| Clarify `check_target_contact()` accumulates across substeps | **P1** | MuJoCo expert |
| Type `MazeLayout.wall_segments` precisely (dataclass or typed tuple) | **P1** | MuJoCo expert |
| Add `set_target_visible()` guidance: position displacement is the portable pattern | **P1** | Genesis + Madrona experts |
| Note MJX `geom_rgba` alpha requires `Model.replace()` | **P2** | MuJoCo expert |
| Verify Genesis `set_pos()` on `fixed=True` + `batch_fixed_verts=True` entities | **P2** | Genesis expert |
| Document platform constraints (Warp=NVIDIA, BatchRenderer=CUDA+Linux) | **P2** | All experts |

---

## 5. Decision Matrix

For quick reference, here's the bottom line:

| If your priority is... | Choose... | Because... |
|------------------------|-----------|-----------|
| **Correctness & validation** | MuJoCo CPU | Same engine, identical output, 2-3 days |
| **Best developer experience** | Genesis | Python-native, PyTorch tensors, 2-4 weeks |
| **Physics-identical GPU training** | MJX Warp | Same physics, GPU rendering, 3-6 weeks |
| **Maximum theoretical throughput** | Madrona | 300K+ FPS but 4-8 weeks C++, high risk |
| **Practical best balance** | **MuJoCo first, then Genesis** | Validation + fast GPU path with manageable risk |

---

*Synthesis produced February 28, 2026. Based on independent expert reviews from MuJoCo/MJX, Genesis, and Madrona domain specialists.*
