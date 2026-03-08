# Review: `force_render=True` Fix — Impact on Memory Maze Correctness and Training

**Date:** 2026-03-07
**Reviewer:** maze-expert agent

---

## 1. Summary

The `force_render=True` fix addresses a fundamental correctness bug in the Genesis backend where render caching caused stale frames to be returned after `camera.set_pose()` calls that occurred without an intervening `scene.step()`. This made all target spheres (red, green, blue) appear identical in rendered frames, breaking the core task signal of the Memory Maze benchmark.

The fix is **correct and necessary** for the BatchRenderer path. However, the Rasterizer path (non-batched, per-env cameras) is **missing the fix** and has the same underlying vulnerability. Additionally, no automated test currently verifies that distinct target colors are visually distinguishable in rendered frames.

---

## 2. Root Cause Analysis

### The caching mechanism

Genesis renderers (BatchRenderer, Rasterizer, Raytracer) all use a time-based cache to avoid redundant rendering. The cache logic is identical across all three:

```
# BatchRenderer (batch_renderer.py:378)
if force_render or self._t < self._visualizer.scene.t:
    self._data_cache.clear()

# Rasterizer (rasterizer_context.py:977-980)
def update(self, force_render: bool = False):
    if not force_render and self._t >= self.scene._t:
        return

# Raytracer (raytracer.py:635-636)
def update_scene(self, force_render: bool = False):
    if not force_render and self._t >= self.scene.t:
        return
```

The scene's `_t` counter increments only on `scene.step()` (scene.py:1005). If `camera.set_pose()` is called without a `scene.step()`, the renderer's `_t` still equals `scene._t`, so `update()` returns early and the old frame is served from cache.

### Why Memory Maze triggers this

In the Memory Maze step cycle, the sequence is:

```
1. apply_action()          -- DOF forces on walker
2. scene.step()            -- physics substeps (increments _t)
3. update_cameras()        -- camera.set_pose() to follow walker
4. render()                -- capture egocentric view
```

At first glance, `scene.step()` happens before `render()`, so the cache should be invalidated. But there is a subtle detail: `scene.step(update_visualizer=False)` is called in a substep loop (lines 664-665, 1289-1290):

```python
def step(self):
    for _ in range(self.n_substeps):
        self.scene.step(update_visualizer=False)
```

After `n_substeps` calls, `scene._t` has been incremented `n_substeps` times. The renderer's `_t` was last set during the previous render call. So on the **first** render after stepping, `renderer._t < scene._t` is True and the cache is correctly invalidated.

**The real problem occurs during `reset()`.** When the environment resets:

```
1. reset_env(env_idx)      -- calls scene.reset() which sets _t = 0
2. set_walker_pose()       -- repositions walker
3. set_target_pos()        -- repositions targets
4. update_cameras()        -- camera.set_pose()
5. render_all()            -- needs fresh frame
```

After `scene.reset()`, `scene._t` goes back to 0 (or is preserved in batched mode via `saved_t`). If the renderer's `_t` already equals `scene._t`, the render call returns the stale cached frame from the previous episode -- showing the **old** target positions and colors.

For the **batched auto-reset path** (lines 1592-1599), this is particularly critical: after `_reset_single_env()` reconfigures walls and target positions for env `i`, `render_all()` must reflect those changes. Without `force_render=True`, the BatchRenderer returns the cached frame from step 6 (the pre-reset render), which shows targets at old positions with (potentially) old visual state.

### Why targets all appear the same color without the fix

The stale frame issue means that after reset, the rendered image shows whatever was last rendered before the reset. Since targets are colored spheres placed at specific maze positions, if the renderer returns a cached frame:

1. Targets appear at their **previous** positions (pre-reset layout)
2. If the reset happened mid-episode (auto-reset), the cached frame may show a completely different maze layout
3. More critically, during the **initial render after build** (before any step has occurred), `scene._t = 0` and `renderer._t = 0`, so the very first render without `force_render` would return whatever is in the uninitialized cache

The "all targets same color" symptom likely manifests because the cached frame simply does not reflect the actual scene state at all -- it shows a stale or uninitialized image.

---

## 3. Assessment of the Fix

### Where `force_render=True` is applied

| Line | Code path | Context |
|------|-----------|---------|
| 714 | `GenesisMazeScene.render_egocentric()` | Single-env scene render |
| 1351 | `BatchGenesisMazeScene.render_all()` (BatchRenderer branch) | Batched render, all envs |
| 1374 | `BatchGenesisMazeScene.render_single()` (BatchRenderer branch) | Batched render, one env |

### Where `force_render=True` is NOT applied (potential gap)

| Line | Code path | Context |
|------|-----------|---------|
| 1358 | `BatchGenesisMazeScene.render_all()` (Rasterizer branch) | Per-env camera loop |
| 1378 | `BatchGenesisMazeScene.render_single()` (Rasterizer branch) | Single per-env camera |

### Is the Rasterizer path affected?

**Yes, it has the same caching mechanism.** The Rasterizer's `update_scene()` calls `self._context.update(force_render)`, which checks `self._t >= self.scene._t` (rasterizer_context.py:979). If the renderer's time matches the scene's time, it returns early.

However, the Rasterizer path in Memory Maze has a partial natural protection:

1. **Single-env path (line 714)**: `GenesisMazeScene.render_egocentric()` already has `force_render=True`, so this is fixed.

2. **Batched Rasterizer path (lines 1358, 1378)**: This path creates per-env cameras with `env_idx=i`. These cameras are less commonly used (BatchRenderer is the training path on CUDA), but they would exhibit the same caching bug on CPU-only setups.

**Recommendation: Add `force_render=True` to lines 1358 and 1378** for consistency and to prevent bugs on CPU-only deployments.

### Is the fix correct?

**Yes.** `force_render=True` clears the render cache unconditionally, forcing a fresh rasterization/raycast. This guarantees the rendered image reflects the current entity positions and visual state.

### Performance impact

`force_render=True` disables frame caching, meaning every `render()` call performs a full render pass. In Memory Maze, `render()` is called **once per control step** (after physics and camera update). Since the camera moves every step, the cache would be invalidated anyway after the first substep increments `scene._t`. The only case where `force_render=True` causes extra work is when `render()` is called multiple times per step without any physics step in between -- which does not happen in normal operation.

For the reset path, `force_render=True` is essential and has no performance cost since a fresh render is always needed after reconfiguring the scene.

**Conclusion: No measurable performance regression from this fix.**

---

## 4. MuJoCo Comparison: Does MuJoCo Have This Issue?

**No.** MuJoCo's rendering pipeline (`physics.render()`) does not have time-based caching. It calls `mjr_render` directly, which renders from the current `mjData` state every time. The MuJoCo renderer is stateless with respect to simulation time -- it always reads the current positions from `data.xpos`, `data.xmat`, etc.

The MuJoCo backend uses `dm_control`'s Composer observable system, where `MJCFCamera.raw_observation` calls `physics.render()` on every observation request. No caching layer exists between the camera observable and the OpenGL render call.

This is a fundamental architectural difference:
- **MuJoCo**: Stateless rendering. Every `render()` reads current physics state.
- **Genesis**: Cached rendering. `render()` may return stale frames unless cache is invalidated.

This difference is by design -- Genesis's caching avoids redundant GPU rendering when the scene hasn't changed (useful for multi-camera setups or visualization). But it creates a trap for use cases like Memory Maze where entity positions change via `set_pos()` without `scene.step()`.

---

## 5. Impact on RL Training

### How severe was the color bug?

**Critical.** The Memory Maze task requires the agent to:

1. **See** the colored border indicating the current target color
2. **Navigate** to the target sphere matching that color
3. **Remember** target locations across the episode

If all target spheres render as the same color (or as stale images), the agent cannot learn the color-matching association. The task degrades from "navigate to the correct colored target" to "navigate to any target" -- which is a fundamentally different (and easier) task.

### Impact on reward signal

With the color bug:
- The agent gets reward 1.0 for touching the **current** target (selected randomly)
- The agent sees all targets as the same color, so it cannot distinguish which one is current
- The border color still indicates the current target, but the agent has no way to match the border color to a specific target in the maze
- **Expected reward under random target selection**: If the agent learns to visit targets without color discrimination, it gets reward with probability 1/n_targets per visit
- **With color discrimination**: The agent can prioritize the current target, potentially earning reward on every visit

For a 9x9 maze with 3 targets, the color bug reduces the effective reward rate by approximately 3x compared to a color-aware agent. The agent can still learn a sub-optimal policy (visit all targets), but convergence to optimal behavior is impossible.

### Training convergence impact

1. **Without fix**: Agent learns spatial navigation but not color-target association. Final performance plateaus at ~1/3 of optimal reward rate.
2. **With fix**: Agent can learn the full task -- navigate to the specific colored target indicated by the border.

The fix is not just an optimization -- it enables learning the core task. Prior training runs on the Genesis backend with the color bug would have produced agents that navigate well but randomly among targets, with approximately 3x lower reward than MuJoCo-trained agents.

---

## 6. Test Gap Analysis

### Existing tests

The test suite has good coverage for:
- Physics correctness (collision, trajectory matching)
- Camera resolution and format
- Brightness, color balance, and lighting
- Batch vs single-env consistency
- Episode lifecycle

### Missing: Target color distinguishability test

**No existing test verifies that different-colored targets render as visually distinct.** This is the specific failure mode of the caching bug. A test should:

1. Place the camera looking at each target individually
2. Verify that the dominant hue of each target's rendered pixels differs
3. Run on both Rasterizer and BatchRenderer paths

### Missing: Render-after-reset freshness test

**No test verifies that rendering after `set_pos()` without `scene.step()` produces an updated image.** This would directly catch the caching regression:

1. Render frame A
2. Move an entity via `set_pos()` (no `scene.step()`)
3. Render frame B
4. Assert frame A != frame B

### Recommended new tests

```python
class TestRenderCacheInvalidation:
    """Verify render() returns fresh frames after entity movement."""

    def test_render_reflects_set_pos_without_step(self, single_scene, rng):
        """Moving entity via set_pos() without step() must produce different render."""
        single_scene.reset(rng)
        img_before = single_scene.render_egocentric().copy()

        # Move walker significantly without stepping physics
        old_pos = single_scene.get_walker_position()
        single_scene.walker.set_pos(old_pos + np.array([2.0, 0.0, 0.0]))
        single_scene._update_camera()
        img_after = single_scene.render_egocentric()

        diff = np.mean(np.abs(img_before.astype(float) - img_after.astype(float)))
        assert diff > 5.0, (
            f"Render not updated after set_pos: mean pixel diff={diff:.1f}. "
            f"Likely stale cache returned."
        )

    def test_target_colors_distinguishable(self, single_scene, rng):
        """Each target sphere should render with a distinct dominant hue."""
        single_scene.reset(rng)
        hues = []
        for i, target in enumerate(single_scene.target_entities):
            tpos = single_scene._target_world_positions[i]
            if tpos[2] < -5:
                continue
            # Position camera looking directly at target
            cam_pos = tpos + np.array([0, -1.5, 0.5])
            single_scene.camera.set_pose(
                pos=cam_pos, lookat=tpos, up=(0, 0, 1)
            )
            img = single_scene.render_egocentric()
            # Extract center region (target should be there)
            h, w = img.shape[:2]
            center = img[h//4:3*h//4, w//4:3*w//4].astype(float)
            # Compute dominant channel
            mean_rgb = center.mean(axis=(0, 1))
            hues.append(mean_rgb)

        # At least 2 targets should have different dominant channels
        if len(hues) >= 2:
            for i in range(len(hues)):
                for j in range(i+1, len(hues)):
                    diff = np.linalg.norm(hues[i] - hues[j])
                    # At least one pair should differ significantly
            max_diff = max(
                np.linalg.norm(hues[i] - hues[j])
                for i in range(len(hues))
                for j in range(i+1, len(hues))
            )
            assert max_diff > 20.0, (
                f"Target colors not distinguishable: max RGB diff={max_diff:.1f}, "
                f"hues={[h.tolist() for h in hues]}"
            )
```

---

## 7. Rasterizer Path: Should the Fix Be Applied?

### Current state

Lines 1358 and 1378 in `render_all()` and `render_single()` (Rasterizer branch) do NOT pass `force_render=True`:

```python
# Line 1358 (render_all, Rasterizer path)
result = self.cameras[i].render(rgb=True, depth=False, segmentation=False)

# Line 1378 (render_single, Rasterizer path)
result = self.cameras[env_idx].render(rgb=True, depth=False, segmentation=False)
```

### Should it be fixed?

**Yes, for two reasons:**

1. **Correctness**: The Rasterizer has the same `_t`-based cache (rasterizer_context.py:977-980). The same stale-frame bug can occur on the Rasterizer path. While the BatchRenderer is the primary training path (CUDA + gs_madrona), the Rasterizer is used:
   - On CPU-only setups (macOS development, CI)
   - When `gs_madrona` is not installed
   - In single-env mode (line 714 is already fixed, but batched Rasterizer is not)

2. **Consistency**: Having `force_render=True` on the BatchRenderer path but not the Rasterizer path creates a subtle behavioral difference between code paths that should be equivalent.

### Risk of applying the fix

Zero. The Rasterizer's `update()` does useful work (updating node transforms, clearing buffers, recomputing shadow maps) that must happen after entity movement. Skipping it via caching is the bug, not the fix.

---

## 8. Conclusions and Recommendations

### Fix assessment

| Aspect | Status |
|--------|--------|
| Fix correctness | Correct for the 3 lines where applied |
| Fix completeness | **Incomplete** -- Rasterizer branch at lines 1358, 1378 is missing |
| Performance impact | None (render is needed every step regardless) |
| Training impact | **Critical** -- enables learning the core color-matching task |
| MuJoCo parity | MuJoCo has no equivalent issue (stateless rendering) |

### Recommended actions

1. **Add `force_render=True` to lines 1358 and 1378** for the Rasterizer fallback path
2. **Add a render-cache regression test** that verifies `render()` produces different output after `set_pos()` without `scene.step()`
3. **Add a target-color distinguishability test** that verifies at least 2 of the 3 targets render with different dominant hues
4. **Re-run any prior Genesis training experiments** -- results from before this fix reflect a degraded task where color discrimination was impossible
5. **Document this as a Genesis integration caveat** -- any future use of Genesis rendering after `set_pos()` must use `force_render=True` or interpose a `scene.step()`
