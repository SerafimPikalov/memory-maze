# Benchmark Validity Analysis: GPU-Batched Memory Maze Options

**Date:** March 1, 2026
**Analyst:** maze-expert agent
**Context:** Evaluating Options A-D for Genesis GPU batch simulation against the Memory Maze benchmark's scientific requirements (paper: arXiv 2210.13383)

---

## 0. Reference Baseline: What the Original Benchmark Does

Before evaluating any option, we must establish exactly what the reference implementation guarantees and why.

**Maze diversity in the original:**
- `labmaze.RandomMaze.regenerate()` is called every episode (`maze.py:124,283`)
- labmaze uses a seeded PRNG internally; each `regenerate()` call produces a new layout
- Over 100M training steps, the agent encounters:
  - 9x9: **100,000 unique episodes** (100M / 1000 steps per episode)
  - 11x11: **50,000 unique episodes**
  - 13x13: **33,333 unique episodes**
  - 15x15: **25,000 unique episodes**
- With 2^31 possible seeds and constrained room-based generation, the probability of seeing the same layout twice is negligible. Effectively **every episode is a unique maze.**

**Why this matters -- the paper's own words (Section 3, design properties):**
> "Random maze generation ensures the agent cannot memorize specific layouts across episodes."

This is not an incidental implementation detail. It is the **central design principle** that makes Memory Maze a memory benchmark rather than a navigation benchmark. The agent's score correlates with how well it remembers **within** each episode, precisely because it cannot carry layout knowledge **between** episodes.

**The offline probing protocol** further depends on layout diversity:
- 30,000 trajectories, each in a unique maze
- The probe predicts `maze_layout` (binary grid) from learned representations
- If the probe saw the same layout multiple times, it could memorize maze-to-representation mappings rather than learning to decode general spatial structure

---

## 1. Option A -- Wall Pool (64 pre-allocated walls, repositioned per episode)

### Description
Pre-allocate 64 wall box entities at scene build time. Each episode, use `labmaze.RandomMaze` to generate a fresh layout, extract wall segments via `covering.make_walls()`, and reposition active walls via `set_pos()`. Unused walls are moved underground (z=-10). This is what the existing `genesis_backend.py` already implements for single-environment mode.

### Maze Diversity
**Unlimited.** Every episode generates a new random maze, identical to the reference implementation. The wall pool is purely an implementation mechanism -- it does not constrain layout diversity in any way.

Wall count survey (from `plan_genesis_port.md`):
| Maze Size | Max Wall Segments |
|-----------|------------------|
| 9x9       | 21               |
| 11x11     | 28               |
| 13x13     | 35               |
| 15x15     | 46               |

64 pre-allocated walls covers all sizes with margin.

### Benchmark Comparability
**Fully comparable.** The observation distribution, reward statistics, and episode dynamics are identical to the reference implementation (modulo physics/rendering fidelity differences, which are orthogonal to this analysis). Results can be directly compared against Table 3 of the paper.

### Offline Probing Compatibility
**Fully compatible.** Each trajectory has a unique maze layout, exactly as the offline dataset requires. A probe trained on this data evaluates the same generalization capability as the original.

### Statistical Validity
**Preserved.** No maze repetition means no risk of cross-episode memorization.

### Critical Technical Question
The question is whether `set_pos()` on 64 pre-allocated fixed boxes actually works correctly in Genesis batch mode:

1. **Does `entity.set_pos(pos, envs_idx=...)` update collision geometry?** The wall tunneling bug noted in `plan_genesis_port.md` suggests this may not work perfectly. If collision geometry is baked at `build()` time and not updated by `set_pos()`, the walls become visual-only.

2. **Does batch mode support different wall configurations per environment?** With `scene.build(n_envs=N)`, all N environments share the same entity set. Calling `wall_entities[i].set_pos(pos, envs_idx=torch.tensor([3]))` would need to set different positions for the same wall entity across different environment indices. This is the key API requirement.

3. **Performance of per-episode repositioning.** With 64 walls x N environments, each reset requires 64 `set_pos()` calls per environment. If resets are asynchronous (different environments reset at different times), this creates a serial bottleneck.

**If these technical requirements are met, Option A is unambiguously the best choice.** It preserves 100% benchmark validity with zero scientific compromise.

### Verdict: STRONGLY RECOMMENDED (if technically feasible in batch mode)

---

## 2. Option B -- Maze Pool (K=100 pre-built separate scenes)

### Description
Pre-generate K=100 maze layouts at startup. Build K separate Genesis scenes, one per layout. During training, cycle through scenes -- when an environment needs a new episode, assign it a randomly selected scene from the pool.

### Maze Diversity

**K=100 distinct layouts.** Over 100M training steps:

| Maze | Episodes | Times Each Maze Seen |
|------|----------|---------------------|
| 9x9  | 100,000  | 1,000               |
| 15x15| 25,000   | 250                 |

This is a 1000x reduction in diversity for 9x9 compared to the reference implementation (100,000 unique layouts vs 100).

**However**, within each layout there is additional randomization:
- Spawn position (chosen from rooms, typically 3-9 options)
- Target-to-position assignment (permutations of N targets across available rooms)
- Initial target selection

For 9x9 (3 targets, ~4 rooms): 24 target placements x 4 spawn positions = **96 configurations per layout**. With K=100 layouts, this gives ~9,600 unique episode configurations. Still far below the reference's 100,000.

### Benchmark Comparability: THE CORE PROBLEM

This is where the analysis becomes scientifically critical. The question is: **can an LSTM(256) or RSSM(2048) memorize 100 maze layouts?**

**Answer: almost certainly yes, and here is the concrete argument.**

An LSTM with 256 hidden units has 256 x 32 = 8,192 bits of state capacity. Identifying one of K=100 mazes requires only log2(100) = 6.6 bits. Even a naive lookup table mapping "maze identity -> optimal behavior" is trivially within the network's capacity.

But the real danger is subtler. The agent does not need to store an explicit lookup table. Through 100M steps of training, the network's weights (not just hidden state) will encode maze-specific knowledge. After seeing maze #47 a thousand times, the CNN encoder will have learned to recognize maze #47's distinctive visual pattern at spawn, and the LSTM will have learned what actions to take in that specific maze.

**This transforms the benchmark from a within-episode memory task to a cross-episode recognition task:**

| Property | Original Benchmark | K=100 Pool |
|----------|--------------------|------------|
| What the agent must learn | Explore + memorize layout each episode | Recognize which of 100 known layouts it's in |
| Cognitive analogy | Navigating a new building every day | Navigating your office building (one you've been to 1000 times) |
| LSTM role | Episodic memory (maintain and update spatial map) | Pattern recognition (match current view to stored template) |
| Score reflects | Quality of online spatial memory | Quality of recognition + recall from long-term storage |
| Information flow | Within-episode only | Across episodes (via weights) |

**The scores will be inflated.** An agent with poor episodic memory but good pattern recognition would score well on K=100 but poorly on the original benchmark. Conversely, an agent with strong episodic memory would score similarly on both. This means K=100 results are not comparable to Table 3 -- they test a different cognitive capability.

**Concrete prediction:** IMPALA on 9x9 would score above 23.4 (the paper's result) with K=100, because the LSTM can leverage cross-episode knowledge. The score inflation would be largest for agents with weak memory (like vanilla Dreamer, which scores only 28.2 on 9x9) because they benefit most from the cross-episode shortcut.

### Offline Probing Compatibility

**Compromised.** With only 100 unique layouts:

1. **Probe overfitting to known layouts.** The wall layout prediction probe (a 4-layer MLP with 1024 units per layer) has ~3M parameters. With only 100 unique wall layouts (each an 81-cell binary grid = 81 bits), the probe can trivially memorize all 100 layouts as a lookup table. The probe's wall prediction accuracy would be near-perfect regardless of whether the underlying representation actually encodes spatial structure.

2. **The 30K trajectory offline dataset** would contain 30,000/100 = 300 trajectories per layout. The probe would see each layout 300 times during training. This is massive overfitting risk.

3. **Mitigation: evaluation on held-out layouts.** If you train on K=80 and evaluate on K=20 held-out layouts, the probe score becomes meaningful again. But this requires the representation to generalize across layouts, which is closer to the original benchmark's intent.

### Statistical Validity

**Questionable at K=100.** The fundamental issue is that with 1000 repetitions per maze, the agent's behavior on maze #47 at episode 99,000 is not independent of its behavior on maze #47 at episode 1,000. Standard RL training assumes episodes are drawn i.i.d. from the environment distribution. With K=100, the effective environment distribution has only 100 atoms, and the agent visits each atom ~1000 times. This creates correlations in the training data that standard variance estimators (used for error bars in Table 3) do not account for.

### What K is Sufficient?

The key threshold is: **the agent should NOT see the same layout enough times to memorize it through the network's weights.**

A conservative bound: each maze should be seen at most ~5-10 times during training. Beyond that, weight-based memorization becomes plausible.

| K Required | 9x9 (100K eps) | 15x15 (25K eps) |
|------------|-----------------|------------------|
| Max 10 repeats | K >= 10,000 | K >= 2,500 |
| Max 5 repeats | K >= 20,000 | K >= 5,000 |
| Max 2 repeats | K >= 50,000 | K >= 12,500 |
| Original (1 repeat) | K >= 100,000 | K >= 25,000 |

**For 9x9, K >= 10,000 is the minimum for scientific credibility. K=100 is not defensible.**

But K=10,000 separate scenes means 10,000 x (scene memory) GPU memory. For Genesis, each scene contains ~64 walls + 1 walker + 6 targets + 1 floor = ~72 entities. This is likely infeasible on a single GPU.

### Verdict: INVALID at K=100. Requires K >= 10,000 for credibility, which is likely infeasible.

---

## 3. Option C -- Fixed Maze Per Batch (all parallel envs share one maze)

### Description
All N parallel environments in a GPU batch run the same maze layout. Only spawn position and target assignment are randomized between episodes. The maze layout changes only when the entire batch is reconstructed.

### Maze Diversity
**1 layout at any given time.** Even if you rotate layouts periodically, at any moment all environments experience the same maze.

### Benchmark Comparability

**Completely invalid as a Memory Maze benchmark result.** This is not an opinion; it follows directly from the benchmark's design goals.

The paper explicitly states that the memory challenge arises because "maze layout and object positions remain fixed within an episode" while "random maze generation ensures the agent cannot memorize specific layouts across episodes." Option C removes the second condition entirely.

With a fixed maze:
- The agent learns a fixed spatial map that never changes
- The LSTM's role reduces to tracking "which targets have I collected this episode"
- Navigation becomes pure reactive policy (turn left at junction A, go straight at junction B)
- The 9x9 task becomes nearly trivial -- a memoryless CNN policy could achieve near-oracle scores

**Concrete prediction:** Even a no-memory baseline (CNN without LSTM) would achieve scores close to the oracle on a fixed maze, because it can learn a fixed visual-to-action mapping. This completely eliminates the benchmark's purpose.

### When Option C is Acceptable

Option C is valid for **one purpose only**: debugging and validating the physics/rendering pipeline. If you want to verify that the Genesis walker moves correctly, that targets are detected, that rewards are computed -- a fixed maze is fine for that. It is not valid for any published benchmark result.

### Verdict: INVALID for benchmarking. Valid for development/debugging only.

---

## 4. Option D -- Maze Atlas (100-1000 mazes in one scene, agent teleports)

### Description
Build 100-1000 maze layouts into a single Genesis scene, spatially separated (e.g., maze i at x_offset = i * 100 meters). Each environment index is assigned to a different atlas slot. At episode reset, the agent teleports to a different maze within the atlas.

### Maze Diversity
**K = atlas size.** Same as Option B in terms of layout diversity.

### Technical Advantages over Option B
- Single scene = single `build()` call, simpler memory management
- No scene switching overhead
- All environments can use different mazes simultaneously via `envs_idx`
- Per-environment reset is just a position teleport

### Technical Risks
1. **GPU memory.** K=1000 mazes at ~64 walls each = 64,000 wall entities in one scene. Each wall is a box primitive, so the geometry is simple, but this is a very large entity count for Genesis's rigid solver.

2. **Rendering artifacts.** With mazes separated by 100m, the far clipping plane (50.0 in the current code) should prevent rendering other mazes. But shadow maps, skybox reflections, or other global rendering effects might leak between mazes. Needs verification.

3. **Physics isolation.** All mazes exist in the same physics world. A wall in maze #1 should not interact with the walker in maze #2, but if the spatial separation is insufficient, contact detection could produce spurious collisions. The separation must exceed the maximum possible contact distance.

4. **Collision broadphase performance.** With 64,000 boxes in one scene, the broadphase collision detection must handle a very large spatial extent. Most broadphase algorithms (sweep-and-prune, spatial hash) have cost proportional to the spatial extent or object count. This could significantly slow physics.

### Benchmark Validity
**Same as Option B.** The atlas is an implementation mechanism; what matters for benchmark validity is the pool size K. The same K-threshold analysis applies:

- K=100: invalid (agent memorizes layouts)
- K=1000: borderline (each maze seen ~100 times for 9x9)
- K=10,000: acceptable if feasible
- K >= 25,000: equivalent to original for 15x15

### Verdict: Equivalent to Option B scientifically. Better engineering than B if K fits in GPU memory.

---

## 5. Cross-Cutting Analysis

### 5.1 The Fundamental Tension

There is an unavoidable tension between two requirements:
1. **GPU batch simulation** requires static scene structure (fixed entity count, compiled kernels)
2. **Benchmark validity** requires unbounded maze diversity (new layout every episode)

Option A resolves this tension by repositioning entities within a static scene. Options B/D sacrifice diversity for engineering simplicity. Option C abandons the benchmark entirely.

### 5.2 What Exactly Does "Memorize" Mean?

It is important to distinguish three types of memorization:

**Type 1: Weight memorization (cross-episode, K-dependent).** The network's weights encode layout-specific knowledge from repeated exposure. This is what K constrains. With K=100, the CNN learns to recognize specific wall patterns at specific spawn locations. This is the primary threat to benchmark validity.

**Type 2: Hidden state memorization (within-episode, benchmark-intended).** The LSTM/RSSM accumulates spatial information during an episode. This is what the benchmark tests. The agent explores, encodes wall positions and target locations in its hidden state, and later uses this stored information for efficient navigation.

**Type 3: Skill memorization (cross-episode, K-independent).** The network learns general skills: wall-following, room-entering, target-approaching. This is beneficial and K-independent -- the agent should learn these skills regardless of layout diversity.

The benchmark is designed so that Type 2 is the bottleneck. Option A preserves this. Options B/D (at low K) make Type 1 the dominant strategy, circumventing the intended challenge.

### 5.3 Can Target Randomization Compensate for Fixed Layouts?

No, but it helps partially.

With a fixed layout, randomizing target positions and spawn location creates combinatorial variation:
- 9x9: 96 configurations per layout (24 target placements x 4 spawns)
- 15x15: 544,320 configurations per layout

This is substantial, especially for larger mazes. However, it does NOT restore the memory challenge because:

1. **The navigation structure is fixed.** The shortest paths between rooms are the same every episode. The agent can learn a fixed room-connectivity graph. Only "which target is in which room" changes.

2. **Visual recognition is layout-dependent.** The visual experience at each corridor junction is identical across episodes with the same layout. The CNN encoder can learn "this junction means turn left for room 3" as a fixed rule.

3. **Target randomization reduces to a simpler problem.** With 3 targets in 4 rooms on a known maze, the agent needs to remember only 3 room assignments (6 bits of information) rather than the full spatial layout (40+ bits). This is dramatically easier.

### 5.4 Offline Dataset Requirements

The paper's offline datasets contain 30,000 trajectories of 1000 steps each, totaling 30M steps. Each trajectory is in a unique maze.

For a ported benchmark to produce a compatible offline dataset:
- K must be >= 30,000 (one unique maze per trajectory), OR
- The dataset generation must use Option A (unlimited diversity), even if training uses a pool

If the dataset contains only K=100 unique layouts with 300 trajectories per layout, the probing results are meaningless (the probe memorizes layout-to-representation mappings).

### 5.5 Evaluation Protocol

Even if training uses a finite maze pool, **evaluation should always use fresh mazes** not seen during training. This is the standard train/test split principle.

However, with Option A, evaluation on fresh mazes is automatic. With Options B/D, you must explicitly generate a held-out evaluation set and verify the agent generalizes to unseen layouts. If it does not generalize, the training scores are inflated by memorization.

---

## 6. Recommended Strategy

### Priority 1: Make Option A work in batch mode.

The technical requirements are:
1. `entity.set_pos(pos, envs_idx=idx)` must update collision geometry per environment
2. Per-environment reset must support repositioning ~64 walls asynchronously
3. Performance: wall repositioning must not dominate the per-episode cost

Investigate Genesis's `scene.reset(envs_idx=...)` and entity-level per-environment state management. If `set_pos()` correctly updates the rigid solver's collision geometry per environment, Option A is clearly the best path.

### Priority 2: If Option A is infeasible, use Option D (Maze Atlas) with K >= 10,000.

The atlas approach avoids per-episode scene modification at the cost of GPU memory. Feasibility depends on:
- Memory per wall entity in Genesis (likely ~100 bytes of rigid body state)
- 10,000 mazes x 64 walls = 640,000 entities
- At 100 bytes each: ~64 MB of rigid body state (manageable)
- But rendering BVH, contact broadphase, etc. add overhead

If K=10,000 fits in GPU memory, the benchmark validity is acceptable for 15x15 (each maze seen ~2.5 times) and borderline for 9x9 (each maze seen ~10 times).

### Priority 3: If memory limits K to 1,000, acknowledge the limitation explicitly.

With K=1,000:
- Each 9x9 maze is seen 100 times. This is too many for strict benchmark validity.
- Mitigation: run a **held-out evaluation** on 1,000 fresh mazes not in the training pool. Report both in-pool and held-out scores. The held-out score measures generalization and is the scientifically valid number.
- This is similar to how supervised learning reports test accuracy, not training accuracy.

### What K is "safe" by maze size?

| Maze Size | Episodes (100M steps) | K for max 10 repeats | K for max 5 repeats |
|-----------|-----------------------|----------------------|---------------------|
| 9x9       | 100,000               | 10,000               | 20,000              |
| 11x11     | 50,000                | 5,000                | 10,000              |
| 13x13     | 33,333                | 3,333                | 6,667               |
| 15x15     | 25,000                | 2,500                | 5,000               |

### Never use Option C for published results.

Option C (fixed maze per batch) is a development tool, not a benchmarking configuration.

---

## 7. Summary Table

| Option | Diversity | Benchmark Valid? | Offline Probing? | Technical Complexity | Recommendation |
|--------|-----------|------------------|------------------|---------------------|----------------|
| **A: Wall Pool** | Unlimited | Yes | Yes | Medium (requires per-env entity repositioning) | **BEST** |
| **B: Maze Pool (K=100)** | 100 layouts | **No** -- agents memorize layouts | **No** -- probe overfits | Low | **INVALID** |
| **B: Maze Pool (K>=10K)** | 10K+ layouts | Acceptable | Acceptable with held-out eval | Medium (memory) | **ACCEPTABLE** |
| **C: Fixed Maze** | 1 layout | **No** -- eliminates memory challenge | **No** | Lowest | **INVALID** (dev only) |
| **D: Atlas (K=1000)** | 1000 layouts | Borderline -- report held-out scores | Borderline | Medium-High | **MARGINAL** |
| **D: Atlas (K>=10K)** | 10K+ layouts | Acceptable | Acceptable with held-out eval | High (memory) | **ACCEPTABLE** |

---

## 8. Key Takeaway

The Memory Maze benchmark's scientific contribution is isolating episodic memory as the performance bottleneck. Any implementation that allows cross-episode layout memorization -- through repeated exposure to a small pool of layouts -- fundamentally changes what the benchmark measures. Option A (wall repositioning) is the only approach that preserves the original benchmark's scientific properties at zero cost to diversity. Every other approach requires either very large K (>= 10,000) or explicit acknowledgment that results are not directly comparable to the paper.

The engineering effort to make Option A work in Genesis batch mode is well-justified by the scientific clarity it provides. If it proves technically infeasible, the Atlas approach (Option D) with K >= 10,000 and mandatory held-out evaluation is the scientifically defensible fallback.
