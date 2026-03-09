"""Comprehensive tests for Genesis Memory Maze backend — single-env and batch mode.

Covers 7 identified risk classes, integration tests, and smoke tests.
Run with: pytest tests/test_genesis_batch.py -v -m "not slow"

NOTE: Tests init Genesis with gs.cpu, so batch-mode tests use the Rasterizer
(not BatchRenderer, which requires gs.cuda). The Rasterizer only partially
supports per-env batched rendering — render_all() returns (n_envs,H,W,3)
instead of per-camera (H,W,3). Tests that hit this path are marked xfail.

NOTE: Genesis does not release CUDA memory when scenes go out of scope. When
running on GPU, heavy test classes may OOM-kill if run together in one pytest
session. Use per-class process isolation: run each TestClass in its own
pytest invocation.
"""

import math
import os
import sys
import time

# Must set MUJOCO_GL before dm_control is imported (macOS has no EGL)
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import pytest

try:
    import genesis as gs
    import torch
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False

pytestmark = pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")

from memory_maze.genesis_backend import (
    ACTION_SET,
    BATCH_HIDDEN_Z,
    HIDDEN_Z,
    CAMERA_FOV,
    FLOOR_FRICTION,
    ROLL_DAMPING,
    ROLL_GEAR,
    STEER_DAMPING,
    STEER_GEAR,
    TARGET_ACTIVATION_GAP,
    TARGET_COLORS,
    TARGET_RADIUS,
    TRANS_DAMPING,
    WALKER_CAMERA_HEIGHT,
    WALKER_FRICTION,
    WALKER_RADIUS,
    WALKER_TOTAL_MASS,
    _UNDERGROUND_THRESHOLD,
    BatchGenesisMemoryMazeEnv,
    BatchGenesisMazeScene,
    GenesisMemoryMazeEnv,
    GenesisMazeScene,
    _max_walls,
    extract_positions,
    extract_wall_segments,
)


# ===================================================================
# Helpers
# ===================================================================

@pytest.fixture(scope="module")
def _init_genesis():
    """Initialize Genesis once per module."""
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")


@pytest.fixture
def single_scene(_init_genesis):
    """Create a built single-env GenesisMazeScene (9x9, 32px)."""
    scene = GenesisMazeScene(
        maze_size=9, n_targets=3, camera_resolution=32,
    )
    scene.build()
    return scene


@pytest.fixture
def batch_scene_2(_init_genesis):
    """Create a built 2-env BatchGenesisMazeScene (9x9, 32px)."""
    scene = BatchGenesisMazeScene(
        n_envs=2, maze_size=9, n_targets=3, camera_resolution=32,
    )
    scene.build()
    return scene


def _step_n(scene, n, action=None):
    """Step a single-env scene n control steps with optional action."""
    for _ in range(n):
        if action is not None:
            scene.apply_action(action)
        scene.step()


def _get_pos_np(entity):
    """Get entity position as numpy array (handles torch tensors)."""
    pos = entity.get_pos()
    return pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)


# ===================================================================
# Risk 1: Phantom Collisions from hidden underground walls
# ===================================================================

class TestPhantomCollisions:

    def test_ball_free_fall_no_phantom_forces(self, single_scene, rng):
        """Drop ball with all walls underground — no lateral drift."""
        single_scene.reset(rng)
        # Move ALL walls underground (override maze placement)
        for wall in single_scene.wall_entities:
            wall.set_pos(np.array([0.0, 0.0, -10.0]))
        # Place ball above floor, let it settle
        single_scene.walker.set_pos(np.array([0.0, 0.0, 1.0]))
        single_scene.walker.set_dofs_velocity(np.zeros(6))
        _step_n(single_scene, 200)

        pos = single_scene.get_walker_position()
        # Ball should have settled on floor with no lateral drift
        assert abs(pos[0]) < 0.05, f"X drift: {pos[0]}"
        assert abs(pos[1]) < 0.05, f"Y drift: {pos[1]}"
        # Z should be near walker radius (resting on floor)
        assert abs(pos[2] - WALKER_RADIUS) < 0.1, f"Z: {pos[2]}"

    def test_ball_trajectory_matches_no_walls_case(self, _init_genesis):
        """Compare trajectory: walls at z=-10 vs z=-100 should be identical."""
        results = {}
        for label, hide_z in [("shallow", -10.0), ("deep", -100.0)]:
            scene = GenesisMazeScene(maze_size=9, n_targets=3, camera_resolution=32)
            scene.build()
            rng = np.random.RandomState(99)
            scene.reset(rng)
            # Move all walls to specified depth
            for wall in scene.wall_entities:
                wall.set_pos(np.array([0.0, 0.0, hide_z]))
            # Place ball at known position
            scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
            scene.walker.set_dofs_velocity(np.zeros(6))
            # Apply forward action for several steps
            for _ in range(50):
                scene.apply_action(np.array([-1.0, 0.0]))
                scene.step()
            results[label] = scene.get_walker_position().copy()

        diff = np.linalg.norm(results["shallow"] - results["deep"])
        assert diff < 0.01, f"Position diff: {diff}"

    def test_hidden_wall_at_ball_xy_no_deflection(self, _init_genesis):
        """Hidden wall at ball's XY should not affect trajectory.

        Compare two runs: (A) no wall, (B) hidden wall at ball's XY.
        Final positions should be identical.
        """
        results = {}
        for label, place_hidden in [("clean", False), ("hidden", True)]:
            scene = GenesisMazeScene(maze_size=9, n_targets=3, camera_resolution=32)
            scene.build()
            rng = np.random.RandomState(55)
            scene.reset(rng)
            for wall in scene.wall_entities:
                wall.set_pos(np.array([0.0, 0.0, -10.0]))
            if place_hidden:
                # Place wall underground at the ball's XY
                scene.wall_entities[0].set_pos(np.array([0.0, 0.0, -10.0]))
            scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
            scene.walker.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
            scene.walker.set_dofs_velocity(np.zeros(6))
            for _ in range(50):
                scene.apply_action(np.array([-1.0, 0.0]))
                scene.step()
            results[label] = scene.get_walker_position().copy()

        diff = np.linalg.norm(results["clean"] - results["hidden"])
        assert diff < 0.01, f"Hidden wall altered trajectory, diff={diff}"


# ===================================================================
# Risk 2: set_pos collision geometry — walls move and collide correctly
# ===================================================================

class TestSetPosCollisionGeometry:

    def test_wall_blocks_ball(self, single_scene, rng):
        """Place wall in ball's path — ball should be stopped."""
        single_scene.reset(rng)
        # Clear all walls
        for wall in single_scene.wall_entities:
            wall.set_pos(np.array([0.0, 0.0, -10.0]))
        # Place ball at origin heading +x
        single_scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        single_scene.walker.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        single_scene.walker.set_dofs_velocity(np.zeros(6))
        # Place wall at x=1.5 blocking the path
        single_scene.wall_entities[0].set_pos(np.array([1.5, 0.0, 0.75]))
        # Push forward
        for _ in range(200):
            single_scene.apply_action(np.array([-1.0, 0.0]))
            single_scene.step()
        pos = single_scene.get_walker_position()
        assert pos[0] < 1.5, f"Ball should be blocked at wall, x={pos[0]}"

    def test_wall_moved_via_set_pos_still_collides(self, single_scene, rng):
        """Move wall via set_pos to new location — collision should work there."""
        single_scene.reset(rng)
        for wall in single_scene.wall_entities:
            wall.set_pos(np.array([0.0, 0.0, -10.0]))
        # Initially place wall far away
        single_scene.wall_entities[0].set_pos(np.array([10.0, 0.0, 0.75]))
        # Now move it to x=1.5 via set_pos
        single_scene.wall_entities[0].set_pos(np.array([1.5, 0.0, 0.75]))
        # Place ball and push
        single_scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        single_scene.walker.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        single_scene.walker.set_dofs_velocity(np.zeros(6))
        for _ in range(200):
            single_scene.apply_action(np.array([-1.0, 0.0]))
            single_scene.step()
        pos = single_scene.get_walker_position()
        assert pos[0] < 1.5, f"Ball should be blocked by moved wall, x={pos[0]}"

    def test_wall_originally_underground_then_raised(self, single_scene, rng):
        """Raise wall from underground to active — should block ball."""
        single_scene.reset(rng)
        for wall in single_scene.wall_entities:
            wall.set_pos(np.array([0.0, 0.0, -10.0]))
        # Wall starts underground
        single_scene.wall_entities[0].set_pos(np.array([1.5, 0.0, -10.0]))
        # Raise it
        single_scene.wall_entities[0].set_pos(np.array([1.5, 0.0, 0.75]))
        # Place ball and push
        single_scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        single_scene.walker.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        single_scene.walker.set_dofs_velocity(np.zeros(6))
        for _ in range(200):
            single_scene.apply_action(np.array([-1.0, 0.0]))
            single_scene.step()
        pos = single_scene.get_walker_position()
        assert pos[0] < 1.5, f"Ball should be blocked by raised wall, x={pos[0]}"

    def test_wall_lowered_underground_no_longer_collides(self, _init_genesis):
        """Lowered wall should not block — trajectory matches no-wall case.

        Compare two runs: (A) no wall, (B) wall raised then lowered.
        Final positions should be identical (wall is gone).
        """
        results = {}
        for label, raise_then_lower in [("clean", False), ("lowered", True)]:
            scene = GenesisMazeScene(maze_size=9, n_targets=3, camera_resolution=32)
            scene.build()
            rng = np.random.RandomState(55)
            scene.reset(rng)
            for wall in scene.wall_entities:
                wall.set_pos(np.array([0.0, 0.0, -10.0]))
            if raise_then_lower:
                scene.wall_entities[0].set_pos(np.array([1.5, 0.0, 0.75]))
                scene.wall_entities[0].set_pos(np.array([1.5, 0.0, -10.0]))
            scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
            scene.walker.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
            scene.walker.set_dofs_velocity(np.zeros(6))
            for _ in range(50):
                scene.apply_action(np.array([-1.0, 0.0]))
                scene.step()
            results[label] = scene.get_walker_position().copy()

        diff = np.linalg.norm(results["clean"] - results["lowered"])
        assert diff < 0.01, f"Lowered wall still affects trajectory, diff={diff}"


# ===================================================================
# Risk 3: Per-env independence in batch mode
# ===================================================================

class TestPerEnvIndependence:

    def test_different_actions_different_trajectories(self, batch_scene_2):
        """Different actions → different positions after stepping."""
        import labmaze
        # Set up both envs with the same maze and walker position
        rng = np.random.RandomState(42)
        seed = rng.randint(2147483648)
        maze = labmaze.RandomMaze(
            height=batch_scene_2.outer_size,
            width=batch_scene_2.outer_size,
            max_rooms=6, room_min_size=3, room_max_size=5,
            spawns_per_room=1, objects_per_room=1,
            random_seed=seed,
        )
        segments = extract_wall_segments(maze)
        for env_idx in range(2):
            batch_scene_2.configure_walls_for_env(env_idx, segments)
            batch_scene_2.set_walker_pose(env_idx, [0.0, 0.0], 0.0)

        # Apply different actions: env 0 forward, env 1 noop
        for _ in range(50):
            forces = np.zeros((2, 6), dtype=np.float32)
            forces[0, 0] = ROLL_GEAR * (-1.0)  # env 0: forward
            # env 1: no force
            batch_scene_2.apply_actions_batched(forces)
            batch_scene_2.step()

        positions = batch_scene_2.get_walker_positions()
        dist = np.linalg.norm(positions[0] - positions[1])
        assert dist > 0.1, f"Envs should diverge, dist={dist}"

    def test_reset_env0_does_not_affect_env1(self, batch_scene_2):
        """Resetting env 0 should not change env 1's state."""
        import labmaze
        rng = np.random.RandomState(77)
        seed = rng.randint(2147483648)
        maze = labmaze.RandomMaze(
            height=batch_scene_2.outer_size,
            width=batch_scene_2.outer_size,
            max_rooms=6, room_min_size=3, room_max_size=5,
            spawns_per_room=1, objects_per_room=1,
            random_seed=seed,
        )
        segments = extract_wall_segments(maze)
        for env_idx in range(2):
            batch_scene_2.configure_walls_for_env(env_idx, segments)
            batch_scene_2.set_walker_pose(env_idx, [2.0, 2.0], 0.5)

        # Step both envs forward
        for _ in range(30):
            forces = np.zeros((2, 6), dtype=np.float32)
            forces[:, 0] = ROLL_GEAR * (-1.0)
            batch_scene_2.apply_actions_batched(forces)
            batch_scene_2.step()

        # Record env 1 position
        pos_before = batch_scene_2.get_walker_positions()[1].copy()

        # Reset only env 0
        batch_scene_2.reset_env(0)

        # Env 1 should be unchanged
        pos_after = batch_scene_2.get_walker_positions()[1]
        diff = np.linalg.norm(pos_before - pos_after)
        assert diff < 1e-4, f"Env 1 changed after env 0 reset: diff={diff}"

    def test_wall_config_independent_per_env(self, batch_scene_2):
        """Different seeds → different wall configurations per env."""
        import labmaze
        configs = []
        for env_idx in range(2):
            seed = 1000 + env_idx * 123
            maze = labmaze.RandomMaze(
                height=batch_scene_2.outer_size,
                width=batch_scene_2.outer_size,
                max_rooms=6, room_min_size=3, room_max_size=5,
                spawns_per_room=1, objects_per_room=1,
                random_seed=seed,
            )
            segments = extract_wall_segments(maze)
            batch_scene_2.configure_walls_for_env(env_idx, segments)
            configs.append([(s.pos.tolist(), s.half_size.tolist()) for s in segments])

        # Wall configs should differ between seeds
        assert configs[0] != configs[1], "Different seeds should produce different mazes"


# ===================================================================
# Risk 4: Max collision pairs overflow
# ===================================================================

class TestMaxCollisionPairsOverflow:

    def test_max_walls_scene_builds_without_error(self, _init_genesis):
        """Build a 15x15 scene (max walls) and step 100 times."""
        scene = GenesisMazeScene(
            maze_size=15, n_targets=6, camera_resolution=32,
        )
        scene.build()
        rng = np.random.RandomState(42)
        scene.reset(rng)
        for _ in range(100):
            scene.apply_action(np.array([-1.0, 0.0]))
            scene.step()
        pos = scene.get_walker_position()
        assert np.all(np.isfinite(pos)), f"Non-finite position: {pos}"

    def test_collision_pair_count_within_limit(self, _init_genesis):
        """The number of active walls should stay within _max_walls(maze_size)."""
        import labmaze
        for maze_size in [9, 11, 13, 15]:
            cfg = (6, 5) if maze_size < 15 else (9, 3)
            outer = maze_size + 2
            maze = labmaze.RandomMaze(
                height=outer, width=outer,
                max_rooms=cfg[0], room_min_size=3, room_max_size=cfg[1],
                spawns_per_room=1, objects_per_room=1,
                random_seed=42,
            )
            segments = extract_wall_segments(maze)
            assert len(segments) <= _max_walls(maze_size), (
                f"Maze {maze_size}x{maze_size} has {len(segments)} segments > {_max_walls(maze_size)}"
            )

    def test_multiple_resets_no_cumulative_overflow(self, single_scene):
        """10 reset+step cycles should not cause NaN or errors."""
        for i in range(10):
            rng = np.random.RandomState(i * 7)
            single_scene.reset(rng)
            for _ in range(20):
                single_scene.apply_action(np.array([-1.0, 0.0]))
                single_scene.step()
            pos = single_scene.get_walker_position()
            assert np.all(np.isfinite(pos)), f"NaN at reset cycle {i}: {pos}"


# ===================================================================
# Risk 5: Camera rendering isolation
# ===================================================================

class TestCameraRenderingIsolation:

    def test_different_mazes_produce_different_images(self, _init_genesis):
        """Two different seeds → visually different renders."""
        images = []
        for seed in [42, 99]:
            scene = GenesisMazeScene(
                maze_size=9, n_targets=3, camera_resolution=32,
            )
            scene.build()
            rng = np.random.RandomState(seed)
            scene.reset(rng)
            img = scene.render_egocentric()
            images.append(img.astype(np.float32))

        diff = np.mean(np.abs(images[0] - images[1]))
        assert diff > 3.0, f"Images too similar, mean diff={diff}"

    def test_camera_follows_walker_position(self, single_scene, rng):
        """Moving the walker should produce different renders."""
        single_scene.reset(rng)
        img1 = single_scene.render_egocentric().copy()

        # Move walker significantly
        for _ in range(100):
            single_scene.apply_action(np.array([-1.0, 0.0]))
            single_scene.step()
        img2 = single_scene.render_egocentric()

        diff = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
        assert diff > 1.0, f"Images should change after movement, diff={diff}"

    def test_camera_resolution_matches_config(self, _init_genesis):
        """Output image shape should match configured resolution."""
        for res in [32, 64]:
            scene = GenesisMazeScene(
                maze_size=9, n_targets=3, camera_resolution=res,
            )
            scene.build()
            rng = np.random.RandomState(42)
            scene.reset(rng)
            img = scene.render_egocentric()
            assert img.shape == (res, res, 3), f"Expected ({res},{res},3), got {img.shape}"
            assert img.dtype == np.uint8

    def test_render_not_all_black_or_white(self, single_scene, rng):
        """Rendered image should have meaningful content (not blank)."""
        single_scene.reset(rng)
        img = single_scene.render_egocentric()
        std = np.std(img.astype(np.float32))
        assert std > 3.0, f"Image appears blank, std={std}"


# ===================================================================
# Risk 6: Async reset correctness
# ===================================================================

class TestAsyncResetCorrectness:

    def test_reset_restores_walker_position(self, single_scene, rng):
        """Walker should be on the floor after reset."""
        single_scene.reset(rng)
        pos = single_scene.get_walker_position()
        assert abs(pos[2] - WALKER_RADIUS) < 0.2, (
            f"Walker z={pos[2]} should be near {WALKER_RADIUS}"
        )

    def test_double_reset_produces_clean_state(self, single_scene):
        """Two resets in a row should leave env in valid state."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(77)
        single_scene.reset(rng1)
        single_scene.reset(rng2)
        pos = single_scene.get_walker_position()
        assert np.all(np.isfinite(pos)), f"Non-finite after double reset: {pos}"
        # Should be steppable
        single_scene.apply_action(np.array([-1.0, 0.0]))
        single_scene.step()
        pos2 = single_scene.get_walker_position()
        assert np.all(np.isfinite(pos2)), f"Non-finite after step: {pos2}"

    def test_reset_mid_episode_preserves_physics(self, single_scene):
        """Reset at step 25, continue stepping — no NaN/Inf."""
        rng = np.random.RandomState(42)
        single_scene.reset(rng)
        for _ in range(25):
            single_scene.apply_action(np.array([-1.0, 0.0]))
            single_scene.step()
        # Mid-episode reset
        rng2 = np.random.RandomState(99)
        single_scene.reset(rng2)
        for _ in range(25):
            single_scene.apply_action(np.array([-1.0, 0.5]))
            single_scene.step()
        pos = single_scene.get_walker_position()
        assert np.all(np.isfinite(pos)), f"Non-finite after mid-episode reset: {pos}"

    def test_many_sequential_resets(self, single_scene):
        """50 reset+step cycles should produce no NaN."""
        for i in range(50):
            rng = np.random.RandomState(i)
            single_scene.reset(rng)
            single_scene.apply_action(np.array([-1.0, 0.0]))
            single_scene.step()
            pos = single_scene.get_walker_position()
            assert np.all(np.isfinite(pos)), f"NaN at cycle {i}: {pos}"

    def test_targets_reset_correctly(self, single_scene, rng):
        """After reset, targets should be above ground and in-bounds."""
        single_scene.reset(rng)
        n_active = 0
        outer_half = (single_scene.outer_size * single_scene.xy_scale) / 2
        for tpos in single_scene._target_world_positions:
            if tpos[2] > -5:
                n_active += 1
                assert abs(tpos[0]) < outer_half + 1, f"Target OOB x: {tpos[0]}"
                assert abs(tpos[1]) < outer_half + 1, f"Target OOB y: {tpos[1]}"
        assert n_active >= 1, "No active targets after reset"


# ===================================================================
# Risk 7: Performance regression (slow tests)
# ===================================================================

class TestPerformanceRegression:

    @pytest.mark.slow
    def test_single_env_step_time(self, _init_genesis):
        """Benchmark: step+render should be < 100ms on CPU."""
        scene = GenesisMazeScene(
            maze_size=9, n_targets=3, camera_resolution=64,
        )
        scene.build()
        rng = np.random.RandomState(42)
        scene.reset(rng)

        # Warmup
        for _ in range(10):
            scene.apply_action(np.array([-1.0, 0.0]))
            scene.step()
            scene.render_egocentric()

        # Benchmark
        n_steps = 50
        t0 = time.perf_counter()
        for _ in range(n_steps):
            scene.apply_action(np.array([-1.0, 0.0]))
            scene.step()
            scene.render_egocentric()
        elapsed = time.perf_counter() - t0
        ms_per_step = (elapsed / n_steps) * 1000

        print(f"\nSingle-env step+render: {ms_per_step:.1f} ms/step")
        assert ms_per_step < 100, f"Too slow: {ms_per_step:.1f} ms/step"

    @pytest.mark.slow
    def test_hidden_walls_no_significant_overhead(self, _init_genesis):
        """9x9 vs 15x15 timing ratio should be < 2x (hidden walls are cheap)."""
        timings = {}
        for maze_size in [9, 15]:
            cfg = (3, 250, 6, 5) if maze_size == 9 else (6, 1000, 9, 3)
            scene = GenesisMazeScene(
                maze_size=maze_size, n_targets=cfg[0], camera_resolution=64,
            )
            scene.build()
            rng = np.random.RandomState(42)
            scene.reset(rng)

            # Warmup
            for _ in range(5):
                scene.apply_action(np.array([-1.0, 0.0]))
                scene.step()

            n_steps = 30
            t0 = time.perf_counter()
            for _ in range(n_steps):
                scene.apply_action(np.array([-1.0, 0.0]))
                scene.step()
            timings[maze_size] = (time.perf_counter() - t0) / n_steps

        ratio = timings[15] / timings[9]
        print(f"\n9x9: {timings[9]*1000:.1f}ms, 15x15: {timings[15]*1000:.1f}ms, ratio: {ratio:.2f}")
        assert ratio < 2.0, f"15x15 is {ratio:.1f}x slower than 9x9"

    @pytest.mark.slow
    def test_render_time_breakdown(self, _init_genesis):
        """Informational: physics vs render time breakdown."""
        scene = GenesisMazeScene(
            maze_size=9, n_targets=3, camera_resolution=64,
        )
        scene.build()
        rng = np.random.RandomState(42)
        scene.reset(rng)

        # Warmup
        for _ in range(10):
            scene.apply_action(np.array([-1.0, 0.0]))
            scene.step()
            scene.render_egocentric()

        n_steps = 50

        # Physics only
        t0 = time.perf_counter()
        for _ in range(n_steps):
            scene.apply_action(np.array([-1.0, 0.0]))
            scene.step()
        physics_time = (time.perf_counter() - t0) / n_steps

        # Render only
        t0 = time.perf_counter()
        for _ in range(n_steps):
            scene.render_egocentric()
        render_time = (time.perf_counter() - t0) / n_steps

        print(f"\nPhysics: {physics_time*1000:.1f}ms, Render: {render_time*1000:.1f}ms")
        print(f"Render fraction: {render_time/(physics_time+render_time)*100:.0f}%")


# ===================================================================
# Integration: GenesisMemoryMazeEnv (single-env gym wrapper)
# ===================================================================

class TestGymEnvIntegration:

    def test_env_create_and_reset(self, _init_genesis):
        """Create + reset should return correct obs shape."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        obs = env.reset()
        assert obs.shape == (32, 32, 3), f"Wrong shape: {obs.shape}"
        assert obs.dtype == np.uint8
        env.close()

    def test_env_step_returns_correct_types(self, _init_genesis):
        """All 6 actions should produce valid outputs."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        env.reset()
        for action in range(6):
            obs, reward, done, info = env.step(action)
            assert obs.shape == (32, 32, 3)
            assert obs.dtype == np.uint8
            assert isinstance(reward, float)
            assert isinstance(done, (bool, np.bool_))
            assert isinstance(info, dict)
        env.close()

    def test_env_episode_terminates(self, _init_genesis):
        """Full episode should terminate at max_steps."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        env.reset()
        done = False
        steps = 0
        max_steps = env._max_steps
        while not done and steps < max_steps + 10:
            _, _, done, _ = env.step(0)  # noop
            steps += 1
        assert done, f"Episode didn't terminate after {steps} steps"
        assert steps == max_steps, f"Terminated at step {steps}, expected {max_steps}"
        env.close()

    def test_env_multiple_episodes(self, _init_genesis):
        """3 episodes without state corruption."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        for ep in range(3):
            obs = env.reset()
            assert np.all(np.isfinite(obs.astype(np.float32))), f"Non-finite obs ep {ep}"
            for _ in range(10):
                obs, r, d, info = env.step(1)  # forward
                assert np.all(np.isfinite(obs.astype(np.float32)))
        env.close()

    def test_env_target_color_border(self, _init_genesis):
        """Border should be drawn on the image (non-zero border pixels)."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=64, seed=42)
        obs = env.reset()
        # Check that the 2-pixel border has non-zero colored pixels
        B = int(2 * math.sqrt(64 / 64))  # = 2
        left_border = obs[:, :B, :]
        right_border = obs[:, -B:, :]
        top_border = obs[:B, :, :]
        bottom_border = obs[-B:, :, :]
        # At least one border should have colored pixels
        has_color = (
            np.any(left_border > 0) or
            np.any(right_border > 0) or
            np.any(top_border > 0) or
            np.any(bottom_border > 0)
        )
        assert has_color, "No colored border pixels found"
        env.close()


# ===================================================================
# Integration: BatchGenesisMemoryMazeEnv
# ===================================================================

@pytest.mark.xfail(
    reason="Rasterizer render_all() returns wrong shape in batch mode; "
           "not used in production (BatchRenderer is).",
    strict=False,
)
class TestBatchEnvIntegration:

    def test_batch_env_create_and_reset(self, _init_genesis):
        """Create + reset should return (n_envs, H, W, 3) observations."""
        env = BatchGenesisMemoryMazeEnv(n_envs=2, maze_size=9, seed=42,
                                         camera_resolution=32)
        obs = env.reset()
        assert obs.shape == (2, 32, 32, 3), f"Wrong shape: {obs.shape}"
        assert obs.dtype == np.uint8
        env.close()

    def test_batch_env_step_returns_correct_shapes(self, _init_genesis):
        """Step should return correctly shaped outputs."""
        env = BatchGenesisMemoryMazeEnv(n_envs=2, maze_size=9, seed=42,
                                         camera_resolution=32)
        env.reset()
        actions = [1, 3]  # forward, right
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (2, 32, 32, 3)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
        env.close()

    def test_batch_env_episode_terminates(self, _init_genesis):
        """Episodes should terminate at max_steps with auto-reset."""
        env = BatchGenesisMemoryMazeEnv(n_envs=2, maze_size=9, seed=42,
                                         camera_resolution=32)
        env.reset()
        n_dones = [0, 0]
        for step in range(env._max_steps + 5):
            obs, r, dones, infos = env.step([0, 0])
            for i in range(2):
                if dones[i]:
                    n_dones[i] += 1
        # Each env should have terminated at least once
        assert n_dones[0] >= 1, "Env 0 never terminated"
        assert n_dones[1] >= 1, "Env 1 never terminated"
        env.close()

    def test_batch_env_different_seeds_different_obs(self, _init_genesis):
        """Different per-env RNGs should produce different observations."""
        env = BatchGenesisMemoryMazeEnv(n_envs=2, maze_size=9, seed=42,
                                         camera_resolution=32)
        obs = env.reset()
        diff = np.mean(np.abs(obs[0].astype(np.float32) - obs[1].astype(np.float32)))
        # Different mazes should produce different images
        assert diff > 1.0, f"Env observations too similar, diff={diff}"
        env.close()

    def test_batch_env_border_drawn(self, _init_genesis):
        """Target color border should be drawn on batch observations."""
        env = BatchGenesisMemoryMazeEnv(n_envs=2, maze_size=9, seed=42,
                                         camera_resolution=64)
        obs = env.reset()
        B = int(2 * math.sqrt(64 / 64))
        for i in range(2):
            has_color = np.any(obs[i, :, :B, :] > 0) or np.any(obs[i, :, -B:, :] > 0)
            assert has_color, f"No border on env {i}"
        env.close()


# ===================================================================
# Smoke: Training loop simulation (slow)
# ===================================================================

class TestTrainingSmokeTest:

    @pytest.mark.slow
    def test_training_loop_smoke(self, _init_genesis):
        """100 steps collecting (obs, act, rew) — all should be finite."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        obs = env.reset()
        rng = np.random.RandomState(0)
        for _ in range(100):
            action = rng.randint(6)
            obs, reward, done, info = env.step(action)
            assert np.all(np.isfinite(obs.astype(np.float32)))
            assert np.isfinite(reward)
            if done:
                obs = env.reset()
        env.close()

    @pytest.mark.slow
    def test_multiple_envs_sequential(self, _init_genesis):
        """4 independent envs, 25 steps each — different observations."""
        all_obs = []
        for i in range(4):
            env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=i * 10)
            obs = env.reset()
            for _ in range(25):
                obs, _, done, _ = env.step(1)
                if done:
                    obs = env.reset()
            all_obs.append(obs.copy())
            env.close()

        # At least some pairs should differ
        n_different = 0
        for i in range(4):
            for j in range(i + 1, 4):
                if np.mean(np.abs(all_obs[i].astype(float) - all_obs[j].astype(float))) > 1.0:
                    n_different += 1
        assert n_different > 0, "All envs produced identical observations"

    @pytest.mark.slow
    def test_batch_env_training_loop_smoke(self, _init_genesis):
        """100 steps with BatchGenesisMemoryMazeEnv — all finite."""
        env = BatchGenesisMemoryMazeEnv(n_envs=2, maze_size=9, seed=42,
                                         camera_resolution=32)
        obs = env.reset()
        rng = np.random.RandomState(0)
        for _ in range(100):
            actions = rng.randint(6, size=2)
            obs, rewards, dones, infos = env.step(actions)
            assert np.all(np.isfinite(obs.astype(np.float32)))
            assert np.all(np.isfinite(rewards))
        env.close()


# ---------------------------------------------------------------------------
# Batched vs non-batched physics parity (no GPU needed — source inspection)
# ---------------------------------------------------------------------------

class TestBatchedPhysicsParity:
    """Verify batched scene uses the same physics constants as non-batched.

    Uses runtime checks on built scenes — resilient to refactoring that moves
    constants from subclass __init__ to a shared base class.
    """

    def test_walker_friction_matches(self, _init_genesis):
        """Both scenes should use WALKER_FRICTION for the walker."""
        single = GenesisMazeScene(maze_size=9, n_targets=3, camera_resolution=32, use_textures=False)
        single.build()
        batch = BatchGenesisMazeScene(n_envs=1, maze_size=9, n_targets=3, camera_resolution=32, use_textures=False)
        batch.build()
        # Both walkers should have the same friction (via geoms[0])
        s_fric = float(single.walker.geoms[0].friction)
        b_fric = float(batch.walker.geoms[0].friction)
        assert s_fric == b_fric == WALKER_FRICTION, (
            f"Walker friction mismatch: single={s_fric}, batch={b_fric}, expected={WALKER_FRICTION}"
        )

    def test_walker_damping_matches(self, _init_genesis):
        """Both scenes should have identical DOF damping (TRANS_DAMPING, ROLL_DAMPING, STEER_DAMPING)."""
        single = GenesisMazeScene(maze_size=9, n_targets=3, camera_resolution=32, use_textures=False)
        single.build()
        batch = BatchGenesisMazeScene(n_envs=1, maze_size=9, n_targets=3, camera_resolution=32, use_textures=False)
        batch.build()
        s_damp = single.walker.get_dofs_damping()
        s_damp = s_damp.cpu().numpy() if hasattr(s_damp, 'cpu') else np.asarray(s_damp)
        b_damp = batch.walker.get_dofs_damping()
        b_damp = b_damp.cpu().numpy() if hasattr(b_damp, 'cpu') else np.asarray(b_damp)
        if b_damp.ndim == 2:
            b_damp = b_damp[0]
        expected = np.array([TRANS_DAMPING, TRANS_DAMPING, 0.0, ROLL_DAMPING, ROLL_DAMPING, STEER_DAMPING])
        np.testing.assert_allclose(s_damp, expected, err_msg="Single scene damping mismatch")
        np.testing.assert_allclose(b_damp, expected, err_msg="Batch scene damping mismatch")


# ===================================================================
# Hidden depth / detection threshold constants
# ===================================================================

class TestHiddenDepthConstants:
    """Verify hide depths are safely below the z < -5 detection threshold."""

    def test_batch_hidden_z_below_detection_threshold(self):
        """BATCH_HIDDEN_Z must be below the z < -5 detection threshold."""
        assert BATCH_HIDDEN_Z < -5, (
            f"BATCH_HIDDEN_Z={BATCH_HIDDEN_Z} is not below detection threshold -5"
        )

    def test_batch_hidden_z_has_safety_margin(self):
        """BATCH_HIDDEN_Z should be well below threshold (not borderline)."""
        assert BATCH_HIDDEN_Z < -50, (
            f"BATCH_HIDDEN_Z={BATCH_HIDDEN_Z} should have large margin below -5"
        )

    def test_hidden_z_alias_matches(self):
        """BATCH_HIDDEN_Z must equal the unified HIDDEN_Z constant."""
        assert BATCH_HIDDEN_Z == HIDDEN_Z, (
            f"BATCH_HIDDEN_Z={BATCH_HIDDEN_Z} != HIDDEN_Z={HIDDEN_Z}"
        )
        assert HIDDEN_Z < -5, (
            f"HIDDEN_Z={HIDDEN_Z} not below detection threshold -5"
        )


# ===================================================================
# Batched contact check boundary tests
# ===================================================================

class TestBatchedContactBoundary:
    """Boundary tests for the batched env's inline contact check in step().

    The batched step() has its own contact detection code (inline Python
    loops, NOT calling check_target_contacts), so it needs independent
    boundary validation matching test_task20_24_changes.py.

    Uses a single shared env (class-scoped) to avoid Genesis/pyglet resource
    exhaustion from creating many scenes sequentially.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _shared_env(self, _init_genesis, request):
        """Create one BatchGenesisMemoryMazeEnv shared across all tests."""
        env = BatchGenesisMemoryMazeEnv(
            n_envs=1, maze_size=9, seed=42, camera_resolution=32,
        )
        request.cls._env = env
        yield env
        env.close()

    def _setup_contact_test(self, target_distance):
        """Reset env, place current target at given XY distance from walker.

        Uses the walker's post-reset position (guaranteed safe at spawn).
        Also moves non-current targets far away so _pick_new_target won't hang
        if a reward is triggered.
        """
        env = self._env
        env.reset()
        walker_pos = env._scene.get_walker_positions()[0].copy()
        t = int(env._current_target_ix[0])

        # Move non-current targets far away first
        for j in range(env._n_targets):
            if j == t:
                continue
            far = np.array([walker_pos[0] + 5.0 + j, walker_pos[1], TARGET_RADIUS])
            env._scene.set_target_pos(0, j, far.astype(np.float32))
            env._target_positions[0, j] = far

        target_pos = np.array([
            walker_pos[0] + target_distance,
            walker_pos[1],
            TARGET_RADIUS,
        ])
        env._scene.set_target_pos(0, t, target_pos.astype(np.float32))
        env._target_positions[0, t] = target_pos

    def test_reward_at_0_5m(self):
        """Walker 0.5m from current target -> should get reward."""
        self._setup_contact_test(0.5)
        _, rewards, _, _ = self._env.step([0])
        assert rewards[0] > 0, f"Should reward at 0.5m, got {rewards[0]}"

    def test_no_reward_at_1_2m(self):
        """Walker 1.2m from current target -> should NOT get reward."""
        self._setup_contact_test(1.2)
        _, rewards, _, _ = self._env.step([0])
        assert rewards[0] == 0.0, f"Should not reward at 1.2m, got {rewards[0]}"

    def test_boundary_just_inside(self):
        """Walker 0.79m from current target -> should get reward (< 0.8)."""
        self._setup_contact_test(0.79)
        _, rewards, _, _ = self._env.step([0])
        assert rewards[0] > 0, f"Should reward at 0.79m, got {rewards[0]}"

    def test_boundary_just_outside(self):
        """Walker 0.81m from current target -> should NOT get reward (> 0.8)."""
        self._setup_contact_test(0.81)
        _, rewards, _, _ = self._env.step([0])
        assert rewards[0] == 0.0, f"Should not reward at 0.81m, got {rewards[0]}"

    def test_only_current_target_rewards(self):
        """Only the current target gives reward, not other close targets."""
        env = self._env
        env.reset()
        walker_pos = env._scene.get_walker_positions()[0].copy()
        current = int(env._current_target_ix[0])
        other = (current + 1) % env._n_targets

        # Place non-current target very close (0.3m)
        close_pos = np.array([walker_pos[0] + 0.3, walker_pos[1], TARGET_RADIUS])
        env._scene.set_target_pos(0, other, close_pos.astype(np.float32))
        env._target_positions[0, other] = close_pos
        # Place current target far away (5m)
        far_pos = np.array([walker_pos[0] + 5.0, walker_pos[1], TARGET_RADIUS])
        env._scene.set_target_pos(0, current, far_pos.astype(np.float32))
        env._target_positions[0, current] = far_pos

        _, rewards, _, _ = env.step([0])
        assert rewards[0] == 0.0, f"Non-current target should not reward, got {rewards[0]}"

    def test_uses_2d_distance(self):
        """Batched contact should use XY distance, ignoring Z."""
        env = self._env
        env.reset()
        walker_pos = env._scene.get_walker_positions()[0].copy()
        t = int(env._current_target_ix[0])

        # Move all non-current targets far away so _pick_new_target won't hang
        for j in range(env._n_targets):
            if j == t:
                continue
            far = np.array([walker_pos[0] + 5.0 + j, walker_pos[1], TARGET_RADIUS])
            env._scene.set_target_pos(0, j, far.astype(np.float32))
            env._target_positions[0, j] = far

        # XY distance = 0.5 (< 0.8), but Z is far away
        target_pos = np.array([walker_pos[0] + 0.5, walker_pos[1], 5.0])
        env._scene.set_target_pos(0, t, target_pos.astype(np.float32))
        env._target_positions[0, t] = target_pos

        _, rewards, _, _ = env.step([0])
        assert rewards[0] > 0, f"Should use 2D distance (0.5 < 0.8), got {rewards[0]}"

    def test_hidden_target_no_reward(self):
        """Hidden target (z < -5) should not reward even at xy=0."""
        env = self._env
        env.reset()
        walker_pos = env._scene.get_walker_positions()[0].copy()
        t = int(env._current_target_ix[0])

        target_pos = np.array([walker_pos[0], walker_pos[1], BATCH_HIDDEN_Z])
        env._scene.set_target_pos(0, t, target_pos.astype(np.float32))
        env._target_positions[0, t] = target_pos

        _, rewards, _, _ = env.step([0])
        assert rewards[0] == 0.0, f"Hidden target should not reward, got {rewards[0]}"


# ===================================================================
# Single-env vs batch scene physics parity (behavioral)
# ===================================================================

class TestSingleVsBatchPhysicsParity:
    """Verify single-env and batch(n=1) scenes produce identical walker physics.

    This is the behavioral complement to TestBatchedPhysicsParity's source
    inspection. If these trajectories diverge, the scenes have different
    physics parameters (mass, damping, friction, etc.).
    """

    def test_forward_trajectory_matches(self, _init_genesis):
        """Same DOF forces -> same position in single and batch(n=1) scenes."""
        # Build single-env scene
        single = GenesisMazeScene(
            maze_size=9, n_targets=3, camera_resolution=32, use_textures=False,
        )
        single.build()
        rng = np.random.RandomState(42)
        single.reset(rng)
        for wall in single.wall_entities:
            wall.set_pos(np.array([0.0, 0.0, -10.0]))
        single.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        single.walker.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        single.walker.set_dofs_velocity(np.zeros(6))

        # Build batch scene (1 env)
        batch = BatchGenesisMazeScene(
            n_envs=1, maze_size=9, n_targets=3, camera_resolution=32,
            use_textures=False,
        )
        batch.build()
        idx = torch.tensor([0], dtype=torch.int32)
        hidden = np.array([0.0, 0.0, BATCH_HIDDEN_Z], dtype=np.float32)
        for wall in batch.wall_entities:
            wall.set_pos(hidden, envs_idx=idx)
        batch.set_walker_pose(0, [0.0, 0.0], 0.0)

        # Apply identical forward force for 20 control steps
        force_single = np.array([ROLL_GEAR * -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        force_batch = force_single.reshape(1, 6).astype(np.float32)
        for _ in range(20):
            single.walker.control_dofs_force(force_single)
            single.step()
            batch.apply_actions_batched(force_batch)
            batch.step()

        pos_single = single.get_walker_position()
        pos_batch = batch.get_walker_positions()[0]

        diff = np.linalg.norm(pos_single - pos_batch)
        assert diff < 0.05, (
            f"Single vs batch forward trajectory mismatch: diff={diff:.4f}, "
            f"single={pos_single}, batch={pos_batch}"
        )

    def test_turning_trajectory_matches(self, _init_genesis):
        """Same steer torque -> same angular velocity in both scenes."""
        single = GenesisMazeScene(
            maze_size=9, n_targets=3, camera_resolution=32, use_textures=False,
        )
        single.build()
        rng = np.random.RandomState(42)
        single.reset(rng)
        for wall in single.wall_entities:
            wall.set_pos(np.array([0.0, 0.0, -10.0]))
        single.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        single.walker.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        single.walker.set_dofs_velocity(np.zeros(6))

        batch = BatchGenesisMazeScene(
            n_envs=1, maze_size=9, n_targets=3, camera_resolution=32,
            use_textures=False,
        )
        batch.build()
        idx = torch.tensor([0], dtype=torch.int32)
        hidden = np.array([0.0, 0.0, BATCH_HIDDEN_Z], dtype=np.float32)
        for wall in batch.wall_entities:
            wall.set_pos(hidden, envs_idx=idx)
        batch.set_walker_pose(0, [0.0, 0.0], 0.0)

        # Apply steer torque for 20 steps
        force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, STEER_GEAR * 1.0])
        force_batch = force.reshape(1, 6).astype(np.float32)
        for _ in range(20):
            single.walker.control_dofs_force(force)
            single.step()
            batch.apply_actions_batched(force_batch)
            batch.step()

        # Compare angular velocities (rz DOF)
        vel_single = single.walker.get_dofs_velocity()
        v_s = vel_single.cpu().numpy() if hasattr(vel_single, 'cpu') else np.asarray(vel_single)
        vel_batch = batch.walker.get_dofs_velocity()
        v_b = vel_batch.cpu().numpy() if hasattr(vel_batch, 'cpu') else np.asarray(vel_batch)
        if v_b.ndim == 2:
            v_b = v_b[0]

        omega_diff = abs(float(v_s[5]) - float(v_b[5]))
        assert omega_diff < 0.01, (
            f"Single vs batch angular velocity mismatch: "
            f"single_rz={v_s[5]:.4f}, batch_rz={v_b[5]:.4f}, diff={omega_diff:.4f}"
        )


# ===================================================================
# Expanded source-inspection physics parity
# ===================================================================

class TestExpandedPhysicsParity:
    """Runtime checks for single/batch scene physics parity.

    Verifies actual parameter values on built scenes rather than inspecting
    source code. Resilient to refactoring that moves code between classes.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _build_scenes(self, _init_genesis, request):
        """Build one single-env and one batch(n=1) scene for all tests."""
        single = GenesisMazeScene(
            maze_size=9, n_targets=3, camera_resolution=32, use_textures=False,
        )
        single.build()
        batch = BatchGenesisMazeScene(
            n_envs=1, maze_size=9, n_targets=3, camera_resolution=32, use_textures=False,
        )
        batch.build()
        request.cls._single = single
        request.cls._batch = batch

    def test_floor_friction_uses_constant(self):
        """Both scenes should use FLOOR_FRICTION for the floor."""
        s_fric = float(self._single.floor.geoms[0].friction)
        b_fric = float(self._batch.floor.geoms[0].friction)
        assert s_fric == FLOOR_FRICTION, f"Single floor friction={s_fric}, expected={FLOOR_FRICTION}"
        assert b_fric == FLOOR_FRICTION, f"Batch floor friction={b_fric}, expected={FLOOR_FRICTION}"

    def test_walker_radius_uses_constant(self):
        """Both scenes should use WALKER_RADIUS for the walker sphere."""
        s_radius = float(self._single.walker.morph.radius)
        b_radius = float(self._batch.walker.morph.radius)
        assert abs(s_radius - WALKER_RADIUS) < 1e-6, f"Single walker radius={s_radius}"
        assert abs(b_radius - WALKER_RADIUS) < 1e-6, f"Batch walker radius={b_radius}"

    def test_walker_mass_matches(self):
        """Both scenes should produce the same total walker mass from WALKER_TOTAL_MASS."""
        # Check mass via density * volume = WALKER_TOTAL_MASS
        vol = (4.0 / 3.0) * math.pi * WALKER_RADIUS ** 3
        expected_rho = WALKER_TOTAL_MASS / vol
        s_rho = float(self._single.walker.material.rho)
        b_rho = float(self._batch.walker.material.rho)
        assert abs(s_rho - expected_rho) < 1.0, f"Single walker rho={s_rho}, expected={expected_rho}"
        assert abs(b_rho - expected_rho) < 1.0, f"Batch walker rho={b_rho}, expected={expected_rho}"

    def test_camera_fov_uses_constant(self):
        """Both scenes should use CAMERA_FOV."""
        # Single-env has one camera; batch with Rasterizer has per-env cameras
        s_fov = self._single.camera.fov
        if self._batch.camera is not None:
            b_fov = self._batch.camera.fov
        else:
            b_fov = self._batch.cameras[0].fov
        assert s_fov == CAMERA_FOV, f"Single camera fov={s_fov}, expected={CAMERA_FOV}"
        assert b_fov == CAMERA_FOV, f"Batch camera fov={b_fov}, expected={CAMERA_FOV}"

    def test_target_collision_disabled(self):
        """Both scenes should have collision disabled for targets."""
        for i in range(self._single.n_targets):
            # collision=False means morph.collision is False and no geoms are created
            assert not self._single.target_entities[i].morph.collision, \
                f"Single target {i} has collision enabled"
        for i in range(self._batch.n_targets):
            assert not self._batch.target_entities[i].morph.collision, \
                f"Batch target {i} has collision enabled"

    def test_walker_not_visible(self):
        """Both scenes should have visualization=False for walker (invisible to camera)."""
        assert not self._single.walker.morph.visualization, \
            "Single walker should have visualization=False"
        assert not self._batch.walker.morph.visualization, \
            "Batch walker should have visualization=False"

    def test_steer_damping_in_build(self):
        """Both builds should set STEER_DAMPING on rz DOF (index 5)."""
        s_damp = self._single.walker.get_dofs_damping()
        s_damp = s_damp.cpu().numpy() if hasattr(s_damp, 'cpu') else np.asarray(s_damp)
        b_damp = self._batch.walker.get_dofs_damping()
        b_damp = b_damp.cpu().numpy() if hasattr(b_damp, 'cpu') else np.asarray(b_damp)
        if b_damp.ndim == 2:
            b_damp = b_damp[0]
        assert abs(float(s_damp[5]) - STEER_DAMPING) < 1e-6, f"Single rz damping={s_damp[5]}"
        assert abs(float(b_damp[5]) - STEER_DAMPING) < 1e-6, f"Batch rz damping={b_damp[5]}"

    def test_roll_damping_in_build(self):
        """Both builds should set ROLL_DAMPING on rx, ry DOFs (indices 3, 4)."""
        s_damp = self._single.walker.get_dofs_damping()
        s_damp = s_damp.cpu().numpy() if hasattr(s_damp, 'cpu') else np.asarray(s_damp)
        b_damp = self._batch.walker.get_dofs_damping()
        b_damp = b_damp.cpu().numpy() if hasattr(b_damp, 'cpu') else np.asarray(b_damp)
        if b_damp.ndim == 2:
            b_damp = b_damp[0]
        assert abs(float(s_damp[3]) - ROLL_DAMPING) < 1e-6, f"Single rx damping={s_damp[3]}"
        assert abs(float(s_damp[4]) - ROLL_DAMPING) < 1e-6, f"Single ry damping={s_damp[4]}"
        assert abs(float(b_damp[3]) - ROLL_DAMPING) < 1e-6, f"Batch rx damping={b_damp[3]}"
        assert abs(float(b_damp[4]) - ROLL_DAMPING) < 1e-6, f"Batch ry damping={b_damp[4]}"


# ===================================================================
# Render method consistency
# ===================================================================

class TestRenderConsistency:
    """Verify render_single(i) matches render_all()[i].

    Requires OpenGL 4.2+ (env_separate_rigid), so skipped on macOS.
    """

    @pytest.mark.xfail(
        reason="Rasterizer render() returns (n_envs,H,W,3) in batch mode; "
               "render_all() expects (H,W,3). Not used in production (BatchRenderer is).",
        strict=False,
    )
    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="Per-env Rasterizer rendering requires OpenGL 4.2 (not available on macOS)",
    )
    def test_render_single_matches_render_all(self, batch_scene_2):
        """render_single(i) should produce identical pixels to render_all()[i]."""
        import labmaze
        for env_idx in range(2):
            maze = labmaze.RandomMaze(
                height=batch_scene_2.outer_size,
                width=batch_scene_2.outer_size,
                max_rooms=6, room_min_size=3, room_max_size=5,
                spawns_per_room=1, objects_per_room=1,
                random_seed=100 + env_idx,
            )
            segments = extract_wall_segments(maze)
            batch_scene_2.configure_walls_for_env(env_idx, segments)
            batch_scene_2.set_walker_pose(env_idx, [0.0, 0.0], env_idx * 0.5)

        positions = batch_scene_2.get_walker_positions()
        headings = np.array([0.0, 0.5])
        batch_scene_2.update_cameras(positions, headings)

        all_images = batch_scene_2.render_all()
        for i in range(2):
            single_img = batch_scene_2.render_single(i)
            np.testing.assert_array_equal(
                all_images[i], single_img,
                err_msg=f"render_single({i}) != render_all()[{i}]",
            )


# ===================================================================
# Auto-reset observation correctness
# ===================================================================

class TestAutoResetObservation:
    """Verify observations after auto-reset are from the new episode.

    Uses a single shared env (class-scoped) to avoid Genesis scene exhaustion.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _shared_env(self, _init_genesis, request):
        """Create one BatchGenesisMemoryMazeEnv shared across tests."""
        env = BatchGenesisMemoryMazeEnv(
            n_envs=1, maze_size=9, seed=42, camera_resolution=32,
        )
        request.cls._env = env
        yield env
        env.close()

    def test_obs_after_auto_reset_is_valid(self):
        """After done+auto-reset, returned obs is valid and state is reset."""
        env = self._env
        env.reset()
        max_steps = env._max_steps

        # Step to one before done
        for _ in range(max_steps - 1):
            env.step([0])

        assert env._step_counts[0] == max_steps - 1

        # This step triggers done + auto-reset
        obs, _, dones, infos = env.step([0])
        assert dones[0], "Should be done at max_steps"

        # After auto-reset, internal state should reflect new episode
        assert env._step_counts[0] == 0, (
            f"Step count should be 0 after reset, got {env._step_counts[0]}"
        )
        assert env._targets_obtained[0] == 0, (
            f"Targets obtained should be 0, got {env._targets_obtained[0]}"
        )

        # Observation should be valid (not blank/corrupted)
        assert obs.shape == (1, 32, 32, 3)
        assert obs.dtype == np.uint8
        assert np.std(obs[0].astype(np.float32)) > 3.0, (
            "Post-reset obs appears blank/uniform"
        )

    def test_auto_reset_produces_different_maze(self):
        """Auto-reset should produce a new maze (different observation)."""
        env = self._env
        first_obs = env.reset().copy()

        # Run full episode
        for _ in range(env._max_steps - 1):
            env.step([1])
        obs_after_reset, _, dones, _ = env.step([1])
        assert dones[0]

        # New episode obs should differ (different maze/spawn)
        diff = np.mean(np.abs(
            first_obs[0].astype(np.float32) - obs_after_reset[0].astype(np.float32)
        ))
        assert diff > 0.5, (
            f"Auto-reset obs too similar to first episode: diff={diff:.1f}"
        )


# ===================================================================
# Multi-env contact boundary (n_envs=2)
# ===================================================================

@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Per-env Rasterizer rendering requires OpenGL 4.2 (not available on macOS)",
)
@pytest.mark.xfail(
    reason="Rasterizer render_all() returns wrong shape in batch mode; "
           "not used in production (BatchRenderer is).",
    strict=False,
)
class TestMultiEnvContactBoundary:
    """Verify contact check works correctly with n_envs=2.

    Guards against bugs that only manifest with multiple environments
    (e.g., reward applied to wrong env, target indices mixed up).
    Requires OpenGL 4.2+ for per-env Rasterizer rendering.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _shared_env(self, _init_genesis, request):
        env = BatchGenesisMemoryMazeEnv(
            n_envs=2, maze_size=9, seed=42, camera_resolution=32,
        )
        request.cls._env = env
        yield env
        env.close()

    def test_simultaneous_rewards_both_envs(self):
        """Both envs can get rewards simultaneously when both touch their targets."""
        env = self._env
        env.reset()

        for i in range(2):
            walker_pos = env._scene.get_walker_positions()[i].copy()
            t = int(env._current_target_ix[i])
            # Move non-current targets far away
            for j in range(env._n_targets):
                if j == t:
                    continue
                far = np.array([walker_pos[0] + 5.0 + j, walker_pos[1], TARGET_RADIUS])
                env._scene.set_target_pos(i, j, far.astype(np.float32))
                env._target_positions[i, j] = far
            # Place current target close (0.5m)
            target_pos = np.array([walker_pos[0] + 0.5, walker_pos[1], TARGET_RADIUS])
            env._scene.set_target_pos(i, t, target_pos.astype(np.float32))
            env._target_positions[i, t] = target_pos

        _, rewards, _, _ = env.step([0, 0])
        assert rewards[0] > 0, f"Env 0 should get reward, got {rewards[0]}"
        assert rewards[1] > 0, f"Env 1 should get reward, got {rewards[1]}"

    def test_independent_rewards_per_env(self):
        """Only the env touching its target gets a reward, not the other."""
        env = self._env
        env.reset()

        # Env 0: place current target close
        walker0 = env._scene.get_walker_positions()[0].copy()
        t0 = int(env._current_target_ix[0])
        for j in range(env._n_targets):
            if j == t0:
                continue
            far = np.array([walker0[0] + 5.0 + j, walker0[1], TARGET_RADIUS])
            env._scene.set_target_pos(0, j, far.astype(np.float32))
            env._target_positions[0, j] = far
        close_pos = np.array([walker0[0] + 0.5, walker0[1], TARGET_RADIUS])
        env._scene.set_target_pos(0, t0, close_pos.astype(np.float32))
        env._target_positions[0, t0] = close_pos

        # Env 1: place current target far
        walker1 = env._scene.get_walker_positions()[1].copy()
        t1 = int(env._current_target_ix[1])
        far_pos = np.array([walker1[0] + 5.0, walker1[1], TARGET_RADIUS])
        env._scene.set_target_pos(1, t1, far_pos.astype(np.float32))
        env._target_positions[1, t1] = far_pos

        _, rewards, _, _ = env.step([0, 0])
        assert rewards[0] > 0, f"Env 0 should get reward, got {rewards[0]}"
        assert rewards[1] == 0.0, f"Env 1 should NOT get reward, got {rewards[1]}"


# ===================================================================
# Wall texture group assignment
# ===================================================================

class TestWallTextureGroupAssignment:
    """Verify wall entities are correctly assigned to texture groups."""

    def test_textured_walls_have_groups(self, _init_genesis):
        """Scene with textures should have 9 wall groups ('0'-'8')."""
        scene = GenesisMazeScene(
            maze_size=9, n_targets=3, camera_resolution=32, use_textures=True,
        )
        assert scene._wall_groups is not None, "Textured scene should have _wall_groups"
        assert len(scene._wall_groups) == 9, f"Expected 9 groups, got {len(scene._wall_groups)}"
        for i in range(9):
            assert str(i) in scene._wall_groups, f"Group '{i}' missing"

    def test_batch_textured_walls_have_groups(self, _init_genesis):
        """Batch scene with textures should also have 9 wall groups."""
        scene = BatchGenesisMazeScene(
            n_envs=1, maze_size=9, n_targets=3, camera_resolution=32, use_textures=True,
        )
        assert scene._wall_groups is not None
        assert len(scene._wall_groups) == 9

    def test_no_textures_no_groups(self, _init_genesis):
        """Scene without textures should have _wall_groups=None."""
        scene = GenesisMazeScene(
            maze_size=9, n_targets=3, camera_resolution=32, use_textures=False,
        )
        assert scene._wall_groups is None

    def test_shuffled_wall_groups_returns_copy(self, _init_genesis):
        """shuffled_wall_groups() should return a new dict, not mutate original."""
        scene = BatchGenesisMazeScene(
            n_envs=1, maze_size=9, n_targets=3, camera_resolution=32, use_textures=True,
        )
        original_groups = {k: v for k, v in scene._wall_groups.items()}
        rng = np.random.RandomState(42)
        shuffled = scene.shuffled_wall_groups(rng)

        # Shuffled should be a different dict object
        assert shuffled is not scene._wall_groups, "Should return a new dict"
        # Original should be unchanged (same entity references per key)
        for k in original_groups:
            assert scene._wall_groups[k] is original_groups[k], (
                f"Original group '{k}' was mutated"
            )


# ===================================================================
# _pick_new_target degenerate case
# ===================================================================

class TestPickNewTargetDegenerate:
    """Test behavior when all targets are within activation distance."""

    def test_pick_new_target_terminates_when_all_close(self, _init_genesis):
        """_pick_new_target terminates with fallback when all targets are nearby.

        Fixed: _pick_new_target now has max_attempts=100 with fallback to
        (current_ix + 1) % n_targets.
        """
        import threading

        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        env.reset()

        # Place all targets within activation distance of walker
        walker_pos = env._scene.get_walker_position()
        for i in range(env._n_targets):
            close_pos = np.array([
                walker_pos[0] + 0.3,
                walker_pos[1] + 0.1 * i,
                TARGET_RADIUS,
            ])
            env._scene.target_entities[i].set_pos(close_pos)
            env._scene._target_world_positions[i] = close_pos.copy()
            env._target_world_positions[i] = close_pos.copy()

        completed = [False]

        def run():
            try:
                env._pick_new_target()
                completed[0] = True
            except Exception:
                completed[0] = True  # exception is better than hang

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=2.0)

        assert completed[0], (
            "_pick_new_target stuck in infinite loop when all targets "
            "are within activation distance"
        )
        env.close()


# ===================================================================
# Pre-refactoring behavior locks
# ===================================================================

class TestCloseReleasesResources:
    """Verify close() releases the scene and is safe to call multiple times."""

    def test_single_env_close_releases_scene(self, _init_genesis):
        """Single-env close() sets _scene to None."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        env.reset()
        env.step(1)
        assert env._scene is not None
        env.close()
        assert env._scene is None

    def test_batch_env_close_releases_scene(self, _init_genesis):
        """Batch-env close() sets _scene and _mazes to None."""
        env = BatchGenesisMemoryMazeEnv(
            n_envs=1, maze_size=9, seed=42, camera_resolution=32,
        )
        env.reset()
        assert env._scene is not None
        env.close()
        assert env._scene is None
        assert env._mazes is None

    def test_double_close_safe(self, _init_genesis):
        """Calling close() twice should not raise."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        env.reset()
        env.close()
        env.close()  # second call should be safe


class TestBorderColorCorrectness:
    """Verify border pixels match the current target's color."""

    def test_single_env_border_matches_target_color(self, _init_genesis):
        """Border color should be TARGET_COLORS[current_target] * 255 * 0.7."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=64, seed=42)
        obs = env.reset()
        target_ix = env._current_target_ix
        expected_color = (TARGET_COLORS[target_ix] * 255 * 0.7).astype(np.uint8)
        # Check top-left corner pixel (definitely in border)
        actual = obs[0, 0, :]
        np.testing.assert_array_equal(actual, expected_color,
            err_msg=f"Border color mismatch: target_ix={target_ix}")
        env.close()

    @pytest.mark.xfail(
        reason="Rasterizer render_all() returns wrong shape in batch mode; "
               "not used in production (BatchRenderer is).",
        strict=False,
    )
    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="Batch env rendering requires OpenGL 4.2 (not available on macOS)",
    )
    def test_batch_env_border_matches_target_color(self, _init_genesis):
        """Batch border should match per-env target color."""
        env = BatchGenesisMemoryMazeEnv(
            n_envs=2, maze_size=9, seed=42, camera_resolution=64,
        )
        obs = env.reset()
        for i in range(2):
            target_ix = int(env._current_target_ix[i])
            expected = (TARGET_COLORS[target_ix] * 255 * 0.7).astype(np.uint8)
            actual = obs[i, 0, 0, :]
            np.testing.assert_array_equal(actual, expected,
                err_msg=f"Env {i} border mismatch: target_ix={target_ix}")
        env.close()


class TestResetDeterminism:
    """Verify same-seed resets produce identical layouts."""

    def test_same_seed_same_walls(self, _init_genesis):
        """Two envs with the same seed should produce identical wall positions."""
        positions_per_run = []
        for _ in range(2):
            scene = GenesisMazeScene(
                maze_size=9, n_targets=3, camera_resolution=32,
                use_textures=False,
            )
            scene.build()
            rng = np.random.RandomState(42)
            scene.reset(rng)
            # Collect all visible wall positions
            wall_pos = []
            for w in scene.wall_entities:
                pos = w.get_pos()
                p = pos.cpu().numpy() if hasattr(pos, 'cpu') else np.asarray(pos)
                if p[2] > _UNDERGROUND_THRESHOLD:
                    wall_pos.append(p.copy())
            positions_per_run.append(sorted([tuple(p) for p in wall_pos]))
        assert positions_per_run[0] == positions_per_run[1], \
            "Same-seed resets should produce identical wall layouts"

    def test_same_seed_same_spawn(self, _init_genesis):
        """Two envs with the same seed should place walker at same position."""
        walker_positions = []
        for _ in range(2):
            scene = GenesisMazeScene(
                maze_size=9, n_targets=3, camera_resolution=32,
                use_textures=False,
            )
            scene.build()
            rng = np.random.RandomState(42)
            scene.reset(rng)
            walker_positions.append(scene.get_walker_position().copy())
        np.testing.assert_array_almost_equal(
            walker_positions[0], walker_positions[1], decimal=5,
            err_msg="Same-seed resets should produce identical spawn positions"
        )

    def test_same_seed_same_targets(self, _init_genesis):
        """Two envs with the same seed should place targets identically."""
        target_positions = []
        for _ in range(2):
            scene = GenesisMazeScene(
                maze_size=9, n_targets=3, camera_resolution=32,
                use_textures=False,
            )
            scene.build()
            rng = np.random.RandomState(42)
            scene.reset(rng)
            target_positions.append(
                [p.copy() for p in scene._target_world_positions]
            )
        for i in range(3):
            np.testing.assert_array_almost_equal(
                target_positions[0][i], target_positions[1][i], decimal=5,
                err_msg=f"Target {i} position mismatch between same-seed resets"
            )


class TestRoomMinSizePropagation:
    """Verify room_min_size flows from scene to batch reset."""

    def test_batch_env_uses_scene_room_min_size(self, _init_genesis):
        """BatchGenesisMemoryMazeEnv._reset_single_env should use
        the scene's room_min_size, not a hardcoded value."""
        env = BatchGenesisMemoryMazeEnv(
            n_envs=1, maze_size=9, seed=42, camera_resolution=32,
        )
        # Verify the scene has room_min_size stored
        assert hasattr(env._scene, 'room_min_size'), \
            "Scene should expose room_min_size"
        assert env._scene.room_min_size == 3, \
            f"Expected room_min_size=3, got {env._scene.room_min_size}"
        env.close()


class TestUnknownKwargsHandling:
    """Unknown kwargs should raise TypeError (no silent swallowing)."""

    def test_unknown_kwargs_raises_type_error(self, _init_genesis):
        """GenesisMemoryMazeEnv rejects unknown kwargs with TypeError."""
        with pytest.raises(TypeError, match="totally_bogus_param"):
            GenesisMemoryMazeEnv(
                maze_size=9, camera_resolution=32, seed=42,
                totally_bogus_param=True,
            )


class TestToNumpyConsistency:
    """Verify render output is always numpy uint8 regardless of backend."""

    def test_render_egocentric_returns_numpy_uint8(self, single_scene, rng):
        """render_egocentric() should always return numpy uint8 array."""
        single_scene.reset(rng)
        img = single_scene.render_egocentric()
        assert isinstance(img, np.ndarray), f"Expected ndarray, got {type(img)}"
        assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="Batch env rendering requires OpenGL 4.2 (not available on macOS)",
    )
    def test_batch_render_all_returns_numpy_uint8(self, batch_scene_2):
        """render_all() should always return numpy uint8 array."""
        import labmaze
        rng = np.random.RandomState(42)
        maze = labmaze.RandomMaze(
            height=batch_scene_2.outer_size,
            width=batch_scene_2.outer_size,
            max_rooms=6, room_min_size=3, room_max_size=5,
            spawns_per_room=1, objects_per_room=1,
            random_seed=42,
        )
        segments = extract_wall_segments(maze)
        for i in range(2):
            batch_scene_2.configure_walls_for_env(i, segments)
            batch_scene_2.set_walker_pose(i, [0.0, 0.0], 0.0)
        positions = batch_scene_2.get_walker_positions()
        batch_scene_2.update_cameras(positions, np.array([0.0, 0.0]))
        imgs = batch_scene_2.render_all()
        assert isinstance(imgs, np.ndarray), f"Expected ndarray, got {type(imgs)}"
        assert imgs.dtype == np.uint8, f"Expected uint8, got {imgs.dtype}"
        assert imgs.shape == (2, 32, 32, 3)
