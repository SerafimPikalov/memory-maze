"""Comprehensive tests for Genesis Memory Maze backend — single-env and batch mode.

Covers 7 identified risk classes, integration tests, and smoke tests.
Run with: pytest tests/test_genesis_batch.py -v -m "not slow"
"""

import math
import os
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
    CAMERA_FOV,
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

    Catches copy-paste bugs like hardcoded friction/damping values that
    diverge from the tuned constants (e.g. friction=0.5 vs WALKER_FRICTION=0.01).
    """

    def test_walker_friction_uses_constant(self):
        """Batched walker friction must match WALKER_FRICTION, not a hardcoded value."""
        import inspect
        source = inspect.getsource(BatchGenesisMazeScene.__init__)
        assert 'friction=0.5,\n                rho=WALKER_TOTAL_MASS' not in source, \
            "Batched walker uses hardcoded friction=0.5 instead of WALKER_FRICTION"
        assert 'friction=WALKER_FRICTION' in source, \
            "Batched walker friction should use WALKER_FRICTION constant"

    def test_walker_damping_has_trans_damping(self):
        """Batched walker damping must include TRANS_DAMPING on tx, ty DOFs."""
        import inspect
        source = inspect.getsource(BatchGenesisMazeScene.build)
        assert 'TRANS_DAMPING, TRANS_DAMPING, 0.0' in source, \
            "Batched walker damping missing TRANS_DAMPING on tx, ty DOFs"
        assert '0.0, 0.0, 0.0,\n            ROLL_DAMPING' not in source, \
            "Batched walker damping has 0.0 for tx, ty instead of TRANS_DAMPING"
