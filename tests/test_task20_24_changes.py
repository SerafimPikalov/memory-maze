"""Targeted tests for Tasks 20-23 changes.

Task 20: TARGET_ACTIVATION_GAP = 0.8m (was 1.2m)
Task 21: grad_norm captured in learn()
Task 22: episodes_per_batch metric added
Task 23: create_env() passes seed to gym.make()

Run:
    pytest memory-maze/tests/test_task20_24_changes.py -v
"""

import math
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import pytest

try:
    import genesis as gs
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False

pytestmark = pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")

from memory_maze.genesis_backend import (
    TARGET_ACTIVATION_GAP,
    TARGET_RADIUS,
    WALKER_RADIUS,
    GenesisMazeScene,
    GenesisMemoryMazeEnv,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def _init_genesis():
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")


@pytest.fixture
def scene(_init_genesis):
    """9x9 scene at 32px for fast tests."""
    s = GenesisMazeScene(maze_size=9, n_targets=3, camera_resolution=32)
    s.build()
    return s


# ===================================================================
# Task 20: TARGET_ACTIVATION_GAP value
# ===================================================================

class TestActivationGap:

    def test_gap_value_is_0_8(self):
        """TARGET_ACTIVATION_GAP must be WALKER_RADIUS + TARGET_RADIUS = 0.8."""
        assert TARGET_ACTIVATION_GAP == WALKER_RADIUS + TARGET_RADIUS
        assert TARGET_ACTIVATION_GAP == pytest.approx(0.8)

    def test_contact_at_0_7m(self, scene):
        """Walker at 0.7m from target center → should activate (0.7 < 0.8)."""
        rng = np.random.RandomState(42)
        scene.reset(rng)

        # Place walker at origin
        scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        scene.walker.set_dofs_velocity(np.zeros(6))

        # Place first target at 0.7m away on x-axis (above ground)
        target_pos = np.array([0.7, 0.0, TARGET_RADIUS])
        scene.target_entities[0].set_pos(target_pos)
        scene._target_world_positions[0] = target_pos.copy()

        walker_pos = np.array([0.0, 0.0, WALKER_RADIUS])
        contacts = scene.check_target_contacts(walker_pos)
        assert contacts[0], "Should activate at 0.7m (< 0.8m threshold)"

    def test_no_contact_at_0_9m(self, scene):
        """Walker at 0.9m from target center → should NOT activate (0.9 > 0.8)."""
        rng = np.random.RandomState(42)
        scene.reset(rng)

        scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        scene.walker.set_dofs_velocity(np.zeros(6))

        target_pos = np.array([0.9, 0.0, TARGET_RADIUS])
        scene.target_entities[0].set_pos(target_pos)
        scene._target_world_positions[0] = target_pos.copy()

        walker_pos = np.array([0.0, 0.0, WALKER_RADIUS])
        contacts = scene.check_target_contacts(walker_pos)
        assert not contacts[0], "Should NOT activate at 0.9m (> 0.8m threshold)"

    def test_contact_boundary_just_inside(self, scene):
        """Walker at 0.79m → should activate."""
        rng = np.random.RandomState(42)
        scene.reset(rng)

        scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        target_pos = np.array([0.79, 0.0, TARGET_RADIUS])
        scene.target_entities[0].set_pos(target_pos)
        scene._target_world_positions[0] = target_pos.copy()

        contacts = scene.check_target_contacts(np.array([0.0, 0.0, WALKER_RADIUS]))
        assert contacts[0], "Should activate at 0.79m (< 0.8m)"

    def test_contact_boundary_just_outside(self, scene):
        """Walker at 0.81m → should NOT activate."""
        rng = np.random.RandomState(42)
        scene.reset(rng)

        scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        target_pos = np.array([0.81, 0.0, TARGET_RADIUS])
        scene.target_entities[0].set_pos(target_pos)
        scene._target_world_positions[0] = target_pos.copy()

        contacts = scene.check_target_contacts(np.array([0.0, 0.0, WALKER_RADIUS]))
        assert not contacts[0], "Should NOT activate at 0.81m (> 0.8m)"

    def test_old_gap_1_2m_would_have_activated(self, scene):
        """At 1.1m, old gap (1.2) would activate but new gap (0.8) should not."""
        rng = np.random.RandomState(42)
        scene.reset(rng)

        scene.walker.set_pos(np.array([0.0, 0.0, WALKER_RADIUS]))
        target_pos = np.array([1.1, 0.0, TARGET_RADIUS])
        scene.target_entities[0].set_pos(target_pos)
        scene._target_world_positions[0] = target_pos.copy()

        contacts = scene.check_target_contacts(np.array([0.0, 0.0, WALKER_RADIUS]))
        assert not contacts[0], (
            "At 1.1m should NOT activate with 0.8m gap "
            "(old 1.2m gap would have incorrectly activated)"
        )

    def test_contact_uses_2d_distance(self, scene):
        """Contact check should use XY distance only, ignoring Z."""
        rng = np.random.RandomState(42)
        scene.reset(rng)

        # Walker and target at same XY but different Z → XY distance = 0.5
        walker_pos = np.array([0.0, 0.0, 0.2])
        target_pos = np.array([0.5, 0.0, 5.0])  # Z far away
        scene.target_entities[0].set_pos(target_pos)
        scene._target_world_positions[0] = target_pos.copy()

        contacts = scene.check_target_contacts(walker_pos)
        assert contacts[0], "Contact should use 2D (XY) distance, not 3D"

    def test_underground_target_no_contact(self, scene):
        """Target underground (z < -5) should never activate."""
        rng = np.random.RandomState(42)
        scene.reset(rng)

        walker_pos = np.array([0.0, 0.0, WALKER_RADIUS])
        target_pos = np.array([0.0, 0.0, -10.0])  # Underground
        scene.target_entities[0].set_pos(target_pos)
        scene._target_world_positions[0] = target_pos.copy()

        contacts = scene.check_target_contacts(walker_pos)
        assert not contacts[0], "Underground target should never activate"


# ===================================================================
# Task 20: Gym env reward with new activation distance
# ===================================================================

class TestGymEnvActivation:

    def test_full_episode_rewards_are_binary(self, _init_genesis):
        """All rewards in a full episode should be 0.0 or 1.0."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        obs = env.reset()
        rng = np.random.RandomState(42)
        rewards = []
        for _ in range(200):
            obs, reward, done, info = env.step(rng.randint(6))
            rewards.append(reward)
            if done:
                break
        env.close()
        for r in rewards:
            assert r in (0.0, 1.0), f"Unexpected reward: {r}"

    def test_pick_new_target_respects_gap(self, _init_genesis):
        """_pick_new_target should not pick a target within 0.8m of walker."""
        env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
        env.reset()

        # Call _pick_new_target many times; each chosen target should be >= 0.8m
        walker_pos = env._scene.get_walker_position()
        for _ in range(50):
            env._pick_new_target()
            ix = env._current_target_ix
            tpos = env._target_world_positions[ix]
            dist = np.linalg.norm(walker_pos[:2] - tpos[:2])
            assert dist >= TARGET_ACTIVATION_GAP, (
                f"Picked target {ix} at dist {dist:.3f} < {TARGET_ACTIVATION_GAP}"
            )
        env.close()


# ===================================================================
# Task 23: Seeded reproducibility
# ===================================================================

class TestSeededReproducibility:

    def test_same_seed_same_obs(self, _init_genesis):
        """Two envs with same seed should produce identical initial obs."""
        obs_list = []
        for _ in range(2):
            env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
            obs = env.reset()
            obs_list.append(obs.copy())
            env.close()

        np.testing.assert_array_equal(
            obs_list[0], obs_list[1],
            err_msg="Same seed should produce identical observations"
        )

    def test_different_seed_different_obs(self, _init_genesis):
        """Two envs with different seeds should produce different obs."""
        obs_list = []
        for seed in [42, 99]:
            env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=seed)
            obs = env.reset()
            obs_list.append(obs.copy())
            env.close()

        diff = np.mean(np.abs(obs_list[0].astype(float) - obs_list[1].astype(float)))
        assert diff > 1.0, f"Different seeds should produce different obs, diff={diff}"

    def test_same_seed_same_trajectory(self, _init_genesis):
        """Same seed + same actions → identical reward sequence."""
        actions = [1, 1, 1, 2, 1, 1, 3, 1, 1, 1]  # fixed action sequence
        reward_sequences = []
        for _ in range(2):
            env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32, seed=42)
            env.reset()
            rewards = []
            for a in actions:
                _, r, _, _ = env.step(a)
                rewards.append(r)
            reward_sequences.append(rewards)
            env.close()

        assert reward_sequences[0] == reward_sequences[1], (
            "Same seed + same actions should produce identical rewards"
        )

    def test_seed_via_gym_make(self, _init_genesis):
        """gym.make with seed kwarg should produce reproducible envs."""
        import gym

        obs_list = []
        for _ in range(2):
            env = gym.make(
                "memory_maze:MemoryMaze-9x9-Genesis-v0",
                disable_env_checker=True,
                seed=42,
            )
            obs = env.reset()
            obs_list.append(obs.copy())
            env.close()

        np.testing.assert_array_equal(
            obs_list[0], obs_list[1],
            err_msg="gym.make with same seed should produce identical obs"
        )

    def test_seed_method_reproducibility(self, _init_genesis):
        """env.seed() should also produce reproducible resets."""
        obs_list = []
        for _ in range(2):
            env = GenesisMemoryMazeEnv(maze_size=9, camera_resolution=32)
            env.seed(42)
            obs = env.reset()
            obs_list.append(obs.copy())
            env.close()

        np.testing.assert_array_equal(
            obs_list[0], obs_list[1],
            err_msg="env.seed(42) should produce identical observations"
        )


# ===================================================================
# Task 21-22: Training metrics (source code checks)
# ===================================================================

class TestTrainingMetricsCode:
    """Verify train_impala.py source has the required metric changes.

    These can't be tested by running learn() without a full IMPALA setup,
    so we check the source code directly.
    """

    @pytest.fixture(scope="class")
    def impala_source(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "train_impala.py"
        )
        with open(src_path) as f:
            return f.read()

    def test_grad_norm_captured(self, impala_source):
        """clip_grad_norm_ return value must be assigned to grad_norm."""
        assert "grad_norm = nn.utils.clip_grad_norm_" in impala_source

    def test_grad_norm_in_stats(self, impala_source):
        """grad_norm must appear in the stats dict."""
        assert '"grad_norm"' in impala_source

    def test_episodes_per_batch_in_stats(self, impala_source):
        """episodes_per_batch must appear in the stats dict."""
        assert '"episodes_per_batch"' in impala_source

    def test_grad_norm_in_stat_keys(self, impala_source):
        """grad_norm must be in stat_keys for W&B logging."""
        # Find the stat_keys list and check it contains grad_norm
        assert '"grad_norm",' in impala_source

    def test_episodes_per_batch_in_stat_keys(self, impala_source):
        """episodes_per_batch must be in stat_keys for W&B logging."""
        assert '"episodes_per_batch",' in impala_source

    def test_create_env_accepts_actor_index(self, impala_source):
        """create_env must accept actor_index parameter."""
        assert "def create_env(flags, actor_index=0)" in impala_source

    def test_create_env_passes_seed(self, impala_source):
        """create_env must pass seed + actor_index to gym.make."""
        assert 'kwargs["seed"] = flags.seed + actor_index' in impala_source

    def test_act_passes_actor_index(self, impala_source):
        """act() must pass actor_index to create_env."""
        assert "create_env(flags, actor_index=actor_index)" in impala_source
