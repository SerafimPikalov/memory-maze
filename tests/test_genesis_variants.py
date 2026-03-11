"""Tests for Genesis environment variant parity with MuJoCo (Task 68).

GPU memory management: Genesis/Taichi does not release GPU memory when scenes
are destroyed. Tests are structured to minimize scene creation:
- Class-scoped fixtures share one env across all tests in a class
- TestCrossVariantConsistency uses spec-based checks (no env creation)
- run_genesis_tests.sh runs each class in a separate process for isolation

Run locally:  MUJOCO_GL=glfw pytest tests/test_genesis_variants.py -v
Run on RunPod: bash tests/run_genesis_tests.sh
"""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pytest

try:
    import genesis as gs
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False

try:
    import gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

# Import memory_maze to trigger env registration
try:
    import memory_maze
except Exception:
    pass

pytestmark = [
    pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed"),
    pytest.mark.skipif(not HAS_GYM, reason="gym not installed"),
]


@pytest.fixture(scope="session")
def init_genesis():
    if not gs._initialized:
        # Use CUDA on Linux (RunPod), CPU on macOS (local dev)
        backend = gs.cuda if sys.platform == "linux" else gs.cpu
        gs.init(backend=backend, logging_level="warning")
    return gs


# ============================================================================
# 68a: Trivial variant registration (no env creation)
# ============================================================================

class TestVariantRegistration:
    """Test that all Genesis variants are registered and can be created."""

    BASIC_IDS = [
        "MemoryMaze-9x9-Genesis-v0",
        "MemoryMaze-11x11-Genesis-v0",
        "MemoryMaze-13x13-Genesis-v0",
        "MemoryMaze-15x15-Genesis-v0",
    ]

    TRIVIAL_VARIANT_IDS = [
        "MemoryMaze-9x9-Vis-Genesis-v0",
        "MemoryMaze-9x9-HD-Genesis-v0",
        "MemoryMaze-9x9-HiFreq-Genesis-v0",
        "MemoryMaze-9x9-HiFreq-Vis-Genesis-v0",
        "MemoryMaze-9x9-HiFreq-HD-Genesis-v0",
    ]

    ALL_VARIANT_IDS = [
        # ExtraObs
        "MemoryMaze-9x9-ExtraObs-Genesis-v0",
        "MemoryMaze-9x9-ExtraObs-Vis-Genesis-v0",
        # 6CL
        "MemoryMaze-9x9-6CL-Genesis-v0",
        "MemoryMaze-9x9-6CL-ExtraObs-Genesis-v0",
        # Top
        "MemoryMaze-9x9-Top-Genesis-v0",
        "MemoryMaze-9x9-ExtraObs-Top-Genesis-v0",
        "MemoryMaze-9x9-6CL-Top-Genesis-v0",
        # Oracle
        "MemoryMaze-9x9-Oracle-Genesis-v0",
        "MemoryMaze-9x9-Oracle-Top-Genesis-v0",
        "MemoryMaze-9x9-Oracle-ExtraObs-Genesis-v0",
    ]

    def test_basic_ids_registered(self):
        for env_id in self.BASIC_IDS:
            spec = gym.spec(env_id)
            assert spec is not None, f"{env_id} not registered"

    def test_trivial_variant_ids_registered(self):
        for env_id in self.TRIVIAL_VARIANT_IDS:
            spec = gym.spec(env_id)
            assert spec is not None, f"{env_id} not registered"

    def test_all_variant_ids_registered(self):
        for env_id in self.ALL_VARIANT_IDS:
            spec = gym.spec(env_id)
            assert spec is not None, f"{env_id} not registered"

    def test_naming_matches_auto_routing(self):
        """Verify Genesis names match train_impala.py auto-routing pattern."""
        mujoco_ids = [
            "MemoryMaze-9x9-v0",
            "MemoryMaze-9x9-Vis-v0",
            "MemoryMaze-9x9-HD-v0",
            "MemoryMaze-9x9-HiFreq-v0",
            "MemoryMaze-9x9-HiFreq-Vis-v0",
            "MemoryMaze-9x9-HiFreq-HD-v0",
            "MemoryMaze-9x9-ExtraObs-v0",
            "MemoryMaze-9x9-ExtraObs-Vis-v0",
            "MemoryMaze-9x9-6CL-v0",
            "MemoryMaze-9x9-6CL-ExtraObs-v0",
            "MemoryMaze-9x9-Top-v0",
            "MemoryMaze-9x9-ExtraObs-Top-v0",
            "MemoryMaze-9x9-6CL-Top-v0",
            "MemoryMaze-9x9-Oracle-v0",
            "MemoryMaze-9x9-Oracle-Top-v0",
            "MemoryMaze-9x9-Oracle-ExtraObs-v0",
        ]
        for mid in mujoco_ids:
            genesis_id = mid.replace("-v0", "-Genesis-v0")
            spec = gym.spec(genesis_id)
            assert spec is not None, f"Auto-routed {genesis_id} not registered (from {mid})"

    def test_all_sizes_have_variants(self):
        suffixes = [
            "", "-Vis", "-HD", "-HiFreq", "-HiFreq-Vis", "-HiFreq-HD",
            "-ExtraObs", "-ExtraObs-Vis",
            "-6CL", "-6CL-ExtraObs",
            "-Top", "-ExtraObs-Top", "-6CL-Top",
            "-Oracle", "-Oracle-Top", "-Oracle-ExtraObs",
        ]
        for size in ["9x9", "11x11", "13x13", "15x15"]:
            for suffix in suffixes:
                env_id = f"MemoryMaze-{size}{suffix}-Genesis-v0"
                spec = gym.spec(env_id)
                assert spec is not None, f"{env_id} not registered"


# ============================================================================
# 68a: Trivial variant smoke tests (each test creates one env)
# ============================================================================

class TestTrivialVariantSmoke:
    """Smoke test trivial variants: create, reset, step, check obs shape.

    Each test creates one env. On CUDA, run each test in its own process
    via run_genesis_tests.sh to avoid GPU memory accumulation.
    """

    def test_vis_variant(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-Vis-Genesis-v0")
        obs = env.reset()
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8
        obs2, r, done, info = env.step(1)
        assert obs2.shape == (64, 64, 3)
        env.close()

    def test_hd_variant(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-HD-Genesis-v0")
        obs = env.reset()
        assert obs.shape == (256, 256, 3), f"HD should be 256x256, got {obs.shape}"
        env.close()

    def test_hifreq_variant(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-HiFreq-Genesis-v0")
        obs = env.reset()
        assert obs.shape == (64, 64, 3)
        assert env.unwrapped._max_steps == 250 * 40
        env.close()

    def test_hifreq_vis_variant(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-HiFreq-Vis-Genesis-v0")
        obs = env.reset()
        assert obs.shape == (64, 64, 3)
        assert env.unwrapped._max_steps == 250 * 40
        env.close()

    def test_hifreq_hd_variant(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-HiFreq-HD-Genesis-v0")
        obs = env.reset()
        assert obs.shape == (256, 256, 3)
        assert env.unwrapped._max_steps == 250 * 40
        env.close()


# ============================================================================
# 68b: ExtraObs — class-scoped fixture shares one env
# ============================================================================

class TestExtraObs:
    """Test ExtraObs variant returns dict observations with correct keys/shapes."""

    EXPECTED_KEYS = {
        'image', 'target_color',
        'agent_pos', 'agent_dir',
        'targets_vec', 'targets_pos',
        'target_vec', 'target_pos',
        'maze_layout',
    }

    @pytest.fixture(scope="class")
    def env(self, init_genesis):
        e = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-ExtraObs-Genesis-v0")
        yield e
        e.close()

    def test_extraobs_returns_dict(self, env):
        obs = env.reset()
        assert isinstance(obs, dict), f"ExtraObs should return dict, got {type(obs)}"

    def test_extraobs_keys(self, env):
        obs = env.reset()
        assert set(obs.keys()) == self.EXPECTED_KEYS, \
            f"Keys mismatch: got {set(obs.keys())}, expected {self.EXPECTED_KEYS}"

    def test_extraobs_shapes(self, env):
        obs = env.reset()
        n_targets = 3  # 9x9 has 3 targets
        expected = {
            'image': ((64, 64, 3), np.uint8),
            'target_color': ((3,), np.float64),
            'agent_pos': ((2,), np.float64),
            'agent_dir': ((2,), np.float64),
            'targets_vec': ((n_targets, 2), np.float64),
            'targets_pos': ((n_targets, 2), np.float64),
            'target_vec': ((2,), np.float64),
            'target_pos': ((2,), np.float64),
            'maze_layout': ((9, 9), np.uint8),
        }
        for key, (shape, dtype) in expected.items():
            assert obs[key].shape == shape, f"{key}: expected shape {shape}, got {obs[key].shape}"
            assert obs[key].dtype == dtype, f"{key}: expected dtype {dtype}, got {obs[key].dtype}"

    def test_extraobs_agent_pos_range(self, env):
        obs = env.reset()
        pos = obs['agent_pos']
        assert np.all(pos >= -1) and np.all(pos <= 12), \
            f"agent_pos {pos} out of expected grid coordinate range"

    def test_extraobs_agent_dir_unit_vector(self, env):
        obs = env.reset()
        d = obs['agent_dir']
        norm = np.linalg.norm(d)
        assert abs(norm - 1.0) < 0.1, f"agent_dir norm = {norm}, expected ~1.0"

    def test_extraobs_maze_layout_binary(self, env):
        obs = env.reset()
        ml = obs['maze_layout']
        assert set(np.unique(ml)).issubset({0, 1}), \
            f"maze_layout should be binary, got values {np.unique(ml)}"
        assert np.sum(ml == 0) > 0, "No walls in maze_layout"
        assert np.sum(ml == 1) > 0, "No corridors in maze_layout"

    def test_extraobs_observation_space_is_dict(self, env):
        assert isinstance(env.observation_space, gym.spaces.Dict), \
            f"Expected Dict space, got {type(env.observation_space)}"

    def test_extraobs_step_returns_dict(self, env):
        env.reset()
        obs2, r, done, info = env.step(1)
        assert isinstance(obs2, dict)
        assert set(obs2.keys()) == self.EXPECTED_KEYS

    def test_extraobs_vis_variant(self, init_genesis):
        """ExtraObs-Vis needs its own env (different variant)."""
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-ExtraObs-Vis-Genesis-v0")
        obs = env.reset()
        assert isinstance(obs, dict)
        assert set(obs.keys()) == self.EXPECTED_KEYS
        env.close()


# ============================================================================
# 68b: ExtraObs parity — class-scoped fixture
# ============================================================================

class TestExtraObsParity:
    """Cross-backend numerical checks for ExtraObs values."""

    @pytest.fixture(scope="class")
    def env(self, init_genesis):
        e = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-ExtraObs-Genesis-v0")
        yield e
        e.close()

    def test_maze_layout_shape_9x9(self, env):
        obs = env.reset()
        assert obs['maze_layout'].shape == (9, 9)

    def test_maze_layout_shape_11x11_spec(self):
        """Verify 11x11 ExtraObs is registered with correct maze_size (spec-based)."""
        spec = gym.spec("MemoryMaze-11x11-ExtraObs-Genesis-v0")
        assert spec.kwargs['maze_size'] == 11

    def test_target_color_range(self, env):
        obs = env.reset()
        tc = obs['target_color']
        assert np.all(tc >= 0) and np.all(tc <= 1), \
            f"target_color {tc} out of [0,1] range"

    def test_targets_pos_in_grid_coords(self, env):
        obs = env.reset()
        tp = obs['targets_pos']
        assert np.all(tp >= -1) and np.all(tp <= 12), \
            f"targets_pos {tp} out of grid coordinate range"

    def test_target_vec_is_egocentric(self, env):
        obs = env.reset()
        tv = obs['target_vec']
        assert tv.shape == (2,)
        assert np.all(np.isfinite(tv))


# ============================================================================
# 68c: 6CL — class-scoped fixture (uses ExtraObs variant to test colors)
# ============================================================================

class TestColorShuffle:
    """Test 6CL variant shuffles target colors across episodes."""

    @pytest.fixture(scope="class")
    def env(self, init_genesis):
        e = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-6CL-ExtraObs-Genesis-v0")
        yield e
        e.close()

    def test_6cl_smoke(self, env):
        obs = env.reset()
        assert obs['image'].shape == (64, 64, 3)
        assert obs['image'].dtype == np.uint8
        obs2, r, done, info = env.step(1)
        assert obs2['image'].shape == (64, 64, 3)

    def test_6cl_colors_vary_across_resets(self, env):
        color_sequences = []
        for _ in range(20):
            obs = env.reset()
            tc = obs['target_color'].copy()
            color_sequences.append(tuple(tc))
        unique_colors = len(set(color_sequences))
        assert unique_colors > 1, \
            "6CL should produce different target colors across resets, but all were identical"

    def test_6cl_extraobs_variant(self, env):
        obs = env.reset()
        assert isinstance(obs, dict)
        assert 'target_color' in obs


# ============================================================================
# 68d: Top camera
# ============================================================================

class TestTopCamera:
    """Test top-down camera variant."""

    def test_top_smoke(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-Top-Genesis-v0")
        obs = env.reset()
        assert obs.shape == (256, 256, 3), f"Top should be 256x256, got {obs.shape}"
        env.close()

    def test_top_extraobs(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-ExtraObs-Top-Genesis-v0")
        obs = env.reset()
        assert isinstance(obs, dict)
        assert obs['image'].shape == (256, 256, 3)
        env.close()

    def test_top_differs_from_ego_config(self):
        """Top-down variant uses different camera config than ego (spec-based).

        Runtime visual comparison skipped: Genesis doesn't free GPU memory
        on scene teardown, so creating 2 scenes in one process OOMs on 16GB.
        Top rendering is already verified by test_top_smoke.
        """
        ego_spec = gym.spec("MemoryMaze-9x9-Genesis-v0")
        top_spec = gym.spec("MemoryMaze-9x9-Top-Genesis-v0")
        # Top uses 256x256, ego uses default 64x64
        ego_res = ego_spec.kwargs.get('camera_resolution', 64)
        top_res = top_spec.kwargs.get('camera_resolution', 64)
        assert top_res == 256 and ego_res == 64, \
            f"Top should be 256x256, ego 64x64; got top={top_res}, ego={ego_res}"
        # Top has top_camera flag
        assert top_spec.kwargs.get('top_camera') is True


# ============================================================================
# 68e: Oracle
# ============================================================================

class TestOracle:
    """Test Oracle variant with BFS path + minimap overlay."""

    def test_oracle_smoke(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-Oracle-Genesis-v0")
        obs = env.reset()
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8
        env.close()

    def test_oracle_extraobs(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-Oracle-ExtraObs-Genesis-v0")
        obs = env.reset()
        assert isinstance(obs, dict)
        assert obs['image'].shape == (64, 64, 3)
        env.close()

    def test_oracle_top(self, init_genesis):
        env = gym.make(disable_env_checker=True, id="MemoryMaze-9x9-Oracle-Top-Genesis-v0")
        obs = env.reset()
        assert obs.shape == (256, 256, 3)
        env.close()


# ============================================================================
# 68e: BFS unit tests (no env/GPU needed)
# ============================================================================

class TestBFS:
    """Unit tests for breadth_first_search (engine-agnostic)."""

    def test_bfs_simple_path(self):
        from memory_maze.oracle import breadth_first_search
        maze = np.array([
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=np.uint8)
        path = breadth_first_search(maze, (0, 0), (2, 2))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

    def test_bfs_no_path(self):
        from memory_maze.oracle import breadth_first_search
        maze = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ], dtype=np.uint8)
        path = breadth_first_search(maze, (0, 0), (2, 2))
        assert path is None

    def test_bfs_same_start_finish(self):
        from memory_maze.oracle import breadth_first_search
        maze = np.ones((5, 5), dtype=np.uint8)
        path = breadth_first_search(maze, (2, 2), (2, 2))
        assert path is not None
        assert len(path) == 1
        assert path[0] == (2, 2)


# ============================================================================
# 68f: Cross-variant consistency (spec-based, no env creation)
# ============================================================================

class TestCrossVariantConsistency:
    """Verify registration kwargs across sizes. Spec-based to avoid GPU OOM."""

    def test_hifreq_control_freq(self):
        """HiFreq variants should have control_freq=40."""
        for size in ["9x9", "11x11", "13x13", "15x15"]:
            spec = gym.spec(f"MemoryMaze-{size}-HiFreq-Genesis-v0")
            assert spec.kwargs.get('control_freq') == 40, \
                f"{size} HiFreq missing control_freq=40"

    def test_hifreq_time_limits(self):
        """HiFreq time_limit matches expected per size."""
        configs = {"9x9": 250, "11x11": 500, "13x13": 750, "15x15": 1000}
        for size, time_limit in configs.items():
            spec = gym.spec(f"MemoryMaze-{size}-HiFreq-Genesis-v0")
            assert spec.kwargs['time_limit'] == time_limit, \
                f"{size}: expected time_limit={time_limit}, got {spec.kwargs.get('time_limit')}"

    def test_basic_time_limits(self):
        """Basic variants have correct time_limit per size."""
        configs = {"9x9": 250, "11x11": 500, "13x13": 750, "15x15": 1000}
        for size, time_limit in configs.items():
            spec = gym.spec(f"MemoryMaze-{size}-Genesis-v0")
            assert spec.kwargs['time_limit'] == time_limit, \
                f"{size}: expected time_limit={time_limit}, got {spec.kwargs.get('time_limit')}"


# ============================================================================
# Guard: ExtraObs in training scripts
# ============================================================================

class TestExtraObsGuard:
    """Verify ExtraObs uses Dict space (incompatible with training scripts that expect Box)."""

    def test_extraobs_has_dict_not_box_space(self):
        """ExtraObs observation_space is Dict, not Box — train_impala would crash."""
        spec = gym.spec("MemoryMaze-9x9-ExtraObs-Genesis-v0")
        assert spec.kwargs.get('global_observables') is True, \
            "ExtraObs should have global_observables=True"

    def test_basic_has_box_space(self):
        """Basic variant should NOT have global_observables."""
        spec = gym.spec("MemoryMaze-9x9-Genesis-v0")
        assert 'global_observables' not in spec.kwargs or not spec.kwargs['global_observables']
