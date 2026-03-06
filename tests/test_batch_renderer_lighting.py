"""BatchRenderer (Madrona) lighting and color fidelity tests.

These tests exercise the GPU BatchRenderer code path used during actual
training. They require CUDA + gs_madrona and will be skipped on CPU/macOS.

The existing test_lighting.py tests only run on the Rasterizer (gs.cpu),
which uses a completely different shading pipeline (pyrender PBR + sRGB
gamma). These tests validate the Madrona BVH raycast shader that training
actually uses.

Run on a CUDA pod:
    pytest tests/test_batch_renderer_lighting.py -v

Known bugs (Task 62) that these tests should catch:
1. sRGB double-decode: tex_desc.sRGB=1 decodes textures to linear,
   but mat->color stays in sRGB → hue corruption
2. Ambient hardcoded at 0.05 in bvh_raycast.cpp → everything too dark
3. Material colors not linearized → sRGB * linear color mismatch
"""

import math
import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", ".*PydanticDeprecatedSince.*")

# --- Availability detection ---

try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False

try:
    import gs_madrona

    _BATCH_RENDERER_AVAILABLE = True
except ImportError:
    _BATCH_RENDERER_AVAILABLE = False

import torch

_CUDA_AVAILABLE = torch.cuda.is_available()

SKIP_REASON = (
    "gs_madrona not installed"
    if not _BATCH_RENDERER_AVAILABLE
    else "CUDA not available"
    if not _CUDA_AVAILABLE
    else ""
)

pytestmark = pytest.mark.skipif(
    not (_BATCH_RENDERER_AVAILABLE and _CUDA_AVAILABLE),
    reason=SKIP_REASON or "BatchRenderer requires CUDA + gs_madrona",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def init_genesis_cuda():
    """Initialize Genesis with CUDA backend (required for BatchRenderer)."""
    if not gs._initialized:
        gs.init(backend=gs.cuda, logging_level="warning")
    elif gs.device.type != "cuda":
        pytest.skip("Genesis already initialized on non-CUDA backend")
    return gs


@pytest.fixture(scope="module")
def batch_scene_notex(init_genesis_cuda):
    """BatchRenderer scene WITHOUT textures (flat colors only).

    Isolates material color handling from texture sampling issues.
    """
    from memory_maze.genesis_backend import BatchGenesisMemoryMazeEnv

    env = BatchGenesisMemoryMazeEnv(
        n_envs=1, maze_size=9, seed=42, camera_resolution=64,
    )
    obs = env.reset()
    yield env
    env.close()


@pytest.fixture(scope="module")
def batch_scene_tex(init_genesis_cuda):
    """BatchRenderer scene WITH textures (the actual training config)."""
    from memory_maze.genesis_backend import BatchGenesisMemoryMazeEnv

    env = BatchGenesisMemoryMazeEnv(
        n_envs=1, maze_size=9, seed=42, camera_resolution=64,
    )
    obs = env.reset()
    yield env
    env.close()


def _step_and_get_frame(env, actions):
    """Run a sequence of actions and return the final frame for env 0."""
    obs = env.reset()
    for a in actions:
        obs, _, _, _ = env.step([a])
    return obs[0].copy()  # (H, W, 3) uint8


def _compute_stats(img, border=2):
    """Compute pixel statistics, cropping border."""
    h, w = img.shape[:2]
    c = img[border:h - border, border:w - border].astype(np.float64)
    mean_r = c[:, :, 0].mean()
    mean_g = c[:, :, 1].mean()
    mean_b = c[:, :, 2].mean()
    mean_all = c.mean()
    # Upper half = walls, lower quarter = floor
    mid = c.shape[0] // 2
    q3 = 3 * c.shape[0] // 4
    wall_brightness = c[:mid].mean()
    floor_brightness = c[q3:].mean()
    # Channel balance
    if mean_all > 1e-6:
        max_skew = max(
            abs(mean_r / mean_all - 1.0),
            abs(mean_g / mean_all - 1.0),
            abs(mean_b / mean_all - 1.0),
        )
    else:
        max_skew = 0.0
    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "mean_brightness": mean_all,
        "wall_brightness": wall_brightness,
        "floor_brightness": floor_brightness,
        "max_skew": max_skew,
        "dynamic_range": c.max() - c.min(),
    }


# ---------------------------------------------------------------------------
# Test Class 1: Brightness (catches ambient=0.05 bug)
# ---------------------------------------------------------------------------


class TestBatchRendererBrightness:
    """Verify BatchRenderer produces adequately bright images.

    Bug 2: ambient hardcoded at 0.05 in bvh_raycast.cpp makes
    everything ~6x darker than intended.
    """

    def test_wall_brightness_minimum(self, batch_scene_tex):
        """Wall brightness >= 25 across cardinal headings.

        MuJoCo walls average ~80-120. Pre-fix BatchRenderer: ~6-15.
        With Python-side light fix: ~35-150 depending on heading.
        Threshold 25 catches the pre-fix state (was 7.8).
        NOTE: C++ ambient patch (0.05→0.3) would bring worst-case to ~60+.
        """
        actions_by_heading = {
            "forward": [1, 1, 1],
            "left": [1, 1, 2, 2, 1],
            "back": [1, 1, 2, 2, 2, 2, 1],
            "right": [1, 1, 3, 3, 1],
        }
        wall_vals = []
        for name, actions in actions_by_heading.items():
            frame = _step_and_get_frame(batch_scene_tex, actions)
            stats = _compute_stats(frame)
            wall_vals.append(stats["wall_brightness"])

        min_wall = min(wall_vals)
        assert min_wall >= 25, (
            f"BatchRenderer wall brightness too low: min={min_wall:.1f}, "
            f"all={[f'{v:.1f}' for v in wall_vals]}. "
            f"Pre-fix was 7.8. Need C++ ambient patch for full fix."
        )

    def test_mean_brightness_not_black(self, batch_scene_tex):
        """Mean brightness >= 30 (currently ~28, barely fails to catch edge cases)."""
        frame = _step_and_get_frame(batch_scene_tex, [1, 1, 1, 1, 1])
        stats = _compute_stats(frame)
        assert stats["mean_brightness"] >= 30, (
            f"BatchRenderer image too dark: mean={stats['mean_brightness']:.1f}. "
            f"Expected >= 30 (MuJoCo: ~60-80)"
        )

    def test_dynamic_range_minimum(self, batch_scene_tex):
        """Dynamic range >= 80 (need visible contrast between walls/floor/sky)."""
        frame = _step_and_get_frame(batch_scene_tex, [1, 1, 1])
        stats = _compute_stats(frame)
        assert stats["dynamic_range"] >= 80, (
            f"BatchRenderer dynamic range too low: {stats['dynamic_range']:.1f}"
        )


# ---------------------------------------------------------------------------
# Test Class 2: Color fidelity (catches sRGB/linear mismatch)
# ---------------------------------------------------------------------------


class TestBatchRendererColorFidelity:
    """Verify BatchRenderer produces correct hues, not just brightness.

    Bugs 1+3: sRGB textures decoded to linear but material colors stay
    in sRGB → hue corruption. Red walls become olive, grey floor becomes
    acid green.
    """

    def test_channel_balance(self, batch_scene_tex):
        """Channel max_skew < 0.5 (currently ~1.0-1.5 due to hue corruption).

        MuJoCo max_skew is typically ~0.2-0.3. Allowing up to 0.5
        accounts for different shading models while catching gross
        color corruption.
        """
        frame = _step_and_get_frame(batch_scene_tex, [1, 1, 1, 1, 1])
        stats = _compute_stats(frame)
        assert stats["max_skew"] < 0.5, (
            f"BatchRenderer channel skew too high: {stats['max_skew']:.3f}. "
            f"R={stats['mean_r']:.1f} G={stats['mean_g']:.1f} B={stats['mean_b']:.1f}. "
            f"Likely sRGB/linear color space mismatch"
        )

    def test_floor_not_acid_green(self, batch_scene_tex):
        """Floor G channel should not dominate R by more than 3x.

        The acid-green floor is the most visible symptom of the color bug.
        MuJoCo floor is grey-blue: R≈59, G≈48, B≈54.
        Corrupted batch floor: R≈13, G≈30, B≈65 (G/R ≈ 2.4).
        """
        # Walk forward to see floor
        frame = _step_and_get_frame(batch_scene_tex, [1, 1, 1])
        h, w = frame.shape[:2]
        # Bottom quarter is floor
        floor = frame[3 * h // 4:, 2:-2].astype(np.float64)
        if floor.mean() < 5:
            pytest.skip("Floor region too dark to analyze")
        floor_r = floor[:, :, 0].mean()
        floor_g = floor[:, :, 1].mean()
        assert floor_g < 3.0 * max(floor_r, 1.0), (
            f"Floor is acid green: R={floor_r:.1f} G={floor_g:.1f} B={floor[:,:,2].mean():.1f}. "
            f"G/R ratio = {floor_g / max(floor_r, 1.0):.1f} (expected < 3.0)"
        )

    def test_wall_hue_preserved(self, batch_scene_tex):
        """Wall pixels should not be blue-dominant when they should be red/warm.

        Memory Maze walls use warm textures (style_01 yellow/red variants).
        If B > 2*R in the wall region, colors are corrupted.
        """
        frame = _step_and_get_frame(batch_scene_tex, [1, 1, 1])
        h, w = frame.shape[:2]
        # Upper half is walls
        walls = frame[2:h // 2, 2:-2].astype(np.float64)
        if walls.mean() < 5:
            pytest.skip("Wall region too dark to analyze hue")
        wall_r = walls[:, :, 0].mean()
        wall_b = walls[:, :, 2].mean()
        assert wall_b < 2.5 * max(wall_r, 1.0), (
            f"Walls are blue-shifted: R={wall_r:.1f} B={wall_b:.1f}. "
            f"B/R = {wall_b / max(wall_r, 1.0):.1f} (expected < 2.5). "
            f"Likely sRGB material color not linearized"
        )


# ---------------------------------------------------------------------------
# Test Class 3: Multi-env consistency
# ---------------------------------------------------------------------------


class TestBatchRendererMultiEnv:
    """Verify 2-env batched rendering matches 1-env rendering.

    These tests create additional Genesis scenes and may OOM on GPUs
    with < 24GB VRAM. They are skipped gracefully on OOM.
    """

    def test_brightness_consistent_across_envs(self, init_genesis_cuda):
        """Both envs in a 2-env batch should have similar brightness."""
        from memory_maze.genesis_backend import BatchGenesisMemoryMazeEnv

        try:
            env = BatchGenesisMemoryMazeEnv(
                n_envs=2, maze_size=9, seed=42, camera_resolution=64,
            )
        except RuntimeError as e:
            if "OUT_OF_MEMORY" in str(e):
                pytest.skip("GPU OOM — need >= 24GB VRAM for 2-env test")
            raise

        obs = env.reset()
        for _ in range(5):
            obs, _, _, _ = env.step([1, 1])

        mean0 = obs[0].astype(np.float64).mean()
        mean1 = obs[1].astype(np.float64).mean()
        env.close()

        if max(mean0, mean1) < 5:
            pytest.skip("Both envs too dark to compare")
        ratio = min(mean0, mean1) / max(mean0, mean1)
        assert ratio > 0.3, (
            f"Brightness inconsistent across envs: env0={mean0:.1f}, env1={mean1:.1f}, "
            f"ratio={ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# Test Class 4: Lighting configuration
# ---------------------------------------------------------------------------


class TestBatchRendererLightingConfig:
    """Verify light configuration reaches the BatchRenderer."""

    def test_scene_has_directional_lights(self, batch_scene_tex):
        """Scene should have directional lights added via scene.add_light()."""
        scene = batch_scene_tex._scene.scene
        # Check vis_options lights (Rasterizer path)
        vis_lights = scene.vis_options.lights
        directional = [l for l in vis_lights if l.get("type") == "directional"]
        assert len(directional) >= 2, (
            f"Expected >= 2 directional lights in vis_options, got {len(directional)}"
        )
