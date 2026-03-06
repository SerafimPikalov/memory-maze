"""Lighting comparison tests for MuJoCo vs Genesis rendering.

Validates that Genesis lighting produces comparable visual quality to MuJoCo:
- Direction-independent illumination (walls visible from all headings)
- Adequate dynamic range and brightness
- No extreme color channel skew
- Regression tests against known-broken baseline values

Run with:
    MUJOCO_GL=glfw pytest tests/test_lighting.py -v
    MUJOCO_GL=glfw pytest tests/test_lighting.py -v -k "MuJoCo"
    MUJOCO_GL=glfw pytest tests/test_lighting.py -v -k "Genesis"
    MUJOCO_GL=glfw pytest tests/test_lighting.py -v -k "Regression"
"""

import math
import os

import numpy as np
import pytest
from dm_control import composer

from memory_maze.tasks import _memory_maze

try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False

if HAS_GENESIS:
    from memory_maze.genesis_backend import GenesisMazeScene


# ---------------------------------------------------------------------------
# LightingTestHarness — rendering-focused wrapper for both backends
# ---------------------------------------------------------------------------


class LightingTestHarness:
    """Uniform interface for rendering tests on both backends.

    Provides render_at_heading(heading) to capture frames at specific
    orientations without stepping physics. This isolates lighting/rendering
    from physics behavior.
    """

    def __init__(self, backend: str, seed: int = 42):
        self.backend = backend
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        if backend == "mujoco":
            self._init_mujoco()
        elif backend == "genesis":
            self._init_genesis()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_mujoco(self):
        self._dm_env = _memory_maze(9, 3, 250, discrete_actions=True, seed=self.seed)
        env = self._dm_env
        self._composer_env = None
        while hasattr(env, "env"):
            env = env.env
            if isinstance(env, composer.Environment):
                self._composer_env = env
                break
        if self._composer_env is None:
            raise RuntimeError("Could not find composer.Environment in wrapper chain")
        self._task = self._composer_env._task
        self._walker = self._task._walker
        self._steer = self._walker.mjcf_model.find("joint", "steer")
        self._cam = self._walker.egocentric_camera
        self._cam_id = None  # resolved after reset (physics recompiled)

    def _init_genesis(self):
        self._scene = GenesisMazeScene(maze_size=9, n_targets=3, use_textures=False)
        self._scene.build()

    def reset(self):
        if self.backend == "mujoco":
            self._dm_env.reset()
            physics = self._composer_env.physics
            self._cam_id = physics.model.name2id(self._cam.full_identifier, "camera")
        else:
            self._scene.reset(self._rng)

    def render_at_heading(self, heading: float) -> np.ndarray:
        """Render a frame at a specific heading without stepping physics.

        Args:
            heading: CCW-positive heading in radians.

        Returns:
            uint8 numpy array [H, W, 3].
        """
        if self.backend == "mujoco":
            return self._render_mujoco(heading)
        else:
            return self._render_genesis(heading)

    def _render_mujoco(self, heading: float) -> np.ndarray:
        physics = self._composer_env.physics
        # Steer axis is (0,0,-1), so negate heading for CCW-positive
        physics.bind(self._steer).qpos = -heading
        physics.forward()
        return physics.render(64, 64, camera_id=self._cam_id)

    def _render_genesis(self, heading: float) -> np.ndarray:
        self._scene._walker_heading = heading
        self._scene._update_camera()
        return self._scene.render_egocentric()

    def render_current(self) -> np.ndarray:
        """Render at the current heading."""
        if self.backend == "mujoco":
            return self._render_mujoco(self.get_heading())
        else:
            return self._scene.render_egocentric()

    def get_heading(self) -> float:
        if self.backend == "mujoco":
            physics = self._composer_env.physics
            return -float(physics.bind(self._steer).qpos[0])
        else:
            return float(self._scene._walker_heading)

    def prerender_cardinal_frames(self):
        """Pre-render frames at 4 cardinal headings and cache them.

        Called during fixture setup so cross-backend tests can compare
        cached frames without GL context conflicts on macOS.
        """
        self._cached_frames = {}
        headings = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        for h in headings:
            self._cached_frames[h] = self.render_at_heading(h).copy()

    def get_cached_frame(self, heading: float) -> np.ndarray:
        """Get a pre-rendered frame from the cache."""
        return self._cached_frames[heading]

    def close(self):
        if self.backend == "mujoco":
            self._dm_env.close()


# ---------------------------------------------------------------------------
# Pixel statistics helpers
# ---------------------------------------------------------------------------


def compute_pixel_stats(img: np.ndarray, border_width: int = 2) -> dict:
    """Compute pixel statistics from an RGB image, cropping border pixels."""
    h, w = img.shape[:2]
    b = border_width
    cropped = img[b : h - b, b : w - b].astype(np.float64)

    mean_r = cropped[:, :, 0].mean()
    mean_g = cropped[:, :, 1].mean()
    mean_b = cropped[:, :, 2].mean()
    mean_brightness = cropped.mean()

    min_pixel = cropped.min()
    max_pixel = cropped.max()
    dynamic_range = max_pixel - min_pixel

    # Upper half = mostly walls, lower quarter = mostly floor
    mid_h = cropped.shape[0] // 2
    quarter_h = 3 * cropped.shape[0] // 4
    wall_brightness = cropped[:mid_h].mean()
    floor_brightness = cropped[quarter_h:].mean()

    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "mean_brightness": mean_brightness,
        "dynamic_range": dynamic_range,
        "min_pixel": min_pixel,
        "max_pixel": max_pixel,
        "wall_brightness": wall_brightness,
        "floor_brightness": floor_brightness,
    }


def direction_independence_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Ratio of mean brightness between two images. 1.0 = perfectly matched."""
    mean_a = img_a.astype(np.float64).mean()
    mean_b = img_b.astype(np.float64).mean()
    if max(mean_a, mean_b) < 1e-6:
        return 1.0
    return min(mean_a, mean_b) / max(mean_a, mean_b)


def channel_balance_score(img: np.ndarray) -> dict:
    """Compute per-channel balance. max_skew = max deviation from 1.0."""
    f = img.astype(np.float64)
    overall = f.mean()
    if overall < 1e-6:
        return {"r_ratio": 1.0, "g_ratio": 1.0, "b_ratio": 1.0, "max_skew": 0.0}
    r_ratio = f[:, :, 0].mean() / overall
    g_ratio = f[:, :, 1].mean() / overall
    b_ratio = f[:, :, 2].mean() / overall
    max_skew = max(abs(r_ratio - 1.0), abs(g_ratio - 1.0), abs(b_ratio - 1.0))
    return {"r_ratio": r_ratio, "g_ratio": g_ratio, "b_ratio": b_ratio, "max_skew": max_skew}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mujoco_harness():
    """MuJoCo lighting harness — built once per module.

    Pre-renders cardinal frames before Genesis touches GL, so cross-backend
    comparisons work on macOS (where GL contexts conflict).
    """
    h = LightingTestHarness(backend="mujoco", seed=42)
    h.reset()
    h.prerender_cardinal_frames()
    yield h
    h.close()


@pytest.fixture(scope="module")
def genesis_harness(init_genesis_if_needed):
    """Genesis lighting harness — built once per module."""
    h = LightingTestHarness(backend="genesis", seed=42)
    h.reset()
    h.prerender_cardinal_frames()
    yield h
    h.close()


@pytest.fixture(scope="session")
def init_genesis_if_needed():
    """Initialize Genesis once per session, skip if not installed."""
    if not HAS_GENESIS:
        pytest.skip("Genesis not installed")
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")
    return gs


# ---------------------------------------------------------------------------
# Test Class 1: MuJoCo Lighting Baseline
# ---------------------------------------------------------------------------


class TestMuJoCoLightingBaseline:
    """Characterize MuJoCo rendering as the reference baseline.

    MuJoCo uses a camera-attached headlight, so brightness varies by heading
    (different walls/corridors visible). Measured values at seed=42:
    - Direction independence (0 vs pi): ~0.63
    - Dynamic range: 136-217 depending on heading
    - Wall brightness: 80-105 depending on heading
    """

    def test_mujoco_direction_independence(self, mujoco_harness):
        """Opposite headings brightness ratio > 0.55 (headlight follows camera)."""
        img_0 = mujoco_harness.render_at_heading(0.0)
        img_pi = mujoco_harness.render_at_heading(math.pi)
        score = direction_independence_score(img_0, img_pi)
        assert score > 0.55, f"MuJoCo direction dependence too high: score={score:.3f}"

    def test_mujoco_dynamic_range(self, mujoco_harness):
        """MuJoCo dynamic range > 100 (varies by heading, typically 136-217)."""
        img = mujoco_harness.render_at_heading(0.0)
        stats = compute_pixel_stats(img)
        assert stats["dynamic_range"] > 100, (
            f"MuJoCo dynamic range too low: {stats['dynamic_range']:.1f}"
        )

    def test_mujoco_wall_brightness(self, mujoco_harness):
        """Upper-half mean brightness (walls) should exceed 60."""
        img = mujoco_harness.render_at_heading(0.0)
        stats = compute_pixel_stats(img)
        assert stats["wall_brightness"] > 60, (
            f"MuJoCo wall brightness too low: {stats['wall_brightness']:.1f}"
        )


# ---------------------------------------------------------------------------
# Test Class 2: Genesis Pixel Stats
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")
class TestGenesisPixelStats:
    """Genesis rendering quality checks."""

    def test_dynamic_range_minimum(self, genesis_harness):
        """Genesis dynamic range should be >= 100 (broken baseline: 73 at worst heading)."""
        img = genesis_harness.render_at_heading(0.0)
        stats = compute_pixel_stats(img)
        assert stats["dynamic_range"] >= 100, (
            f"Genesis dynamic range too low: {stats['dynamic_range']:.1f}"
        )

    def test_max_pixel_above_threshold(self, genesis_harness):
        """Brightest pixel should be >= 150 (broken baseline: 118 at worst heading)."""
        img = genesis_harness.render_at_heading(0.0)
        stats = compute_pixel_stats(img)
        assert stats["max_pixel"] >= 150, (
            f"Genesis max pixel too dim: {stats['max_pixel']:.1f}"
        )

    def test_wall_brightness_reasonable(self, genesis_harness):
        """Minimum wall brightness across 4 headings should be >= 55."""
        headings = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        wall_vals = []
        for h in headings:
            img = genesis_harness.render_at_heading(h)
            stats = compute_pixel_stats(img)
            wall_vals.append(stats["wall_brightness"])
        min_wall = min(wall_vals)
        assert min_wall >= 55, (
            f"Genesis wall brightness too low: min={min_wall:.1f}, "
            f"all={[f'{v:.1f}' for v in wall_vals]}"
        )

    def test_no_channel_extreme_skew(self, genesis_harness):
        """No single color channel should be > 2x the overall mean."""
        img = genesis_harness.render_at_heading(0.0)
        balance = channel_balance_score(img)
        assert balance["max_skew"] < 1.0, (
            f"Genesis channel skew too high: r={balance['r_ratio']:.2f}, "
            f"g={balance['g_ratio']:.2f}, b={balance['b_ratio']:.2f}"
        )


# ---------------------------------------------------------------------------
# Test Class 3: Direction Independence
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")
class TestDirectionIndependence:
    """Core lighting correctness: illumination shouldn't depend on heading."""

    def test_opposite_headings_similar_brightness(self, genesis_harness):
        """Opposite headings (0 vs pi) brightness ratio > 0.7."""
        img_0 = genesis_harness.render_at_heading(0.0)
        img_pi = genesis_harness.render_at_heading(math.pi)
        score = direction_independence_score(img_0, img_pi)
        assert score > 0.7, (
            f"Opposite heading brightness ratio too low: {score:.3f}"
        )

    def test_four_cardinal_headings_similar(self, genesis_harness):
        """Four cardinal headings should have min/max brightness ratio > 0.65."""
        headings = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        means = []
        for h in headings:
            img = genesis_harness.render_at_heading(h)
            means.append(img.astype(np.float64).mean())
        ratio = min(means) / max(means) if max(means) > 0 else 1.0
        assert ratio > 0.65, (
            f"Cardinal heading brightness ratio too low: {ratio:.3f}, "
            f"means={[f'{m:.1f}' for m in means]}"
        )

    def test_perpendicular_headings_similar(self, genesis_harness):
        """Perpendicular headings (0 vs pi/2) brightness ratio > 0.6."""
        img_0 = genesis_harness.render_at_heading(0.0)
        img_90 = genesis_harness.render_at_heading(math.pi / 2)
        score = direction_independence_score(img_0, img_90)
        assert score > 0.6, (
            f"Perpendicular heading brightness ratio too low: {score:.3f}"
        )


# ---------------------------------------------------------------------------
# Test Class 4: Cross-Backend Visual Comparison
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")
class TestCrossBackendVisual:
    """Compare Genesis rendering against MuJoCo reference.

    Uses pre-rendered cached frames to avoid GL context conflicts on macOS
    (MuJoCo GLFW and Genesis OpenGL Rasterizer can't share GL in one process).
    """

    def test_mean_brightness_within_factor(self, mujoco_harness, genesis_harness):
        """Genesis mean brightness should be within 2x of MuJoCo's (ratio > 0.5)."""
        mj_img = mujoco_harness.get_cached_frame(0.0)
        ge_img = genesis_harness.get_cached_frame(0.0)
        mj_mean = mj_img.astype(np.float64).mean()
        ge_mean = ge_img.astype(np.float64).mean()
        ratio = min(mj_mean, ge_mean) / max(mj_mean, ge_mean) if max(mj_mean, ge_mean) > 0 else 1.0
        assert ratio > 0.5, (
            f"Brightness mismatch: MuJoCo={mj_mean:.1f}, Genesis={ge_mean:.1f}, ratio={ratio:.3f}"
        )

    def test_dynamic_range_comparable(self, mujoco_harness, genesis_harness):
        """Genesis average dynamic range should be >= 50% of MuJoCo's average."""
        headings = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        mj_ranges = []
        ge_ranges = []
        for h in headings:
            mj_ranges.append(compute_pixel_stats(mujoco_harness.get_cached_frame(h))["dynamic_range"])
            ge_ranges.append(compute_pixel_stats(genesis_harness.get_cached_frame(h))["dynamic_range"])
        mj_avg = np.mean(mj_ranges)
        ge_avg = np.mean(ge_ranges)
        ratio = ge_avg / mj_avg if mj_avg > 0 else 1.0
        assert ratio >= 0.5, (
            f"Genesis dynamic range too low vs MuJoCo: "
            f"Genesis avg={ge_avg:.1f}, MuJoCo avg={mj_avg:.1f}, ratio={ratio:.2f}"
        )

    def test_direction_independence_gap(self, mujoco_harness, genesis_harness):
        """Genesis direction independence should be >= 80% of MuJoCo's (or > 0.75)."""
        mj_0 = mujoco_harness.get_cached_frame(0.0)
        mj_pi = mujoco_harness.get_cached_frame(math.pi)
        mj_score = direction_independence_score(mj_0, mj_pi)

        ge_0 = genesis_harness.get_cached_frame(0.0)
        ge_pi = genesis_harness.get_cached_frame(math.pi)
        ge_score = direction_independence_score(ge_0, ge_pi)

        assert ge_score >= 0.8 * mj_score or ge_score > 0.75, (
            f"Genesis direction independence too low: "
            f"Genesis={ge_score:.3f}, MuJoCo={mj_score:.3f}"
        )


# ---------------------------------------------------------------------------
# Test Class 5: Lighting Regression
# ---------------------------------------------------------------------------

# Broken baseline: measured worst-case values with only 2 downward-pointing lights.
# Heading 0: drange=73, max_pixel=118, blue ratio=0.51 (severely blue-starved).
# Heading pi/2: wall_brightness=44.9.
# These thresholds are set just above the broken worst-case so they fail
# before the lighting fix and pass after.
BROKEN_MIN_DYNAMIC_RANGE = 80   # worst heading gets 73
BROKEN_MIN_WALL_BRIGHTNESS = 50  # worst heading gets 44.9
BROKEN_MAX_CHANNEL_DEVIATION = 0.45  # worst heading: |0.51-1| = 0.49


@pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")
class TestLightingRegression:
    """Assert that fixed lighting beats the known-broken baseline.

    These tests use worst-case-across-headings metrics that fail with the
    broken 2-light setup and pass after adding cardinal directional lights.
    """

    def test_min_dynamic_range_improved(self, genesis_harness):
        """Minimum dynamic range across cardinal headings should exceed 80."""
        headings = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        ranges = []
        for h in headings:
            img = genesis_harness.render_at_heading(h)
            stats = compute_pixel_stats(img)
            ranges.append(stats["dynamic_range"])
        min_range = min(ranges)
        assert min_range > BROKEN_MIN_DYNAMIC_RANGE, (
            f"Min dynamic range not improved: {min_range:.1f} <= {BROKEN_MIN_DYNAMIC_RANGE}, "
            f"all={[f'{r:.0f}' for r in ranges]}"
        )

    def test_min_wall_brightness_improved(self, genesis_harness):
        """Minimum wall brightness across cardinal headings should exceed 50."""
        headings = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        walls = []
        for h in headings:
            img = genesis_harness.render_at_heading(h)
            stats = compute_pixel_stats(img)
            walls.append(stats["wall_brightness"])
        min_wall = min(walls)
        assert min_wall > BROKEN_MIN_WALL_BRIGHTNESS, (
            f"Min wall brightness not improved: {min_wall:.1f} <= {BROKEN_MIN_WALL_BRIGHTNESS}, "
            f"all={[f'{w:.1f}' for w in walls]}"
        )

    def test_channel_balance_improved(self, genesis_harness):
        """Max channel deviation across cardinal headings should be < 0.45."""
        headings = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        max_dev = 0.0
        worst_heading = 0
        for h in headings:
            img = genesis_harness.render_at_heading(h)
            balance = channel_balance_score(img)
            if balance["max_skew"] > max_dev:
                max_dev = balance["max_skew"]
                worst_heading = h
        assert max_dev < BROKEN_MAX_CHANNEL_DEVIATION, (
            f"Channel balance not improved: max_deviation={max_dev:.3f} >= {BROKEN_MAX_CHANNEL_DEVIATION} "
            f"(worst heading={worst_heading:.2f})"
        )


# ---------------------------------------------------------------------------
# Test Class 6: Lighting Configuration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")
class TestLightingConfiguration:
    """Code-level verification of lighting setup."""

    def test_genesis_scene_has_cardinal_lights(self, genesis_harness):
        """Scene should have >= 4 directional lights for direction-independent illumination."""
        lights = genesis_harness._scene.scene.vis_options.lights
        directional = [l for l in lights if l.get("type") == "directional"]
        assert len(directional) >= 4, (
            f"Expected >= 4 directional lights, got {len(directional)}: {lights}"
        )

    def test_ambient_light_reasonable(self, genesis_harness):
        """Ambient light channels should each be in [0.3, 0.8]."""
        ambient = genesis_harness._scene.scene.vis_options.ambient_light
        for i, ch in enumerate(ambient):
            assert 0.3 <= ch <= 0.8, (
                f"Ambient light channel {i} out of range: {ch} (expected [0.3, 0.8])"
            )

    def test_lights_cover_multiple_directions(self, genesis_harness):
        """Lights should cover multiple distinct directions (not all pointing the same way)."""
        lights = genesis_harness._scene.scene.vis_options.lights
        directional = [l for l in lights if l.get("type") == "directional"]
        if len(directional) < 2:
            pytest.fail(f"Need >= 2 directional lights to check coverage, got {len(directional)}")

        # Extract direction vectors and check they're not all parallel
        dirs = []
        for l in directional:
            d = np.array(l["dir"], dtype=np.float64)
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                dirs.append(d / norm)

        # Check that at least two lights have significantly different directions
        max_angle = 0.0
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                dot = abs(np.dot(dirs[i], dirs[j]))
                angle = math.acos(min(dot, 1.0))
                max_angle = max(max_angle, angle)

        assert max_angle > math.radians(30), (
            f"All lights point in nearly the same direction (max angle={math.degrees(max_angle):.1f} deg)"
        )
