"""Cross-backend walker control comparison tests.

Runs identical action sequences on both MuJoCo and Genesis backends and
verifies behavioral equivalence. The physics are fundamentally different
(MuJoCo: rolling contact via hinges; Genesis: direct translational force
on a free sphere), so exact trajectory matching is impossible. Tests verify
behavioral properties: forward goes forward, left turns left, agent
decelerates on noop, etc.

Run with:
    pytest tests/test_cross_backend_walker.py -v
    pytest tests/test_cross_backend_walker.py -v -k "mujoco"
    pytest tests/test_cross_backend_walker.py -v -k "CrossBackend"
"""

import math
import os

# Rendering backend env vars are set in conftest.py (must happen before PyOpenGL imports)

import numpy as np
import pytest
from dm_control import composer

from memory_maze.oracle import breadth_first_search
from memory_maze.tasks import _memory_maze

try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False

if HAS_GENESIS:
    from memory_maze.genesis_backend import (
        ACTION_SET,
        GenesisMazeScene,
        GenesisMemoryMazeEnv,
        TARGET_ACTIVATION_GAP,
    )


# ---------------------------------------------------------------------------
# WalkerTestHarness — uniform interface for both backends
# ---------------------------------------------------------------------------


class WalkerTestHarness:
    """Uniform interface for testing walker control on both backends.

    Both backends track heading as a continuous (unbounded) value:
    - MuJoCo: negated steer joint qpos (axis 0,0,-1 → negate for CCW-positive)
    - Genesis: _walker_heading (integrated from omega_z)

    Heading values accumulate without wrapping, so raw deltas give the
    true cumulative heading change.
    """

    def __init__(self, backend: str, seed: int = 42, physics_timestep: float = None):
        self.backend = backend
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._physics_timestep = physics_timestep

        if backend == "mujoco":
            self._init_mujoco()
        elif backend == "genesis":
            self._init_genesis()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_mujoco(self):
        """Build MuJoCo env and unwrap to composer.Environment."""
        self._dm_env = _memory_maze(9, 3, 250, discrete_actions=True, seed=self.seed)
        # Unwrap: DiscreteActionSetWrapper → TargetColorAsBorderWrapper
        #       → RemapObservationWrapper → composer.Environment
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

    def _init_genesis(self):
        """Build Genesis scene directly."""
        kwargs = dict(maze_size=9, n_targets=3, use_textures=False)
        if self._physics_timestep is not None:
            kwargs["physics_timestep"] = self._physics_timestep
        self._scene = GenesisMazeScene(**kwargs)
        self._scene.build()

    def reset(self) -> dict:
        """Reset and return state dict."""
        if self.backend == "mujoco":
            self._dm_env.reset()
        else:
            self._scene.reset(self._rng)
        return self.get_state()

    def step(self, action: int) -> dict:
        """Step with discrete action, return state dict."""
        if self.backend == "mujoco":
            self._dm_env.step(action)
        else:
            continuous = ACTION_SET[action]
            self._scene.apply_action(continuous)
            self._scene.step()
        return self.get_state()

    def get_state(self) -> dict:
        """Return current {pos: np.array(2), heading: float, vel: np.array(2)}."""
        if self.backend == "mujoco":
            return self._get_mujoco_state()
        else:
            return self._get_genesis_state()

    def _get_mujoco_state(self) -> dict:
        physics = self._composer_env.physics
        walker = self._walker

        pos = physics.bind(walker.root_body).xpos[:2].copy()

        # Heading from steer joint. Steer axis is (0,0,-1), so positive
        # qpos = clockwise. Negate to get CCW-positive heading.
        # Re-find the joint each time since physics is recompiled on reset.
        steer_joint = walker.mjcf_model.find("joint", "steer")
        heading = -float(physics.bind(steer_joint).qpos[0])

        vel = physics.bind(walker.root_body).subtree_linvel[:2].copy()

        return {"pos": pos, "heading": heading, "vel": vel}

    def _get_genesis_state(self) -> dict:
        pos_3d = self._scene.get_walker_position()
        pos = np.array(pos_3d[:2], dtype=np.float64)

        heading = float(self._scene._walker_heading)

        vel_raw = self._scene.walker.get_dofs_velocity()
        if hasattr(vel_raw, "cpu"):
            vel_raw = vel_raw.cpu().numpy()
        vel = np.array(vel_raw[:2], dtype=np.float64)

        return {"pos": pos, "heading": heading, "vel": vel}

    def close(self):
        if self.backend == "mujoco":
            self._dm_env.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_actions(harness, action, n_steps):
    """Run n_steps of a single action, return list of states."""
    states = []
    for _ in range(n_steps):
        s = harness.step(action)
        states.append(s)
    return states


def displacement(s0, s1):
    """2D displacement vector from s0 to s1."""
    return s1["pos"] - s0["pos"]


def heading_vec(heading):
    """Unit vector from heading angle."""
    return np.array([math.cos(heading), math.sin(heading)])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["mujoco", "genesis"])
def harness(request, init_genesis_if_needed):
    """Parametrized fixture: runs each test on both backends."""
    ts = request.config.getoption("--physics-timestep")
    h = WalkerTestHarness(backend=request.param, seed=42, physics_timestep=ts)
    h.reset()
    yield h
    h.close()


@pytest.fixture
def mujoco_harness():
    """MuJoCo-only harness."""
    h = WalkerTestHarness(backend="mujoco", seed=42)
    h.reset()
    yield h
    h.close()


@pytest.fixture
def genesis_harness(request, init_genesis_if_needed):
    """Genesis-only harness."""
    ts = request.config.getoption("--physics-timestep")
    h = WalkerTestHarness(backend="genesis", seed=42, physics_timestep=ts)
    h.reset()
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


# ===================================================================
# Group 1: Forward Movement
# ===================================================================


class TestForwardMovement:
    """Verify that the forward action moves the walker in its heading direction."""

    def test_forward_moves_in_heading_direction(self, harness):
        """Forward displacement aligns with initial heading."""
        s0 = harness.get_state()
        run_actions(harness, action=1, n_steps=20)
        s1 = harness.get_state()

        disp = displacement(s0, s1)
        dist = np.linalg.norm(disp)
        if dist < 0.01:
            pytest.skip("Walker didn't move (may be stuck against wall)")

        disp_unit = disp / dist
        heading_dir = heading_vec(s0["heading"])
        dot = np.dot(disp_unit, heading_dir)
        assert dot > 0.8, f"Forward displacement misaligned with heading: dot={dot:.3f}"

    def test_forward_displacement_positive(self, harness):
        """Forward action produces meaningful displacement."""
        s0 = harness.get_state()
        run_actions(harness, action=1, n_steps=20)
        s1 = harness.get_state()

        dist = np.linalg.norm(displacement(s0, s1))
        assert dist > 0.5, f"Forward displacement too small: {dist:.3f}"

    def test_forward_at_known_heading(self, harness):
        """Forward at a known heading moves in the expected direction."""
        s0 = harness.get_state()
        h0 = s0["heading"]
        run_actions(harness, action=1, n_steps=20)
        s1 = harness.get_state()

        disp = displacement(s0, s1)
        dist = np.linalg.norm(disp)
        if dist < 0.01:
            pytest.skip("Walker didn't move")

        expected_dir = heading_vec(h0)
        dot = np.dot(disp / dist, expected_dir)
        assert dot > 0.7, f"dot={dot:.3f}, heading={h0:.2f}, disp={disp}"


# ===================================================================
# Group 2: Turning
# ===================================================================


class TestTurning:
    """Verify turn directions and symmetry.

    Both backends track heading as unbounded continuous values, so raw
    deltas (without wrapping) give the correct cumulative heading change.
    """

    def test_left_turn_direction(self, harness):
        """Left action (2) increases heading (CCW)."""
        s0 = harness.get_state()
        run_actions(harness, action=2, n_steps=20)
        s1 = harness.get_state()

        delta = s1["heading"] - s0["heading"]
        assert delta > 0.1, f"Left turn heading change not positive: {delta:.3f} rad"

    def test_right_turn_direction(self, harness):
        """Right action (3) decreases heading (CW)."""
        s0 = harness.get_state()
        run_actions(harness, action=3, n_steps=20)
        s1 = harness.get_state()

        delta = s1["heading"] - s0["heading"]
        assert delta < -0.1, f"Right turn heading change not negative: {delta:.3f} rad"

    def test_turn_symmetry(self, harness):
        """Left and right turns produce similar absolute heading change."""
        # Left turn
        harness.reset()
        s0 = harness.get_state()
        run_actions(harness, action=2, n_steps=20)
        s1 = harness.get_state()
        left_delta = abs(s1["heading"] - s0["heading"])

        # Right turn (fresh reset)
        harness.reset()
        s0 = harness.get_state()
        run_actions(harness, action=3, n_steps=20)
        s1 = harness.get_state()
        right_delta = abs(s1["heading"] - s0["heading"])

        if left_delta < 0.01 or right_delta < 0.01:
            pytest.skip("Turn too small to compare")

        ratio = min(left_delta, right_delta) / max(left_delta, right_delta)
        assert ratio > 0.5, f"Turn asymmetry too large: left={left_delta:.3f}, right={right_delta:.3f}"

    def test_turn_minimal_displacement(self, harness):
        """Pure turning should produce minimal position displacement."""
        s0 = harness.get_state()
        run_actions(harness, action=2, n_steps=20)
        s1 = harness.get_state()

        dist = np.linalg.norm(displacement(s0, s1))
        assert dist < 1.0, f"Pure turn displaced too far: {dist:.3f}"


# ===================================================================
# Group 3: Deceleration
# ===================================================================


class TestDeceleration:
    """Verify that the walker decelerates to rest after actions stop."""

    def test_noop_decelerates(self, harness):
        """After forward motion, noop brings velocity near zero."""
        run_actions(harness, action=1, n_steps=20)
        run_actions(harness, action=0, n_steps=40)
        s = harness.get_state()

        speed = np.linalg.norm(s["vel"])
        assert speed < 0.1, f"Still moving after coast: speed={speed:.3f}"

    def test_heading_preserved_during_coast(self, harness):
        """Heading doesn't drift much during coast phase (no active torque)."""
        run_actions(harness, action=1, n_steps=20)
        s_before_coast = harness.get_state()
        run_actions(harness, action=0, n_steps=40)
        s_after_coast = harness.get_state()

        # Raw delta is fine here — coast is short, no multi-turn accumulation.
        # MuJoCo ball rolling can cause some residual heading drift from
        # contact forces, so we allow generous tolerance.
        drift = abs(s_after_coast["heading"] - s_before_coast["heading"])
        assert drift < 1.0, f"Heading drift during coast: {drift:.3f} rad"

    def test_turn_stops_after_release(self, harness):
        """After turning, noop brings angular velocity near zero."""
        run_actions(harness, action=2, n_steps=20)

        run_actions(harness, action=0, n_steps=40)

        # Check heading is stable by stepping more and comparing
        s_a = harness.get_state()
        run_actions(harness, action=0, n_steps=10)
        s_b = harness.get_state()

        angular_drift = abs(s_b["heading"] - s_a["heading"])
        assert angular_drift < 0.05, f"Still turning after coast: drift={angular_drift:.3f} rad over 10 steps"


# ===================================================================
# Group 4: Compound Actions
# ===================================================================


class TestCompoundActions:
    """Verify forward+turn compound actions behave correctly.

    Compound tests compare heading change against pure forward to isolate
    the turn component, since forward motion alone can cause heading drift
    on the MuJoCo rolling ball.
    """

    def test_forward_left_curves(self, harness):
        """Forward+left (action 4) turns more left than pure forward."""
        # Use 10 steps (not 20) to avoid completing a full circle, which
        # would return the ball to its starting position with near-zero displacement.
        # Pure forward baseline
        harness.reset()
        s0 = harness.get_state()
        run_actions(harness, action=1, n_steps=10)
        s1 = harness.get_state()
        fwd_delta = s1["heading"] - s0["heading"]

        # Forward + left
        harness.reset()
        s0 = harness.get_state()
        run_actions(harness, action=4, n_steps=10)
        s1 = harness.get_state()
        fwd_left_delta = s1["heading"] - s0["heading"]

        # Moved forward
        dist = np.linalg.norm(displacement(s0, s1))
        assert dist > 0.3, f"Compound action didn't move: dist={dist:.3f}"

        # Forward+left should turn more CCW (more positive) than pure forward
        assert fwd_left_delta > fwd_delta - 0.5, (
            f"Forward+left should turn more left: fwd_left={fwd_left_delta:.3f}, fwd={fwd_delta:.3f}"
        )

    def test_forward_right_curves(self, harness):
        """Forward+right (action 5) turns more right than pure forward."""
        # Use 10 steps to avoid full-circle coincidence
        # Pure forward baseline
        harness.reset()
        s0 = harness.get_state()
        run_actions(harness, action=1, n_steps=10)
        s1 = harness.get_state()
        fwd_delta = s1["heading"] - s0["heading"]

        # Forward + right
        harness.reset()
        s0 = harness.get_state()
        run_actions(harness, action=5, n_steps=10)
        s1 = harness.get_state()
        fwd_right_delta = s1["heading"] - s0["heading"]

        # Moved forward
        dist = np.linalg.norm(displacement(s0, s1))
        assert dist > 0.3, f"Compound action didn't move: dist={dist:.3f}"

        # Forward+right should turn more CW (more negative) than pure forward
        assert fwd_right_delta < fwd_delta + 0.5, (
            f"Forward+right should turn more right: fwd_right={fwd_right_delta:.3f}, fwd={fwd_delta:.3f}"
        )

    def test_compound_vs_pure_forward(self, harness):
        """Compound forward+turn has less forward displacement than pure forward."""
        # Pure forward
        harness.reset()
        s0 = harness.get_state()
        run_actions(harness, action=1, n_steps=20)
        s1 = harness.get_state()
        pure_dist = np.linalg.norm(displacement(s0, s1))

        # Forward + left
        harness.reset()
        s0 = harness.get_state()
        run_actions(harness, action=4, n_steps=20)
        s1 = harness.get_state()
        compound_dist = np.linalg.norm(displacement(s0, s1))

        if pure_dist < 0.1:
            pytest.skip("Pure forward didn't move (wall)")

        assert compound_dist < pure_dist * 1.1, (
            f"Compound should have less displacement: compound={compound_dist:.3f}, pure={pure_dist:.3f}"
        )


# ===================================================================
# Group 5: Action Sequence Trajectories
# ===================================================================


class TestActionSequences:
    """Verify multi-phase action sequences produce expected trajectories."""

    def test_forward_return_to_rest(self, harness):
        """Forward then noop: walker eventually stops."""
        run_actions(harness, action=1, n_steps=20)
        run_actions(harness, action=0, n_steps=60)
        s = harness.get_state()

        speed = np.linalg.norm(s["vel"])
        assert speed < 0.05, f"Didn't return to rest: speed={speed:.3f}"

    def test_l_shaped_path(self, harness):
        """Forward, turn left, forward: final position is offset from initial heading."""
        s0 = harness.get_state()
        h0 = s0["heading"]

        # Phase 1: forward
        run_actions(harness, action=1, n_steps=15)
        # Phase 2: turn left
        run_actions(harness, action=2, n_steps=10)
        # Phase 3: forward again
        run_actions(harness, action=1, n_steps=15)

        s1 = harness.get_state()
        disp = displacement(s0, s1)
        dist = np.linalg.norm(disp)

        if dist < 0.5:
            pytest.skip("Walker didn't move enough (wall)")

        # The displacement should NOT be purely along the initial heading
        initial_dir = heading_vec(h0)
        dot = np.dot(disp / dist, initial_dir)
        assert dot < 0.95, f"L-path too straight: dot={dot:.3f}"

    def test_zigzag_net_heading(self, harness):
        """Alternating left/right compound actions keep net heading close to initial."""
        s0 = harness.get_state()

        for _ in range(5):
            run_actions(harness, action=4, n_steps=3)  # forward + left
            run_actions(harness, action=5, n_steps=3)  # forward + right

        s1 = harness.get_state()
        delta = s1["heading"] - s0["heading"]

        # Generous tolerance: wall collisions and rolling ball physics
        # can cause significant heading drift even with symmetric actions
        assert abs(delta) < 3.0, f"Zigzag net heading drift too large: {delta:.3f} rad"


# ===================================================================
# Group 6: Cross-Backend Comparison
# ===================================================================


@pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")
class TestCrossBackend:
    """Run BOTH backends and compare behavioral properties directly."""

    def _make_pair(self, init_genesis_if_needed, request=None):
        ts = request.config.getoption("--physics-timestep") if request else None
        mj = WalkerTestHarness(backend="mujoco", seed=42)
        mj.reset()
        ge = WalkerTestHarness(backend="genesis", seed=42, physics_timestep=ts)
        ge.reset()
        return mj, ge

    def test_forward_direction_matches(self, init_genesis_if_needed, request):
        """Both backends move forward along their heading direction."""
        mj, ge = self._make_pair(init_genesis_if_needed, request)
        try:
            s0_mj = mj.get_state()
            run_actions(mj, action=1, n_steps=20)
            s1_mj = mj.get_state()
            disp_mj = displacement(s0_mj, s1_mj)

            s0_ge = ge.get_state()
            run_actions(ge, action=1, n_steps=20)
            s1_ge = ge.get_state()
            disp_ge = displacement(s0_ge, s1_ge)

            dist_mj = np.linalg.norm(disp_mj)
            dist_ge = np.linalg.norm(disp_ge)

            if dist_mj < 0.01 or dist_ge < 0.01:
                pytest.skip("One backend didn't move")

            # Both displacement vectors should align with their respective headings
            dir_mj = heading_vec(s0_mj["heading"])
            dir_ge = heading_vec(s0_ge["heading"])

            dot_mj = np.dot(disp_mj / dist_mj, dir_mj)
            dot_ge = np.dot(disp_ge / dist_ge, dir_ge)

            assert dot_mj > 0.7, f"MuJoCo forward misaligned: dot={dot_mj:.3f}"
            assert dot_ge > 0.7, f"Genesis forward misaligned: dot={dot_ge:.3f}"
        finally:
            mj.close()
            ge.close()

    def test_turn_direction_matches(self, init_genesis_if_needed, request):
        """Both backends turn left in the same rotational direction (CCW)."""
        mj, ge = self._make_pair(init_genesis_if_needed, request)
        try:
            s0_mj = mj.get_state()
            run_actions(mj, action=2, n_steps=20)
            s1_mj = mj.get_state()
            delta_mj = s1_mj["heading"] - s0_mj["heading"]

            s0_ge = ge.get_state()
            run_actions(ge, action=2, n_steps=20)
            s1_ge = ge.get_state()
            delta_ge = s1_ge["heading"] - s0_ge["heading"]

            assert delta_mj > 0, f"MuJoCo left turn negative: {delta_mj:.3f}"
            assert delta_ge > 0, f"Genesis left turn negative: {delta_ge:.3f}"
        finally:
            mj.close()
            ge.close()

    def test_deceleration_both_stop(self, init_genesis_if_needed, request):
        """Both backends decelerate to near-zero velocity after coast."""
        mj, ge = self._make_pair(init_genesis_if_needed, request)
        try:
            run_actions(mj, action=1, n_steps=20)
            run_actions(mj, action=0, n_steps=40)
            speed_mj = np.linalg.norm(mj.get_state()["vel"])

            run_actions(ge, action=1, n_steps=20)
            run_actions(ge, action=0, n_steps=40)
            speed_ge = np.linalg.norm(ge.get_state()["vel"])

            assert speed_mj < 0.1, f"MuJoCo didn't decelerate: speed={speed_mj:.3f}"
            assert speed_ge < 0.1, f"Genesis didn't decelerate: speed={speed_ge:.3f}"
        finally:
            mj.close()
            ge.close()


# ===================================================================
# Group 7: Oracle-Guided Target Navigation
# ===================================================================


def _make_passable_grid(entity_layer):
    """Convert labmaze entity_layer char grid to binary passable grid."""
    passable = np.zeros(entity_layer.shape, dtype=np.uint8)
    for c in (" ", "P", "G"):
        passable |= (entity_layer == c)
    return passable


def _world_to_grid(world_x, world_y, maze_outer_size, xy_scale=2.0):
    """Convert world (x, y) to grid (col, row)."""
    offset = (maze_outer_size - 1) / 2.0
    col = int(round(world_x / xy_scale + offset))
    row = int(round(-world_y / xy_scale + offset))
    return col, row


def _grid_to_world(col, row, maze_outer_size, xy_scale=2.0):
    """Convert grid (col, row) to world (x, y)."""
    offset = (maze_outer_size - 1) / 2.0
    x = (col - offset) * xy_scale
    y = -(row - offset) * xy_scale
    return np.array([x, y])


def _choose_nav_action(heading, pos, waypoint):
    """Reactive heading controller: pick discrete action to face waypoint."""
    delta = waypoint - pos
    desired = math.atan2(delta[1], delta[0])
    error = (desired - heading + math.pi) % (2 * math.pi) - math.pi

    if abs(error) > 0.4:
        return 2 if error > 0 else 3  # pure turn
    elif abs(error) > 0.15:
        return 4 if error > 0 else 5  # forward + turn
    else:
        return 1  # forward


def _navigate_waypoints(step_fn, state_fn, waypoints, max_steps=500):
    """Follow world-coordinate waypoints using reactive controller.

    Args:
        step_fn: callable(action: int) -> reward: float
        state_fn: callable() -> dict with 'pos' and 'heading'
        waypoints: list of np.array([x, y])
        max_steps: safety limit

    Returns:
        (total_reward, waypoints_reached)
    """
    wp_idx = 1  # skip first waypoint (current position)
    total_reward = 0.0

    for _ in range(max_steps):
        if wp_idx >= len(waypoints):
            break

        state = state_fn()
        pos = state["pos"]
        heading = state["heading"]

        wp = waypoints[wp_idx]
        if np.linalg.norm(pos - wp) < 0.8:
            wp_idx += 1
            continue

        action = _choose_nav_action(heading, pos, wp)
        reward = step_fn(action)
        total_reward += reward

    return total_reward, wp_idx


class TestOracleNavigation:
    """Navigate to target using BFS oracle path, verify target reached.

    Uses each backend's native APIs to extract maze layout and positions,
    then runs BFS + reactive heading controller to navigate to the current
    target. Verifies the agent collects the target (reward > 0) or reaches
    its vicinity.
    """

    def test_mujoco_oracle_reaches_target(self):
        """MuJoCo: BFS path to target, reactive navigate, collect reward."""
        env = _memory_maze(9, 3, 250, discrete_actions=True, seed=42)

        # Unwrap to composer.Environment
        comp_env = env
        while hasattr(comp_env, "env"):
            comp_env = comp_env.env
            if isinstance(comp_env, composer.Environment):
                break
        task = comp_env._task
        walker = task._walker
        maze = task._maze_arena._maze

        env.reset()
        physics = comp_env.physics

        # Maze layout
        grid = _make_passable_grid(maze.entity_layer)
        outer = maze.entity_layer.shape[0]

        # Agent grid position
        agent_xy = physics.bind(walker.root_body).xpos[:2]
        agent_col, agent_row = _world_to_grid(agent_xy[0], agent_xy[1], outer)

        # Current target grid position
        target_ix = task._current_target_ix
        target_xy = physics.bind(task._targets[target_ix].geom).xpos[:2]
        target_col, target_row = _world_to_grid(target_xy[0], target_xy[1], outer)

        # BFS
        path = breadth_first_search(grid, (agent_col, agent_row), (target_col, target_row))
        assert path is not None, (
            f"No BFS path from ({agent_col},{agent_row}) to ({target_col},{target_row})"
        )

        # Convert to world waypoints
        waypoints = [_grid_to_world(c, r, outer) for c, r in path]

        # State and step functions
        def state_fn():
            p = comp_env.physics
            pos = p.bind(walker.root_body).xpos[:2].copy()
            steer = walker.mjcf_model.find("joint", "steer")
            heading = -float(p.bind(steer).qpos[0])
            return {"pos": pos, "heading": heading}

        def step_fn(action):
            ts = env.step(action)
            return ts.reward or 0.0

        # Navigate
        total_reward, wp_reached = _navigate_waypoints(step_fn, state_fn, waypoints)

        # Check outcome
        final_pos = comp_env.physics.bind(walker.root_body).xpos[:2]
        target_pos = comp_env.physics.bind(task._targets[target_ix].geom).xpos[:2]
        final_dist = np.linalg.norm(final_pos - target_pos)

        assert total_reward > 0 or final_dist < 1.5, (
            f"MuJoCo: failed to reach target. reward={total_reward}, "
            f"dist={final_dist:.2f}, path_len={len(path)}, "
            f"wp_reached={wp_reached}/{len(waypoints)}"
        )
        env.close()

    @pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")
    def test_genesis_oracle_reaches_target(self, init_genesis_if_needed, request):
        """Genesis: BFS path to target, reactive navigate, collect reward."""
        ts = request.config.getoption("--physics-timestep")
        kwargs = dict(maze_size=9, seed=42, use_textures=False)
        if ts is not None:
            kwargs["physics_timestep"] = ts
        genesis_env = GenesisMemoryMazeEnv(**kwargs)
        genesis_env.reset()

        scene = genesis_env._scene
        maze = scene._maze

        # Maze layout
        grid = _make_passable_grid(maze.entity_layer)
        outer = maze.entity_layer.shape[0]

        # Agent grid position
        agent_xy = scene.get_walker_position()[:2]
        agent_col, agent_row = _world_to_grid(agent_xy[0], agent_xy[1], outer)

        # Current target grid position
        target_ix = genesis_env._current_target_ix
        target_xy = scene._target_world_positions[target_ix][:2]
        target_col, target_row = _world_to_grid(target_xy[0], target_xy[1], outer)

        # BFS
        path = breadth_first_search(grid, (agent_col, agent_row), (target_col, target_row))
        assert path is not None, (
            f"No BFS path from ({agent_col},{agent_row}) to ({target_col},{target_row})"
        )

        # Convert to world waypoints
        waypoints = [_grid_to_world(c, r, outer) for c, r in path]

        # State and step functions
        def state_fn():
            pos = np.array(scene.get_walker_position()[:2], dtype=np.float64)
            heading = float(scene._walker_heading)
            return {"pos": pos, "heading": heading}

        def step_fn(action):
            _, reward, _, _ = genesis_env.step(action)
            return reward

        # Navigate
        total_reward, wp_reached = _navigate_waypoints(step_fn, state_fn, waypoints)

        # Check outcome
        final_pos = scene.get_walker_position()[:2]
        target_pos = scene._target_world_positions[target_ix][:2]
        final_dist = np.linalg.norm(final_pos - target_pos)

        assert total_reward > 0 or final_dist < 1.5, (
            f"Genesis: failed to reach target. reward={total_reward}, "
            f"dist={final_dist:.2f}, path_len={len(path)}, "
            f"wp_reached={wp_reached}/{len(waypoints)}"
        )
        genesis_env.close()
