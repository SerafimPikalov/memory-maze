#!/usr/bin/env python3
"""Record oracle-guided navigation videos for both backends.

Uses the SAME maze layout, spawn position, heading, and target for both
MuJoCo and Genesis so the videos are directly comparable.

Strategy: run MuJoCo first, extract the maze entity_layer, walker position,
heading, and target positions, then inject them into the Genesis scene.

Usage:
    python tests/record_oracle_nav.py
    # Outputs: oracle_nav_mujoco.mp4, oracle_nav_genesis.mp4
"""

import math
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import imageio
import numpy as np
from dm_control import composer
from PIL import Image

from memory_maze.oracle import breadth_first_search
from memory_maze.tasks import _memory_maze

try:
    import genesis as gs
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False

if HAS_GENESIS:
    from memory_maze.genesis_backend import (
        GenesisMazeScene,
        ACTION_SET,
        WALKER_RADIUS,
        TARGET_RADIUS,
        extract_wall_cells,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_passable_grid(entity_layer):
    passable = np.zeros(entity_layer.shape, dtype=np.uint8)
    for c in (" ", "P", "G"):
        passable |= (entity_layer == c)
    return passable


def _world_to_grid(world_x, world_y, maze_outer_size, xy_scale=2.0):
    offset = (maze_outer_size - 1) / 2.0
    col = int(round(world_x / xy_scale + offset))
    row = int(round(-world_y / xy_scale + offset))
    return col, row


def _grid_to_world(col, row, maze_outer_size, xy_scale=2.0):
    offset = (maze_outer_size - 1) / 2.0
    x = (col - offset) * xy_scale
    y = -(row - offset) * xy_scale
    return np.array([x, y])


def _choose_nav_action(heading, pos, waypoint):
    delta = waypoint - pos
    desired = math.atan2(delta[1], delta[0])
    error = (desired - heading + math.pi) % (2 * math.pi) - math.pi
    if abs(error) > 0.4:
        return 2 if error > 0 else 3
    elif abs(error) > 0.15:
        return 4 if error > 0 else 5
    else:
        return 1


def upscale(img, size=256):
    return np.array(Image.fromarray(img).resize((size, size), Image.NEAREST))


def navigate_and_record(step_fn, state_fn, waypoints, max_steps=500):
    """Reactive controller that records frames. Returns (frames, reward, wp_reached)."""
    frames = []
    wp_idx = 1
    total_reward = 0.0

    for _ in range(max_steps):
        if wp_idx >= len(waypoints):
            break

        state = state_fn()
        pos, heading = state["pos"], state["heading"]

        wp = waypoints[wp_idx]
        if np.linalg.norm(pos - wp) < 0.8:
            wp_idx += 1
            continue

        action = _choose_nav_action(heading, pos, wp)
        reward, frame = step_fn(action)
        total_reward += reward
        frames.append(upscale(frame))

    return frames, total_reward, wp_idx


def save_video(frames, path, fps=8):
    writer = imageio.get_writer(path, fps=fps, quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seed = 42
    out_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    mj_path = os.path.join(out_dir, "oracle_nav_mujoco.mp4")
    ge_path = os.path.join(out_dir, "oracle_nav_genesis.mp4")

    # ==== Step 1: MuJoCo — set up, extract shared state, record ====
    print(f"[MuJoCo] Setting up env (seed={seed})...")
    env = _memory_maze(9, 3, 250, discrete_actions=True, seed=seed)

    comp_env = env
    while hasattr(comp_env, "env"):
        comp_env = comp_env.env
        if isinstance(comp_env, composer.Environment):
            break

    task = comp_env._task
    walker = task._walker
    maze_obj = task._maze_arena._maze

    ts = env.reset()
    physics = comp_env.physics

    # Extract shared scenario
    entity_layer = maze_obj.entity_layer.copy()
    grid = _make_passable_grid(entity_layer)
    outer = entity_layer.shape[0]

    agent_world = physics.bind(walker.root_body).xpos[:2].copy()
    steer_joint = walker.mjcf_model.find("joint", "steer")
    agent_heading = -float(physics.bind(steer_joint).qpos[0])

    target_ix = task._current_target_ix
    target_world_positions = []
    for t in task._targets:
        target_world_positions.append(physics.bind(t.geom).xpos[:3].copy())

    agent_col, agent_row = _world_to_grid(agent_world[0], agent_world[1], outer)
    target_xy = target_world_positions[target_ix][:2]
    target_col, target_row = _world_to_grid(target_xy[0], target_xy[1], outer)

    print(f"  Agent at world ({agent_world[0]:.1f}, {agent_world[1]:.1f}) = grid ({agent_col},{agent_row})")
    print(f"  Target {target_ix} at world ({target_xy[0]:.1f}, {target_xy[1]:.1f}) = grid ({target_col},{target_row})")
    print(f"  Heading: {agent_heading:.2f} rad ({math.degrees(agent_heading):.0f} deg)")

    path = breadth_first_search(grid, (agent_col, agent_row), (target_col, target_row))
    if path is None:
        print("[MuJoCo] ERROR: No BFS path found!")
        env.close()
        return
    print(f"  BFS path: {len(path)} cells")

    waypoints = [_grid_to_world(c, r, outer) for c, r in path]

    # Record MuJoCo
    first_frame = [upscale(ts.observation["image"])]

    def mj_state():
        p = comp_env.physics
        pos = p.bind(walker.root_body).xpos[:2].copy()
        s = walker.mjcf_model.find("joint", "steer")
        h = -float(p.bind(s).qpos[0])
        return {"pos": pos, "heading": h}

    def mj_step(action):
        t = env.step(action)
        return (t.reward or 0.0), t.observation["image"]

    frames, reward, wp_reached = navigate_and_record(mj_step, mj_state, waypoints)
    frames = first_frame + frames
    print(f"[MuJoCo] Done: {len(frames)} frames, reward={reward}, wp={wp_reached}/{len(waypoints)}")
    save_video(frames, mj_path)
    print(f"[MuJoCo] Saved: {mj_path}")
    env.close()

    # ==== Step 2: Genesis — inject same maze/spawn/target, record ====
    if not HAS_GENESIS:
        print("\n[Genesis] Skipped: Genesis not installed")
        return

    print(f"\n[Genesis] Setting up scene with MuJoCo's maze layout...")
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")

    scene = GenesisMazeScene(
        maze_size=9,
        n_targets=3,
        use_textures=True,
    )
    scene.build()

    # Do a normal reset first to initialize the maze object
    rng = np.random.RandomState(seed)
    scene.reset(rng)

    # Now override: inject MuJoCo's entity_layer into the Genesis maze
    # and reconfigure walls, walker, targets to match exactly
    scene._maze._entity_layer[:] = entity_layer
    if scene.use_textures:
        from memory_maze.genesis_backend import _apply_block_variations
        _apply_block_variations(scene._maze)
        shuffled = scene.shuffled_wall_groups(rng)
    else:
        shuffled = None
    wall_segments = extract_wall_cells(scene._maze, scene.xy_scale, scene.z_height)
    scene._configure_walls_for_env(0, wall_segments, wall_groups=shuffled)

    # Place walker at exact MuJoCo position and heading
    scene.walker.set_pos(np.array([agent_world[0], agent_world[1], WALKER_RADIUS]))
    scene._walker_heading = agent_heading
    qw = math.cos(agent_heading / 2)
    qz = math.sin(agent_heading / 2)
    scene.walker.set_quat(np.array([qw, 0.0, 0.0, qz]))
    scene.walker.set_dofs_velocity(np.zeros(6))

    # Place targets at exact MuJoCo positions
    scene._target_world_positions = []
    target_height = -0.6  # default target_height_above_ground
    for i in range(scene.n_targets):
        if i < len(target_world_positions):
            tpos = target_world_positions[i].copy()
            # Use MuJoCo's z position directly
            scene.target_entities[i].set_pos(tpos)
            scene._target_world_positions.append(tpos)
        else:
            scene.target_entities[i].set_pos(np.array([0.0, 0.0, -10.0]))
            scene._target_world_positions.append(np.array([0.0, 0.0, -10.0]))

    scene._update_camera()

    # Render first frame
    first_obs = scene.render_egocentric()
    # Draw target border (same as GenesisMemoryMazeEnv)
    from memory_maze.genesis_backend import TARGET_COLORS
    color = TARGET_COLORS[target_ix]
    B = int(2 * math.sqrt(64 / 64))
    border_color = (color * 255 * 0.7).astype(np.uint8)
    first_obs[:, :B] = border_color
    first_obs[:, -B:] = border_color
    first_obs[:B, :] = border_color
    first_obs[-B:, :] = border_color

    first_frame_ge = [upscale(first_obs)]

    # Track target collection manually
    current_target_ix = target_ix
    ge_total_reward = 0.0

    def ge_state():
        pos = np.array(scene.get_walker_position()[:2], dtype=np.float64)
        return {"pos": pos, "heading": float(scene._walker_heading)}

    def ge_step(action):
        nonlocal ge_total_reward, current_target_ix
        continuous = ACTION_SET[action]
        scene.apply_action(continuous)
        scene.step()

        # Check target contact
        reward = 0.0
        walker_pos = scene.get_walker_position()
        contacts = scene.check_target_contacts(walker_pos)
        if contacts[current_target_ix]:
            reward = 1.0
            ge_total_reward += reward

        # Render with border
        img = scene.render_egocentric()
        c = TARGET_COLORS[current_target_ix]
        bc = (c * 255 * 0.7).astype(np.uint8)
        img[:, :B] = bc
        img[:, -B:] = bc
        img[:B, :] = bc
        img[-B:, :] = bc
        return reward, img

    frames_ge, reward_ge, wp_ge = navigate_and_record(ge_step, ge_state, waypoints)
    frames_ge = first_frame_ge + frames_ge
    print(f"[Genesis] Done: {len(frames_ge)} frames, reward={ge_total_reward}, wp={wp_ge}/{len(waypoints)}")
    save_video(frames_ge, ge_path)
    print(f"[Genesis] Saved: {ge_path}")


if __name__ == "__main__":
    main()
