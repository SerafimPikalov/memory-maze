#!/usr/bin/env python3
"""Compare camera positions between MuJoCo and Genesis frame-by-frame.

Runs the same action sequence (from MuJoCo oracle navigation) on both
backends using identical maze/spawn/target, and logs camera world position
at every step.

Usage:
    python tests/compare_camera.py
"""

import math
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
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
        WALKER_CAMERA_FORWARD_OFFSET,
        WALKER_CAMERA_HEIGHT,
        WALKER_RADIUS,
        TARGET_RADIUS,
        extract_wall_cells,
        _apply_block_variations,
    )


# ---------------------------------------------------------------------------
# Helpers (shared with record_oracle_nav.py)
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


# ---------------------------------------------------------------------------
# MuJoCo camera extraction
# ---------------------------------------------------------------------------

def get_mujoco_camera_pos(physics, walker):
    """Get egocentric camera world position from MuJoCo physics."""
    cam = walker.mjcf_model.find("camera", "egocentric")
    return physics.bind(cam).xpos.copy()


# ---------------------------------------------------------------------------
# Genesis camera extraction
# ---------------------------------------------------------------------------

def get_genesis_camera_pos(scene):
    """Compute Genesis camera world position from walker state."""
    pos = scene.get_walker_position()
    heading = scene._walker_heading
    cam_x = pos[0] + WALKER_CAMERA_FORWARD_OFFSET * math.cos(heading)
    cam_y = pos[1] + WALKER_CAMERA_FORWARD_OFFSET * math.sin(heading)
    cam_z = pos[2] + WALKER_CAMERA_HEIGHT
    return np.array([cam_x, cam_y, cam_z])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seed = 42

    # ==== MuJoCo setup ====
    print("Setting up MuJoCo...")
    env = _memory_maze(9, 3, 250, discrete_actions=True, seed=seed)
    comp_env = env
    while hasattr(comp_env, "env"):
        comp_env = comp_env.env
        if isinstance(comp_env, composer.Environment):
            break

    task = comp_env._task
    walker = task._walker
    maze_obj = task._maze_arena._maze

    env.reset()
    physics = comp_env.physics

    # Extract scenario
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

    path = breadth_first_search(grid, (agent_col, agent_row), (target_col, target_row))
    assert path is not None, "No BFS path"
    waypoints = [_grid_to_world(c, r, outer) for c, r in path]

    # ==== Phase 1: Run MuJoCo oracle, record actions ====
    print(f"Running MuJoCo oracle ({len(path)} cell path)...")
    mj_actions = []
    mj_cam_positions = []

    # Initial camera pos
    mj_cam_positions.append(get_mujoco_camera_pos(comp_env.physics, walker))

    wp_idx = 1
    for _ in range(500):
        if wp_idx >= len(waypoints):
            break
        p = comp_env.physics
        pos = p.bind(walker.root_body).xpos[:2].copy()
        s = walker.mjcf_model.find("joint", "steer")
        heading = -float(p.bind(s).qpos[0])

        wp = waypoints[wp_idx]
        if np.linalg.norm(pos - wp) < 0.8:
            wp_idx += 1
            continue

        action = _choose_nav_action(heading, pos, wp)
        mj_actions.append(action)
        env.step(action)
        mj_cam_positions.append(get_mujoco_camera_pos(comp_env.physics, walker))

    env.close()
    print(f"  {len(mj_actions)} actions recorded")

    # ==== Genesis setup — inject same maze ====
    if not HAS_GENESIS:
        print("\nGenesis not installed, skipping.")
        return

    print("\nSetting up Genesis with same maze...")
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")

    scene = GenesisMazeScene(maze_size=9, n_targets=3, use_textures=True)
    scene.build()

    rng = np.random.RandomState(seed)
    scene.reset(rng)

    # Inject MuJoCo maze layout
    scene._maze._entity_layer[:] = entity_layer
    if scene.use_textures:
        _apply_block_variations(scene._maze)
        scene.shuffle_wall_textures(rng)
    wall_segments = extract_wall_cells(scene._maze, scene.xy_scale, scene.z_height)
    scene._configure_walls(wall_segments)

    # Place walker at exact same position/heading
    scene.walker.set_pos(np.array([agent_world[0], agent_world[1], WALKER_RADIUS]))
    scene._walker_heading = agent_heading
    qw = math.cos(agent_heading / 2)
    qz = math.sin(agent_heading / 2)
    scene.walker.set_quat(np.array([qw, 0.0, 0.0, qz]))
    scene.walker.set_dofs_velocity(np.zeros(6))

    # Place targets
    scene._target_world_positions = []
    for i in range(scene.n_targets):
        if i < len(target_world_positions):
            tpos = target_world_positions[i].copy()
            scene.target_entities[i].set_pos(tpos)
            scene._target_world_positions.append(tpos)
        else:
            scene.target_entities[i].set_pos(np.array([0.0, 0.0, -10.0]))
            scene._target_world_positions.append(np.array([0.0, 0.0, -10.0]))
    scene._update_camera()

    # ==== Phase 2: Replay same actions on Genesis ====
    print(f"Replaying {len(mj_actions)} actions on Genesis...")
    ge_cam_positions = []
    ge_cam_positions.append(get_genesis_camera_pos(scene))

    for action in mj_actions:
        continuous = ACTION_SET[action]
        scene.apply_action(continuous)
        scene.step()
        ge_cam_positions.append(get_genesis_camera_pos(scene))

    # ==== Phase 3: Compare ====
    n = len(mj_cam_positions)
    assert len(ge_cam_positions) == n

    mj_pos = np.array(mj_cam_positions)
    ge_pos = np.array(ge_cam_positions)
    diff = mj_pos - ge_pos
    dist = np.linalg.norm(diff, axis=1)

    print(f"\n{'='*70}")
    print(f"Camera position comparison: {n} frames")
    print(f"{'='*70}")
    print(f"{'Frame':>5}  {'MuJoCo X':>9} {'Y':>9} {'Z':>9}  "
          f"{'Genesis X':>9} {'Y':>9} {'Z':>9}  {'Dist':>7}")
    print(f"{'-'*5}  {'-'*9} {'-'*9} {'-'*9}  {'-'*9} {'-'*9} {'-'*9}  {'-'*7}")

    # Print every 5th frame + first + last
    for i in range(n):
        if i == 0 or i == n - 1 or i % 5 == 0:
            m, g = mj_pos[i], ge_pos[i]
            print(f"{i:5d}  {m[0]:9.4f} {m[1]:9.4f} {m[2]:9.4f}  "
                  f"{g[0]:9.4f} {g[1]:9.4f} {g[2]:9.4f}  {dist[i]:7.4f}")

    print(f"\nSummary:")
    print(f"  Initial distance:  {dist[0]:.6f}")
    print(f"  Mean distance:     {dist.mean():.4f}")
    print(f"  Max distance:      {dist.max():.4f} (frame {dist.argmax()})")
    print(f"  Final distance:    {dist[-1]:.4f}")

    # XYZ breakdown
    print(f"\n  Mean |dx|: {np.abs(diff[:,0]).mean():.4f}")
    print(f"  Mean |dy|: {np.abs(diff[:,1]).mean():.4f}")
    print(f"  Mean |dz|: {np.abs(diff[:,2]).mean():.4f}")

    # Heading comparison
    print(f"\nHeading at key frames:")
    print(f"  Initial MuJoCo heading: {agent_heading:.4f} rad")
    print(f"  Initial Genesis heading: {scene._walker_heading:.4f} rad")


if __name__ == "__main__":
    main()
