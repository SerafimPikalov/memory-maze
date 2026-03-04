# CLAUDE.md — Memory Maze

This is the Memory Maze benchmark (arxiv.org/abs/2210.13383), with both MuJoCo (original) and Genesis (ported) backends. This directory has its own `.git` — it's a separate repository from the parent project.

## Package Structure (`memory_maze/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Gym environment registration (MuJoCo + Genesis variants) |
| `tasks.py` | dm_control task definitions for each maze size |
| `maze.py` | Maze generation using `labmaze` |
| `wrappers.py` | dm_env observation wrappers |
| `gym_wrappers.py` | dm_env → gym.Env adapter (`GymWrapper`) |
| `oracle.py` | `PathToTargetWrapper` — BFS shortest-path oracle for validation |
| `genesis_backend.py` | Genesis-based drop-in replacement |
| `assets/` | UV-mapped box OBJ for textured walls |
| `helpers.py` | Shared utilities |

## Environment IDs

| Size | MuJoCo | Genesis | Objects | Steps |
|------|--------|---------|---------|-------|
| 9x9 | `MemoryMaze-9x9-v0` | `MemoryMaze-9x9-Genesis-v0` | 3 | 1000 |
| 11x11 | `MemoryMaze-11x11-v0` | `MemoryMaze-11x11-Genesis-v0` | 4 | 2000 |
| 13x13 | `MemoryMaze-13x13-v0` | `MemoryMaze-13x13-Genesis-v0` | 5 | 3000 |
| 15x15 | `MemoryMaze-15x15-v0` | `MemoryMaze-15x15-Genesis-v0` | 6 | 4000 |

MuJoCo variants: append `-ExtraObs` for debug observations, `-HD` for 256x256, `-Top` for top-down, `-Oracle` for shortest-path overlay, `-HiFreq` for 40Hz control, `-6CL` for 6-color targets.

## Genesis Backend (`genesis_backend.py`)

Drop-in replacement for MuJoCo — same `gym.Env` interface, same observation/action spaces.

- **Scene**: 225 pre-allocated wall entities (9 texture groups × 25), plane floor, `gs.renderers.Rasterizer()`
- **Wall textures**: UV-mapped OBJ mesh (`assets/textured_box.obj`) with labmaze `style_01` PNG textures; 9 spatial blocks (`'0'`–`'8'`) each with a distinct texture, matching MuJoCo's `TextMazeVaryingWalls`. Palette-mode PNGs pre-converted to RGB. `use_textures=True` by default, `False` for flat colors.
- **Floor texture**: Plane with labmaze `blue` floor texture (Plane has native UVs)
- **Walker**: `gs.morphs.Sphere(radius=0.2)` with density matching MuJoCo's 21 kg, force-based control via `control_dofs_force()`
- **Camera**: Manual `camera.set_pose()` per control step, fov=80, 64x64, height 0.7m above ball + 0.15m forward offset (matching MuJoCo)
- **Targets**: Non-colliding spheres, distance-based activation (gap=0.8m), color cycling
- **BatchRenderer**: Optional Madrona-based batch renderer (`gs_madrona`) for GPU-only headless rendering
- **macOS fix**: `GENESIS_SKIP_TK_INIT=1` and `MPLBACKEND=Agg` env vars prevent Tk/matplotlib crashes in subprocesses

## Build & Test

```bash
pip install -e .                                      # editable install
pytest                                                # run tests
python gui/run_gui.py                                 # interactive GUI (needs: pygame pillow imageio)
python gui/run_gui.py --env "memory_maze:MemoryMaze-9x9-HD-v0"  # high-res GUI
```

## Analysis Documents (in this directory)

| File | Content |
|------|---------|
| `architecture_report.md` | Full codebase architecture analysis |
| `batch_simulation_options_analysis.md` | Batch simulation approach comparison |
| `review_genesis.md` | Genesis integration review |
| `review_madrona.md` | Madrona integration review |
| `review_mujoco.md` | MuJoCo backend review |
| `review_synthesis.md` | Cross-review synthesis |
