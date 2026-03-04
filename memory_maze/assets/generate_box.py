"""Generate a UV-mapped unit box OBJ for the Memory Maze Genesis backend.

The box is 1x1x1 centered at origin. At runtime it gets scaled via
gs.morphs.Mesh(file=path, scale=(xy_scale, xy_scale, z_height)).

UV mapping:
  - Side faces (±X, ±Y): u [0,1], v [0,0.75]  (matches texrepeat_v = 0.75)
  - Top/bottom faces (±Z): u [0,1], v [0,1]
"""

import os

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textured_box.obj")


def generate_box_obj():
    lines = ["# UV-mapped unit box for Memory Maze Genesis backend", ""]

    # 24 vertices: 4 per face, ordered for outward-facing quads
    # Face order: +X, -X, +Y, -Y, +Z (top), -Z (bottom)
    vertices = [
        # +X face (right)
        (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5),
        # -X face (left)
        (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5),
        # +Y face (front)
        (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
        # -Y face (back)
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5),
        # +Z face (top)
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
        # -Z face (bottom)
        (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5),
    ]

    # UV coordinates: sides get v [0, 0.75], top/bottom get v [0, 1]
    side_uvs = [(0, 0), (1, 0), (1, 0.75), (0, 0.75)]
    cap_uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]

    # Normals per face
    normals = [
        (1, 0, 0),   # +X
        (-1, 0, 0),  # -X
        (0, 1, 0),   # +Y
        (0, -1, 0),  # -Y
        (0, 0, 1),   # +Z
        (0, 0, -1),  # -Z
    ]

    # Write vertices
    for v in vertices:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    lines.append("")

    # Write UVs: 4 side UVs (indices 1-4) + 4 cap UVs (indices 5-8)
    for uv in side_uvs:
        lines.append(f"vt {uv[0]} {uv[1]}")
    for uv in cap_uvs:
        lines.append(f"vt {uv[0]} {uv[1]}")
    lines.append("")

    # Write normals
    for n in normals:
        lines.append(f"vn {n[0]} {n[1]} {n[2]}")
    lines.append("")

    # Write faces: f v/vt/vn
    # Faces 0-3 are sides (use side UVs 1-4), faces 4-5 are caps (use cap UVs 5-8)
    for face_idx in range(6):
        v_base = face_idx * 4 + 1  # 1-indexed
        n_idx = face_idx + 1       # 1-indexed
        if face_idx < 4:  # side face
            uv_indices = [1, 2, 3, 4]
        else:  # cap face
            uv_indices = [5, 6, 7, 8]
        parts = []
        for i in range(4):
            parts.append(f"{v_base + i}/{uv_indices[i]}/{n_idx}")
        lines.append(f"f {' '.join(parts)}")

    lines.append("")

    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated {OUTPUT}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: 6 (quads)")
    print(f"  UVs: {len(side_uvs) + len(cap_uvs)}")
    print(f"  Normals: {len(normals)}")


if __name__ == "__main__":
    generate_box_obj()
