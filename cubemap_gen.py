"""
Cubemap generator for Lichtfeld Studio
--------------------------------------
For each of 6 faces:
  1. render_view with black bg  → BW2A black temp
  2. render_view with white bg  → BW2A white temp
  3. BW2A                       → RGBA splat with correct alpha
  4. Sample HDRI equirectangular into cube face
  5. Composite RGBA splat over HDRI face
  6. Save as {id}_{face}.png
"""

import numpy as np
from pathlib import Path
from PIL import Image

import lichtfeld as lf
import lichtfeld.io as lio

# ── Configure these ───────────────────────────────────────
output_folder = r"B:\[LFS]\CubeMap-TEST"
eye_distance  = -1.0  # metres along Z from origin
size          = 1024  # output face size in pixels (square)
# ─────────────────────────────────────────────────────────

Path(output_folder).mkdir(parents=True, exist_ok=True)

view  = lf.get_current_view()
rs    = lf.get_render_settings()
scene = lf.get_scene()
nodes = scene.get_visible_nodes()
id_prefix = nodes[0].name if nodes else "cubemap"

eye = (0.0, 0.0, float(eye_distance))

# ── Cubemap face definitions ──────────────────────────────
# Each face: (target offset from eye, up vector)
# The forward vector for each face determines which part of
# the equirectangular HDRI gets sampled.
FACES = {
    "Front":  {"target": (0, 0,  1), "up": (0,  1,  0),
               "right": (-1,  0,  0), "fwd": ( 0,  0,  1), "upv": (0,  1,  0)},
    "Back":   {"target": (0, 0, -1), "up": (0,  1,  0),
               "right": ( 1,  0,  0), "fwd": ( 0,  0, -1), "upv": (0,  1,  0)},
    "Right":  {"target": (1, 0,  0), "up": (0,  1,  0),
               "right": ( 0,  0, -1), "fwd": ( 1,  0,  0), "upv": (0,  1,  0)},
    "Left":   {"target": (-1, 0, 0), "up": (0,  1,  0),
               "right": ( 0,  0,  1), "fwd": (-1,  0,  0), "upv": (0,  1,  0)},
    "Top":    {"target": (0,  1, 0), "up": (0,  0, -1),
               "right": ( 1,  0,  0), "fwd": ( 0,  1,  0), "upv": (0,  0, -1)},
    "Bottom": {"target": (0, -1, 0), "up": (0,  0,  1),
               "right": ( 1,  0,  0), "fwd": ( 0, -1,  0), "upv": (0,  0,  1)},
}


# ── HDRI sampling ─────────────────────────────────────────

def load_hdri():
    """Load the HDRI from render settings path."""
    path = rs.environment_map_path
    if not path:
        return None
    ext = Path(path).suffix.lower()
    if ext in (".hdr", ".exr"):
        import imageio.v2 as imageio
        arr = imageio.imread(path)
        return np.array(arr, dtype=np.float32) / 255.0
    else:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0


def sample_hdri_face(hdri, face_info, out_size, rotation_deg=0.0):
    """
    Sample the equirectangular HDRI into a cubemap face.
    Returns a float32 (H, W, 3) array.
    """
    if hdri is None:
        return np.zeros((out_size, out_size, 3), dtype=np.float32)

    h_src, w_src = hdri.shape[:2]
    fwd   = np.array(face_info["fwd"],   dtype=np.float64)
    right = np.array(face_info["right"], dtype=np.float64)
    upv   = np.array(face_info["upv"],   dtype=np.float64)

    # Build pixel grid — each pixel maps to a ray in [-1,1] tangent space
    xs = (np.arange(out_size) + 0.5) / out_size * 2.0 - 1.0  # -1..1
    ys = (np.arange(out_size) + 0.5) / out_size * 2.0 - 1.0

    # (out_size, out_size, 3) ray directions
    xx, yy = np.meshgrid(xs, -ys)  # flip y so top is up
    rays = (fwd[None, None, :] +
            xx[:, :, None] * right[None, None, :] +
            yy[:, :, None] * upv[None, None, :])

    # Normalise
    norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / norms

    rx, ry, rz = rays[..., 0], rays[..., 1], rays[..., 2]

    # Convert to spherical — apply HDRI rotation
    rot_rad = np.radians(rotation_deg)
    lon = np.arctan2(rx, -rz) + rot_rad   # -π..π, offset by rotation
    lon = (lon + np.pi) % (2 * np.pi) - np.pi  # keep in -π..π
    lat = np.arcsin(np.clip(ry, -1.0, 1.0))     # -π/2..π/2

    # Map to pixel coords
    u = ((lon / (2 * np.pi)) + 0.5) * w_src
    v = (0.5 - lat / np.pi) * h_src

    # Bilinear sample
    u0 = np.floor(u).astype(int) % w_src
    v0 = np.clip(np.floor(v).astype(int), 0, h_src - 1)
    u1 = (u0 + 1) % w_src
    v1 = np.clip(v0 + 1, 0, h_src - 1)
    fu = u - np.floor(u)
    fv = v - np.floor(v)

    c00 = hdri[v0, u0]
    c10 = hdri[v1, u0]
    c01 = hdri[v0, u1]
    c11 = hdri[v1, u1]

    sampled = (c00 * (1 - fu)[..., None] * (1 - fv)[..., None] +
               c01 * fu[..., None]       * (1 - fv)[..., None] +
               c10 * (1 - fu)[..., None] * fv[..., None] +
               c11 * fu[..., None]       * fv[..., None])

    return sampled.astype(np.float32)


# ── BW2A helpers ──────────────────────────────────────────

def make_bg(r, g, b):
    bg = view.translation * 0.0
    bg[0], bg[1], bg[2] = r, g, b
    return bg


def render_face_rgba(face_info, render_size):
    """
    Render a cubemap face as RGBA using BW2A.
    Returns PIL RGBA image at render_size x render_size.
    """
    ex, ey, ez = eye
    tx, ty, tz = face_info["target"]
    target = (ex + tx, ey + ty, ez + tz)
    up     = face_info["up"]

    rot, trans = lf.look_at(eye=eye, target=target, up=up)

    # Black bg render
    img_black = lf.render_view(rot, trans,
                               width=render_size, height=render_size,
                               fov=90.0, bg_color=make_bg(0, 0, 0))
    # White bg render
    img_white = lf.render_view(rot, trans,
                               width=render_size, height=render_size,
                               fov=90.0, bg_color=make_bg(1, 1, 1))

    # Convert to numpy
    black = np.array(Image.open(_tensor_to_pil(img_black)).convert("RGB")).astype(float)
    white = np.array(Image.open(_tensor_to_pil(img_white)).convert("RGB")).astype(float)

    # BW2A
    diff  = white - black
    alpha = 1.0 - np.clip(np.mean(diff, axis=2) / 255.0, 0, 1)
    recovered = np.clip(black / (alpha[:, :, None] + 1e-10), 0, 255).astype(np.uint8)
    alpha_u8  = (alpha * 255).astype(np.uint8)
    rgba      = np.dstack((recovered, alpha_u8))
    return Image.fromarray(rgba, "RGBA")


def _tensor_to_pil(tensor):
    """Save lichtfeld tensor to a temp file and return path."""
    tmp = str(Path(output_folder) / "_tmp_render.png")
    lio.save_image(tmp, tensor)
    return tmp


# ── Main loop ─────────────────────────────────────────────

print(f"Loading HDRI: {rs.environment_map_path}")
hdri = load_hdri()
if hdri is None:
    print("WARNING: no HDRI loaded — background will be black")

hdri_rotation = float(rs.environment_rotation_degrees)
print(f"HDRI rotation: {hdri_rotation}°")

for face_name, face_info in FACES.items():
    print(f"Rendering {face_name}...")

    # 1. Get RGBA splat
    splat_rgba = render_face_rgba(face_info, size)

    # 2. Sample HDRI into this face
    hdri_face = sample_hdri_face(hdri, face_info, size, hdri_rotation)
    hdri_pil  = Image.fromarray((hdri_face * 255).clip(0, 255).astype(np.uint8), "RGB")

    # 3. Composite splat over HDRI
    hdri_pil  = hdri_pil.convert("RGBA")
    composite = Image.alpha_composite(hdri_pil, splat_rgba)

    # 4. Save
    path = str(Path(output_folder) / f"{id_prefix}_{face_name}.png")
    composite.convert("RGB").save(path, "PNG")
    print(f"  saved {path}")

# Cleanup temp
tmp = Path(output_folder) / "_tmp_render.png"
if tmp.exists():
    tmp.unlink()

print("done")
