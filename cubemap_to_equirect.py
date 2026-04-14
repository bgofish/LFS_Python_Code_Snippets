"""
Cubemap to Equirectangular converter
-------------------------------------
Reads 6 cubemap face PNGs and stitches them into a single
equirectangular (lat/long) panorama image.

Input files expected:
  {prefix}_Front.png
  {prefix}_Back.png
  {prefix}_Right.png
  {prefix}_Left.png
  {prefix}_Top.png
  {prefix}_Bottom.png

Output:
  {prefix}_Equirectangular.png
"""

import numpy as np
from pathlib import Path
from PIL import Image

# ── Configure these ───────────────────────────────────────
input_folder  = r"B:\[LFS]\CubeMap-TEST"
output_folder = r"B:\[LFS]\CubeMap-TEST"
id_prefix     = "20000-2p49M"       # filename prefix before _Front etc
output_width  = 4096                # equirectangular width (height = width/2)
# ─────────────────────────────────────────────────────────

output_height = output_width // 2

# ── Load cubemap faces ────────────────────────────────────
face_names = ["Front", "Back", "Right", "Left", "Top", "Bottom"]
faces = {}
for name in face_names:
    path = Path(input_folder) / f"{id_prefix}_{name}.png"
    img  = Image.open(path).convert("RGB")
    faces[name] = np.array(img, dtype=np.float32) / 255.0
    print(f"loaded {path.name}  {img.size}")

face_size = faces["Front"].shape[0]

# ── Face coordinate systems ───────────────────────────────
# For each face: given a ray (x,y,z), the face defines how
# to map to 2D face UV coordinates.
# Convention: u = right axis, v = up axis (both -1..1)

def ray_to_face_uv(rx, ry, rz):
    """
    Given a unit ray (rx, ry, rz), return (face_name, u, v)
    where u,v are in -1..1 on the selected face.
    """
    ax, ay, az = np.abs(rx), np.abs(ry), np.abs(rz)
    max_a = np.maximum(np.maximum(ax, ay), az)

    # Determine dominant axis for each pixel
    # Returns face name, u, v arrays

    # +Z Front
    mask_pz = (az == max_a) & (rz > 0)
    # -Z Back
    mask_nz = (az == max_a) & (rz <= 0)
    # +X Right
    mask_px = (ax == max_a) & (rx > 0) & ~mask_pz & ~mask_nz
    # -X Left
    mask_nx = (ax == max_a) & (rx <= 0) & ~mask_pz & ~mask_nz
    # +Y Top
    mask_py = (ay == max_a) & (ry > 0) & ~mask_pz & ~mask_nz & ~mask_px & ~mask_nx
    # -Y Bottom
    mask_ny = ~mask_pz & ~mask_nz & ~mask_px & ~mask_nx & ~mask_py

    u = np.zeros_like(rx)
    v = np.zeros_like(rx)
    face_idx = np.zeros(rx.shape, dtype=np.int32)

    # Front +Z:  right=+X, up=+Y
    u[mask_pz]  = -rx[mask_pz] / az[mask_pz]
    v[mask_pz]  =  ry[mask_pz] / az[mask_pz]
    face_idx[mask_pz] = 0

    # Back -Z:  right=-X, up=+Y
    u[mask_nz]  =  rx[mask_nz] / az[mask_nz]
    v[mask_nz]  =  ry[mask_nz] / az[mask_nz]
    face_idx[mask_nz] = 1

    # Right +X:  right=-Z, up=+Y
    u[mask_px]  = -rz[mask_px] / ax[mask_px]
    v[mask_px]  =  ry[mask_px] / ax[mask_px]
    face_idx[mask_px] = 2

    # Left -X:  right=+Z, up=+Y
    u[mask_nx]  =  rz[mask_nx] / ax[mask_nx]
    v[mask_nx]  =  ry[mask_nx] / ax[mask_nx]
    face_idx[mask_nx] = 3

    # Top +Y:  right=+X, up=-Z
    u[mask_py]  =  rx[mask_py] / ay[mask_py]
    v[mask_py]  = -rz[mask_py] / ay[mask_py]
    face_idx[mask_py] = 4

    # Bottom -Y:  right=+X, up=+Z
    u[mask_ny]  =  rx[mask_ny] / ay[mask_ny]
    v[mask_ny]  =  rz[mask_ny] / ay[mask_ny]
    face_idx[mask_ny] = 5

    return face_idx, u, v


# ── Build equirectangular ─────────────────────────────────

print(f"Building {output_width}x{output_height} equirectangular...")

# For each output pixel, compute the ray direction
lon = (np.arange(output_width)  + 0.5) / output_width  * 2 * np.pi - np.pi  # -π..π
lat = (np.arange(output_height) + 0.5) / output_height * np.pi - np.pi / 2  # -π/2..π/2

LON, LAT = np.meshgrid(lon, lat)

# Spherical to cartesian — matching our cubemap convention
# +Z = front, +X = right, +Y = up
rx =  np.cos(LAT) * np.sin(LON)
ry =  np.sin(LAT)
rz =  np.cos(LAT) * np.cos(LON)

face_idx, u, v = ray_to_face_uv(rx, ry, rz)

# Map u,v (-1..1) to pixel coords (0..face_size-1)
px = np.clip(((u + 1.0) * 0.5 * face_size), 0, face_size - 1)
py = np.clip(((1.0 - (v + 1.0) * 0.5) * face_size), 0, face_size - 1)

# Bilinear sample from each face
face_order = ["Front", "Back", "Right", "Left", "Top", "Bottom"]
face_arrays = np.stack([faces[n] for n in face_order], axis=0)
# face_arrays shape: (6, H, W, 3)

x0 = np.floor(px).astype(int).clip(0, face_size - 1)
y0 = np.floor(py).astype(int).clip(0, face_size - 1)
x1 = (x0 + 1).clip(0, face_size - 1)
y1 = (y0 + 1).clip(0, face_size - 1)
fx = px - np.floor(px)
fy = py - np.floor(py)

fi = face_idx  # (H, W)
c00 = face_arrays[fi, y0, x0]
c01 = face_arrays[fi, y0, x1]
c10 = face_arrays[fi, y1, x0]
c11 = face_arrays[fi, y1, x1]

result = (c00 * (1 - fx)[..., None] * (1 - fy)[..., None] +
          c01 * fx[..., None]       * (1 - fy)[..., None] +
          c10 * (1 - fx)[..., None] * fy[..., None] +
          c11 * fx[..., None]       * fy[..., None])

# Rotate 180° by rolling horizontally by half the width
result = np.roll(result, output_width // 2, axis=1)

# ── Save ─────────────────────────────────────────────────
out_img = Image.fromarray((result * 255).clip(0, 255).astype(np.uint8), "RGB")
out_path = str(Path(output_folder) / f"{id_prefix}_Equirectangular.png")
out_img.save(out_path)
print(f"saved {out_path}")
print("done")
