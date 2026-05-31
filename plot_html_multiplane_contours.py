from pathlib import Path
from io import BytesIO
import base64, json, re
import warnings
import numpy as np
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from scipy.ndimage import binary_fill_holes

src = Path("data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90.html")
base = Path("figures") / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90_multiplane_contour_all_regions"
overlay_fill = "#0072b2"
overlay_edge = "#004c73"
s = src.read_text()
imgs = [np.asarray(Image.open(BytesIO(base64.b64decode(x))).convert("RGBA")) for x in re.findall(r'src="data:image/png;base64,([^"]+)"', s)]
cfg = json.loads(re.search(r"brainsprite\((\{.*?\})\);", s).group(1))
nx, ny, nz = [cfg["nbSlice"][k] for k in "XYZ"]
aff = np.array(cfg["affine"])
def volume(a):
    v = np.zeros((nx, ny, nz) + a.shape[2:], a.dtype)
    for x in range(nx):
        c, r = x % (a.shape[1] // ny), x // (a.shape[1] // ny)
        v[x] = a[r * nz:(r + 1) * nz, c * ny:(c + 1) * ny][::-1].transpose(1, 0, 2)
    return v
bg = volume(imgs[0])[..., :3].mean(-1).astype(float)
mask = volume(imgs[2])[..., 3] > 0
mx = np.percentile(bg[bg > 0], 99.5)
def coord(mode, k):
    i = "xyz".index(mode)
    return int(round(aff[i, i] * (k + 1) + aff[i, 3]))
def pick(mode, n=8, gap=4):
    i = "xyz".index(mode)
    c = mask.sum(tuple(j for j in range(3) if j != i))
    out = []
    for k in np.argsort(c)[::-1]:
        if c[k] and all(abs(k - j) >= gap for j in out):
            out.append(int(k))
        if len(out) == n:
            break
    return [coord(mode, k) for k in sorted(out)]
plane_specs = [("x", "Sagittal"), ("y", "Coronal"), ("z", "Axial")]
cuts = {mode: pick(mode) for mode, _ in plane_specs}
def index(mode, cut):
    i = "xyz".index(mode)
    return int(round((cut - aff[i, 3]) / aff[i, i] - 1))
def plane(v, mode, cut):
    k = index(mode, cut)
    if mode == "x":
        return v[k].T[::-1]
    if mode == "y":
        return v[:, k, :].T[::-1]
    return v[:, :, k].T[::-1]
def pad(a):
    return np.pad(a, ((3, 5), (2, 2)), mode="constant")
def crop(a, m):
    b = binary_fill_holes(a > 0) | m
    y, x = np.where(b)
    y0, y1, x0, x1 = max(y.min() - 4, 0), min(y.max() + 5, a.shape[0]), max(x.min() - 4, 0), min(x.max() + 5, a.shape[1])
    return pad(a[y0:y1, x0:x1]), pad(m[y0:y1, x0:x1])
def anat(a):
    brain = binary_fill_holes(a > 0)
    r = plt.cm.gray(np.clip(a / mx, 0, 1))
    r[~brain, 3] = 0
    return r
def overlay(ax, m):
    if np.any(m):
        ax.contourf(m.astype(float), levels=[0.5, 1.5], colors=[overlay_fill], alpha=0.46, antialiased=True)
        ax.contour(m.astype(float), levels=[0.5], colors=overlay_edge, linewidths=0.95)
base.parent.mkdir(exist_ok=True)
n_cols = max(len(cuts[mode]) for mode, _ in plane_specs)
fig, axes = plt.subplots(len(plane_specs), n_cols, figsize=(15.1, 7.0), facecolor="white")
for row, (mode, label) in enumerate(plane_specs):
    for col in range(n_cols):
        ax = axes[row, col]
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)
        if col >= len(cuts[mode]):
            ax.set_axis_off()
            continue
        cut = cuts[mode][col]
        a, m = crop(plane(bg, mode, cut), plane(mask, mode, cut))
        ax.imshow(anat(a), interpolation="nearest")
        overlay(ax, m)
        ax.text(0.5, -0.055, f"{mode} = {cut:g}", transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color="0.2")
        if col == 0:
            ax.text(-0.10, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=8.8, weight="bold", color="0.1")
        if mode in {"y", "z"}:
            ax.text(0.15, 0.99, "L", transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color="0.15")
            ax.text(0.85, 0.99, "R", transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color="0.15")
        ax.set_axis_off()
legend = [Patch(facecolor=overlay_fill, edgecolor=overlay_edge, alpha=0.52, label="vigour-network")]
fig.legend(handles=legend, loc="lower center", frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.012))
fig.subplots_adjust(left=0.055, right=0.995, bottom=0.11, top=0.985, wspace=0.015, hspace=0.20)
fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
print(f"{base}.png")
print(f"{base}.pdf")
