from pathlib import Path
from io import BytesIO
import base64, json, re
import warnings
import numpy as np
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_fill_holes

src = Path("data/voxel_weights_mean_foldavg_sub9_ses1_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90.html")
base = Path("figures") / "voxel_weights_mean_foldavg_sub9_ses1_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90_multiplane_contour_all_regions"
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
cuts = {"x": [22, -24, -62], "y": [-64, -24, 2, 60], "z": [-42, -20, -4, 56]}
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
    return np.pad(a, ((10, 14), (8, 8)), mode="constant")
def anat(a):
    r = plt.cm.gray(np.clip(a / mx, 0, 1))[..., :3]
    r[~binary_fill_holes(a > 0)] = 1
    return r
base.parent.mkdir(exist_ok=True)
fig = plt.figure(figsize=(14.4, 12), facecolor="white")
outer = fig.add_gridspec(3, 1, hspace=0.05)
rows = [outer[0].subgridspec(1, 3, wspace=0.06), outer[1].subgridspec(1, 4, wspace=0.06), outer[2].subgridspec(1, 4, wspace=0.06)]
for i, mode in enumerate("xyz"):
    for j, cut in enumerate(cuts[mode]):
        ax = fig.add_subplot(rows[i][0, j])
        a, m = pad(plane(bg, mode, cut)), pad(plane(mask, mode, cut))
        ax.imshow(anat(a), interpolation="nearest")
        ax.contour(m.astype(float), levels=[0.5], colors="#dc2626", linewidths=2.2)
        ax.text(0.02, 0.02, f"{mode}={cut}", transform=ax.transAxes, ha="left", va="bottom", fontsize=16, color="black")
        if mode != "x":
            ax.text(0.08, 0.96, "L", transform=ax.transAxes, ha="center", va="top", fontsize=16, color="black")
            ax.text(0.92, 0.96, "R", transform=ax.transAxes, ha="center", va="top", fontsize=16, color="black")
        ax.set_axis_off()
fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
print(f"{base}.png")
print(f"{base}.pdf")
