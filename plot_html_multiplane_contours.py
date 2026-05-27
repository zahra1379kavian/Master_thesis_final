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
def coord(mode, k):
    i = "xyz".index(mode)
    return int(round(aff[i, i] * (k + 1) + aff[i, 3]))
def pick(mode, n=10, gap=3):
    i = "xyz".index(mode)
    c = mask.sum(tuple(j for j in range(3) if j != i))
    out = []
    for k in np.argsort(c)[::-1]:
        if c[k] and all(abs(k - j) >= gap for j in out):
            out.append(int(k))
        if len(out) == n:
            break
    return [coord(mode, k) for k in sorted(out)]
cuts = {mode: pick(mode) for mode in "xyz"}
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
base.parent.mkdir(exist_ok=True)
fig, axes = plt.subplots(6, 5, figsize=(12.6, 14.8), facecolor="white")
items = [(mode, cut) for mode in "xyz" for cut in cuts[mode]]
for i, (ax, (mode, cut)) in enumerate(zip(axes.ravel(), items)):
    a, m = crop(plane(bg, mode, cut), plane(mask, mode, cut))
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    ax.imshow(anat(a), interpolation="nearest")
    ax.contour(m.astype(float), levels=[0.5], colors="#dc2626", linewidths=1.8)
    ax.set_axis_off()
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, wspace=-0.04, hspace=0.01)
for ax in axes.ravel()[:5]:
    box = ax.get_position()
    fig.text(box.x0 + box.width * 0.22, box.y1 - box.height * 0.04, "L", ha="center", va="top", fontsize=11, color="black")
    fig.text(box.x0 + box.width * 0.78, box.y1 - box.height * 0.04, "R", ha="center", va="top", fontsize=11, color="black")
fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
print(f"{base}.png")
print(f"{base}.pdf")
