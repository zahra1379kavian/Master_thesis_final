from pathlib import Path
from io import BytesIO
import base64, json, re
import warnings
import numpy as np
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.text import Text
import nibabel as nib
from nilearn import datasets, image
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from motor_overlap_overlay import motor_overlap_masks

src = Path("data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90.html")
base = Path("figures") / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90_multiplane_contour_all_regions"
plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
atlas_cache_dir = Path("/home/zkavian/nilearn_data")
atlas_version = "3v2"
region_order = (
    "Sensorimotor cortex",
    "Frontal cortex",
    "Parietal cortex",
    "Temporal cortex",
    "Occipital cortex",
    "Cerebellum",
    "Limbic/subcortical",
)
region_colors = {
    "Sensorimotor cortex": "#0066CC",
    "Frontal cortex": "#CC0000",
    "Parietal cortex": "#008000",
    "Temporal cortex": "#8B4513",
    "Occipital cortex": "#D4A017",
    "Cerebellum": "#800080",
    "Limbic/subcortical": "#008C8C",
}
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
def crop(a, m, label=None, *extra_masks):
    b = binary_fill_holes(a > 0) | m
    for extra_mask in extra_masks:
        b |= extra_mask
    y, x = np.where(b)
    y0, y1, x0, x1 = max(y.min() - 4, 0), min(y.max() + 5, a.shape[0]), max(x.min() - 4, 0), min(x.max() + 5, a.shape[1])
    cropped = [pad(a[y0:y1, x0:x1]), pad(m[y0:y1, x0:x1])]
    if label is not None:
        cropped.append(pad(label[y0:y1, x0:x1]))
    cropped.extend(pad(extra_mask[y0:y1, x0:x1]) for extra_mask in extra_masks)
    return tuple(cropped)
def anat(a):
    brain = binary_fill_holes(a > 0)
    r = plt.cm.gray(np.clip(a / mx, 0, 1))
    r[~brain, 3] = 0
    return r
def atlas_category(label_name):
    name = re.sub(r"_(L|R)$", "", label_name)
    if name.startswith(("Precentral", "Postcentral", "Supp_Motor_Area", "Paracentral_Lobule", "Rolandic_Oper")):
        return "Sensorimotor cortex"
    if name.startswith(("Frontal", "OFC")) or name in {"Rectus", "Olfactory"}:
        return "Frontal cortex"
    if name.startswith("Parietal") or name in {"Angular", "SupraMarginal", "Precuneus"}:
        return "Parietal cortex"
    if name.startswith("Temporal") or name in {"Heschl", "Fusiform"}:
        return "Temporal cortex"
    if name.startswith("Occipital") or name in {"Calcarine", "Cuneus", "Lingual"}:
        return "Occipital cortex"
    if name.startswith(("Cerebellum", "Vermis")):
        return "Cerebellum"
    if name.startswith(("Cingulate", "ACC", "Thal", "SN")) or name in {
        "Insula",
        "Hippocampus",
        "ParaHippocampal",
        "Amygdala",
        "Caudate",
        "Putamen",
        "Pallidum",
        "N_Acc",
        "VTA",
        "Red_N",
        "LC",
        "Raphe_D",
        "Raphe_M",
    }:
        return "Limbic/subcortical"
    return "Unassigned"
def resample_aal_to_mask():
    reference_img = nib.Nifti1Image(np.zeros(mask.shape, dtype=np.uint8), aff)
    atlas = datasets.fetch_atlas_aal(version=atlas_version, data_dir=str(atlas_cache_dir), verbose=0)
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    if atlas_img.shape[:3] == reference_img.shape[:3] and np.allclose(atlas_img.affine, reference_img.affine):
        atlas_data = np.rint(atlas_img.get_fdata()).astype(np.int32, copy=False)
    else:
        atlas_data = np.rint(
            image.resample_to_img(
                atlas_img,
                reference_img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            ).get_fdata()
        ).astype(np.int32, copy=False)
    return atlas_data, atlas
def region_label_data():
    atlas_data, atlas = resample_aal_to_mask()
    category_ids = {name: idx + 1 for idx, name in enumerate(region_order)}
    atlas_labels = np.zeros(mask.shape, dtype=np.int16)
    for label_value, label_name in zip(atlas.indices, atlas.labels):
        label_value = int(label_value)
        if label_value == 0:
            continue
        category = atlas_category(str(label_name))
        atlas_labels[atlas_data == label_value] = category_ids[category]
    labels = np.where(display_mask, atlas_labels, 0).astype(np.int16, copy=False)
    outside_atlas = display_mask & (labels == 0)
    if np.any(outside_atlas):
        nearest = distance_transform_edt(atlas_labels == 0, return_distances=False, return_indices=True)
        labels[outside_atlas] = atlas_labels[tuple(axis_indices[outside_atlas] for axis_indices in nearest)]
    return labels
motor_overlap_display, _ = motor_overlap_masks(mask, aff)
display_mask = mask | motor_overlap_display
region_labels = region_label_data()
region_counts = {
    name: int(np.count_nonzero(region_labels == idx + 1))
    for idx, name in enumerate(region_order)
}
def overlay(ax, labels, m):
    rgba = np.zeros(labels.shape + (4,), dtype=float)
    for idx, name in enumerate(region_order, start=1):
        hit = labels == idx
        if not np.any(hit):
            continue
        rgba[hit, :3] = to_rgb(region_colors[name])
        rgba[hit, 3] = 0.80
    ax.imshow(rgba, interpolation="nearest")
    if np.any(m):
        ax.contour(m.astype(float), levels=[0.5], colors="#111111", linewidths=0.55, alpha=0.95)

def bold_figure_text(fig):
    for text in fig.findobj(match=Text):
        text.set_fontweight("bold")

base.parent.mkdir(exist_ok=True)
n_cols = max(len(cuts[mode]) for mode, _ in plane_specs)
fig, axes = plt.subplots(len(plane_specs), n_cols, figsize=(15.1, 6.4), facecolor="white")
for row, (mode, plane_label) in enumerate(plane_specs):
    for col in range(n_cols):
        ax = axes[row, col]
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)
        if col >= len(cuts[mode]):
            ax.set_axis_off()
            continue
        cut = cuts[mode][col]
        a, m, label_slice = crop(
            plane(bg, mode, cut),
            plane(display_mask, mode, cut),
            plane(region_labels, mode, cut),
        )
        ax.imshow(anat(a), interpolation="nearest")
        overlay(ax, label_slice, m)
        height, width = a.shape[:2]
        brain_y, brain_x = np.where(binary_fill_holes(a > 0))
        label_x = (brain_x.min() + brain_x.max()) / 2
        label_y = brain_y.max() + 3
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(label_y + 12, brain_y.min() - 12 if row == 0 else brain_y.min() - 2)
        ax.text(label_x, label_y, f"{mode} = {cut:g}", ha="center", va="top", fontsize=17, fontweight="bold", color="0.2")
        if row == 0:
            orient_y = brain_y.min() + 1
            ax.text(brain_x.min() + 2, orient_y, "L", ha="center", va="bottom", fontsize=17, fontweight="bold", color="0.15")
            ax.text(brain_x.max() - 2, orient_y, "R", ha="center", va="bottom", fontsize=17, fontweight="bold", color="0.15")
        ax.set_axis_off()
legend = [
    Patch(facecolor=region_colors[name], edgecolor="0.2", alpha=0.90, label=name)
    for name in region_order
    if region_counts[name] > 0
]
fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, prop={"size": 15, "weight": "bold"}, bbox_to_anchor=(0.5, 0.055))
bold_figure_text(fig)
fig.subplots_adjust(left=0.015, right=0.995, bottom=0.20, top=0.985, wspace=0.015, hspace=0.06)
fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
print(f"{base}.png")
print(f"{base}.pdf")
