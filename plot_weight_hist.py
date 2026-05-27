from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

src = Path("data/voxel_weights_mean_foldavg_sub9_ses1_task1_bold1_beta0.75_smooth1.8_gamma1.5.nii.gz")
base = Path("figures") / "voxel_weights_mean_foldavg_sub9_ses1_task1_bold1_beta0.75_smooth1.8_gamma1.5_hist_p90"
png = Path(f"{base}.png")
pdf = Path(f"{base}.pdf")
w = np.asarray(nib.load(src).get_fdata(), float)
w = w[np.isfinite(w) & (w != 0)]
thr = np.percentile(w, 90)
base.parent.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(w, bins=80, density=True, color="#2563eb", alpha=0.75)
ax.axvline(thr, color="#dc2626", lw=2, label=f"90th percentile = ${thr / 1e-4:.2f}\\times10^{{-4}}$")
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((-4, -4))
ax.xaxis.set_major_formatter(fmt)
ax.set(title="Voxel weight probability histogram", xlabel="Weight", ylabel="Probability density")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(png, dpi=200)
fig.savefig(pdf)
print(f"threshold={thr:.12g}")
print(png)
print(pdf)
