"""results/ EDA 시각화 재현 — PIL + matplotlib (TensorFlow 불필요)

Covid19 CT 데이터셋(폴더 구조: <root>/{train,test}/{Covid,Normal}/*.jpg|png)을
준비한 뒤 실행:
    python make_figures.py path/to/Covid19_CT_Image_dataset
"""
import os
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
GREEN, RED = "#76B900", "#E15759"

base = sys.argv[1] if len(sys.argv) > 1 else "Covid19_CT_Image_dataset"
R = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(R, exist_ok=True)

def load(folder, n):
    return [Image.open(f).convert("L").resize((128, 128)) for f in sorted(glob.glob(folder + "/*"))[:n]]

cov, nor = load(f"{base}/train/Covid", 4), load(f"{base}/train/Normal", 4)
fig, axs = plt.subplots(2, 4, figsize=(9, 4.8))
for j, im in enumerate(cov): axs[0, j].imshow(im, cmap="gray"); axs[0, j].axis("off")
for j, im in enumerate(nor): axs[1, j].imshow(im, cmap="gray"); axs[1, j].axis("off")
fig.suptitle("폐 CT 샘플 — COVID(상) vs Normal(하)")
fig.tight_layout(); fig.savefig(f"{R}/ct_samples.png", dpi=120); plt.close(fig)

counts = {k: len(glob.glob(f"{base}/{k}/*")) for k in ["train/Covid", "train/Normal", "test/Covid", "test/Normal"]}
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(list(counts), list(counts.values()), color=[GREEN if "Normal" in k else RED for k in counts])
for i, v in enumerate(counts.values()): ax.text(i, v, str(v), ha="center", va="bottom")
ax.set(title="Covid19 CT 데이터셋 구성", ylabel="이미지 수"); plt.xticks(rotation=15)
fig.tight_layout(); fig.savefig(f"{R}/ct_class_distribution.png", dpi=120); plt.close(fig)
print("saved figures to", R)
