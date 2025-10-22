# tools/check_pair.py  —— 免参数版
import os, glob, random, h5py, numpy as np, torch, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def fscore_np(pred, gt, tau=0.02):
    d = torch.cdist(torch.tensor(pred), torch.tensor(gt)).numpy()
    pr = (d.min(axis=1) < tau).mean()
    rc = (d.min(axis=0) < tau).mean()
    return 0.0 if pr+rc==0 else 2*pr*rc/(pr+rc)

def main(root='./data/PCN/train', k=3):
    files = sorted(glob.glob(os.path.join(root, '*.h5')))
    if not files:
        print(f"[ERR] {root} 下没有 .h5；先用 dir2h5_smart 生成。"); return
    picks = random.sample(files, min(k, len(files)))
    for path in picks:
        with h5py.File(path, 'r') as f:
            p, g = f['partial'][:], f['gt'][:]
        f1 = fscore_np(p, g, 0.02)
        print(f"[{os.path.basename(path)}] Partial-vs-GT F1@0.02 = {f1:.3f}")
        fig = plt.figure(figsize=(9,3))
        for i,(pts,title) in enumerate([(p,'Partial'),(g,'GT')]):
            ax = fig.add_subplot(1,2,i+1, projection='3d')
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
            ax.set_title(title); ax.set_axis_off(); ax.view_init(20,45)
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
