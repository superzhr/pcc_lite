
import h5py, glob, torch, os
from torch.utils.data import Dataset

def normalize_unit_sphere(partial, gt):
    """
    以 GT 的中心与半径做归一化：
    - 平移到中心为 0
    - 缩放到最大半径为 1
    返回同尺度的 (partial, gt), 以及缩放参数（可选）
    """
    center = gt.mean(dim=0, keepdim=True)  # (1,3)
    gt0 = gt - center
    pr0 = partial - center
    radius = gt0.norm(dim=1).max().clamp(min=1e-6)  # 标量
    gt1 = gt0 / radius
    pr1 = pr0 / radius
    return pr1, gt1

class PCNDataset(Dataset):
    def __init__(self, root, split='train', n_partial=1024, n_gt=4096, normalize=True):
        self.files = sorted(glob.glob(os.path.join(root, split, '*.h5')))
        self.np = n_partial; self.ng = n_gt; self.normalize = normalize
        if len(self.files) == 0:
            print(f"[WARN] No .h5 files found in {os.path.join(root, split)}")

    def __len__(self): return len(self.files)

    def _rand_sample(self, x, n):
        if x.shape[0] >= n:
            idx = torch.randperm(x.shape[0])[:n]
        else:
            rep = (n + x.shape[0] - 1) // x.shape[0]
            idx = torch.randperm(x.shape[0]*rep)[:n] % x.shape[0]
        return x[idx]

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as f:
            partial = torch.tensor(f['partial'][:], dtype=torch.float32) # (N,3)
            gt      = torch.tensor(f['gt'][:], dtype=torch.float32)      # (M,3)
        partial = self._rand_sample(partial, self.np)
        gt = self._rand_sample(gt, self.ng)
        if self.normalize:
            partial, gt = normalize_unit_sphere(partial, gt)
        return partial, gt
