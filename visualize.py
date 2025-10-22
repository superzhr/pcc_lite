
import yaml, torch, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch.utils.data import DataLoader
from pcc_lite.datasets.pcn import PCNDataset
from pcc_lite.models.seedformer_lite import SeedFormerLite

cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PCNDataset(cfg["data_root"], "test")
model = SeedFormerLite(k_seeds=cfg["k_seeds"], up_ratio=cfg["up_ratio"]).to(device)
model.load_state_dict(torch.load("checkpoints/seedformer_lite_best.pth", map_location=device))
model.eval()

idx = random.randrange(len(dataset))
partial, gt = dataset[idx]
with torch.no_grad():
    _, fine = model(partial.unsqueeze(0).to(device))
fine = fine.squeeze(0).cpu().numpy()
partial = partial.cpu().numpy()
gt = gt.cpu().numpy()

fig = plt.figure(figsize=(12,4))
for i,(pts,title) in enumerate([(partial,"Partial"), (fine,"Prediction"), (gt,"Ground Truth")]):
    ax = fig.add_subplot(1,3,i+1, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
    ax.set_title(title); ax.set_axis_off(); ax.view_init(elev=20, azim=45)
plt.tight_layout(); plt.show()
