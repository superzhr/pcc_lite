import yaml, torch
from torch.utils.data import DataLoader
from pcc_lite.datasets.pcn import PCNDataset          # 我们自定义的数据集类
from pcc_lite.models.seedformer_lite import SeedFormerLite  # 轻量模型
from pcc_lite.losses.chamfer import chamfer_L2        # 纯 PyTorch 的 Chamfer 距离
from pcc_lite.utils.metrics import fscore             # F-score 指标

# 读取配置文件（字典），常见键：data_root / batch_size / k_seeds / up_ratio 等
cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

# 自动选设备：有 CUDA 用 GPU，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 构建测试集。PCNDataset 会去 cfg["data_root"]/test/*.h5 读取样本，
# 每个样本包含 partial(N,3) 和 gt(M,3)。类里会采样到固定点数（如 2048/8192），
# 也可能做归一化（单位球），这取决于你当前的数据集实现版本。
test_set = PCNDataset(cfg["data_root"], "test")

# DataLoader 负责“打包成 batch、并行加载（num_workers）、按顺序迭代”
# 这里 shuffle=False 保证评测可复现
loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

# 实例化模型。k_seeds 是“粗点/查询”的数量，up_ratio 是上采样倍率。
# 模型会返回 (coarse, fine) 两个点云。
model = SeedFormerLite(k_seeds=cfg["k_seeds"], up_ratio=cfg["up_ratio"]).to(device)

# 加载训练好的最好权重
ckpt = "checkpoints/seedformer_lite_best.pth"
model.load_state_dict(torch.load(ckpt, map_location=device))

# 切 eval 模式：关掉 dropout、让 LayerNorm/BatchNorm 用推理行为
model.eval()

# 评测累计器：cd_meter、f_meter 分别累加 Chamfer 与 F-score
cd_meter, f_meter, n = 0.0, 0.0, 0

# no_grad：前向不保留梯度，显存更省、速度更快
with torch.no_grad():
    for partial, gt in loader:
        # 把 batch 张量搬到同一设备上（GPU/CPU）
        partial, gt = partial.to(device), gt.to(device)

        # 前向：返回 (coarse, fine)，我们用细化后的 fine 做评测
        _, fine = model(partial)

        # 计算指标并累加：
        #   chamfer_L2：对称的 L2 CD（pred->gt 与 gt->pred 求均）
        #   fscore    ：F1@tau（默认 tau=0.01；若做了单位球归一化，建议 0.02）
        cd_meter += chamfer_L2(fine, gt).item()
        f_meter += fscore(fine, gt).item()
        n += 1

# 求平均并打印
print(f"[TEST] CD={cd_meter/max(n,1):.6f}, F1={f_meter/max(n,1):.4f}")
