
import torch

def chamfer_L2(pred, gt):
    dists = torch.cdist(pred, gt, p=2)  # (B,N,M)
    min_pred_gt = dists.min(dim=2).values.mean(dim=1)
    min_gt_pred = dists.min(dim=1).values.mean(dim=1)
    return (min_pred_gt + min_gt_pred).mean()

def density_aware_cd(pred, gt, k=8, eps=1e-6):
    """
    使用近邻距离近似密度：越稀疏权重越大
    """
    d = torch.cdist(gt, gt, p=2)  # (B,M,M)
    knn = d.topk(k+1, largest=False).values[:, :, 1:]  # (B,M,k) 跳过自己
    density = knn.mean(dim=2)  # (B,M)
    w_gt = 1.0 / (density + eps)
    w_gt = w_gt / (w_gt.sum(dim=1, keepdim=True) + eps)  # 归一化
    # pred->gt
    d_pg = torch.cdist(pred, gt, p=2)  # (B,N,M)
    min_d_pg, idx_pg = d_pg.min(dim=2)  # (B,N)
    # 按落到的 gt 点的权重来加权
    w_assigned = w_gt.gather(1, idx_pg)  # (B,N)
    term1 = (w_assigned * min_d_pg).sum(dim=1)
    # gt->pred（保持与普通CD一致的对称性）
    min_d_gp = d_pg.min(dim=1).values  # (B,M)
    term2 = (w_gt * min_d_gp).sum(dim=1)
    return (term1 + term2).mean()
