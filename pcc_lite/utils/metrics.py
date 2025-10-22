
import torch
def fscore(pred, gt, tau=0.02):
    d = torch.cdist(pred, gt, p=2)  # (B,N,M)
    pr = (d.min(dim=2).values < tau).float().mean(dim=1)   # precision
    rc = (d.min(dim=1).values < tau).float().mean(dim=1)   # recall
    f = 2 * pr * rc / (pr + rc + 1e-8)
    return f.mean()
