import os, yaml, math, time, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pcc_lite.datasets.pcn import PCNDataset
from pcc_lite.models.seedformer_lite import SeedFormerLite
from pcc_lite.losses.chamfer import chamfer_L2, density_aware_cd
from pcc_lite.utils.metrics import fscore


def _int(v, d):    # 小工具：从 cfg 里取数值，取不到就用默认
    try: return int(v)
    except: return d

def _float(v, d):
    try: return float(v)
    except: return d


def main():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

    # —— 基础配置（带默认值，防止类型不对）——
    epochs      = _int(cfg.get("epochs", 80), 80)
    batch_size  = _int(cfg.get("batch_size", 4), 4)
    lr          = _float(cfg.get("lr", 1e-3), 1e-3)
    k_seeds     = _int(cfg.get("k_seeds", 512), 512)
    up_ratio    = _int(cfg.get("up_ratio", 4), 4)

    warmup      = _int(cfg.get("warmup_coarse_epochs", 0), 0)
    use_dacd    = bool(cfg.get("use_density_cd", False))
    tau         = _float(cfg.get("fscore_tau", 0.02), 0.02)

    # —— 进阶提速相关（可选项）——
    npt_train   = _int(cfg.get("n_partial_train", 1024), 1024)
    ngt_train   = _int(cfg.get("n_gt_train", 2048), 2048)
    npt_val     = _int(cfg.get("n_partial_val", 1024), 1024)
    ngt_val     = _int(cfg.get("n_gt_val", 4096), 4096)

    num_workers = _int(cfg.get("num_workers", 4), 4)
    val_every   = _int(cfg.get("val_every", 5), 5)        # 每多少个 epoch 验证一次
    grad_accum  = _int(cfg.get("grad_accum", 1), 1)       # 梯度累积步数
    amp_enabled = bool(cfg.get("amp", True))              # 混合精度
    weight_decay = _float(cfg.get("weight_decay", 1e-4), 1e-4)

    data_root = cfg.get("data_root", "./data/PCN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # TF32（Ampere+ 显卡对大矩阵乘法有提速）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # —— 数据集：若旧版数据类不支持 n_partial/n_gt，这里做兼容 —— 
    try:
        train_set = PCNDataset(data_root, "train", n_partial=npt_train, n_gt=ngt_train)
        val_set   = PCNDataset(data_root, "val",   n_partial=npt_val,   n_gt=ngt_val)
    except TypeError:
        print("[WARN] 当前 PCNDataset 不支持 n_partial/n_gt，使用默认点数。")
        train_set = PCNDataset(data_root, "train")
        val_set   = PCNDataset(data_root, "val")

    # DataLoader（Windows 下 persistent_workers 需要 num_workers>0）
    if num_workers > 0:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=max(1, num_workers // 2), pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    # —— 模型/优化器/调度器 —— 
    model = SeedFormerLite(k_seeds=k_seeds, up_ratio=up_ratio).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))

    best_cd = math.inf
    for epoch in range(epochs):
        model.train()
        start_ep = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        opt.zero_grad(set_to_none=True)
        for it, (partial, gt) in enumerate(pbar, start=1):
            partial, gt = partial.to(device, non_blocking=True), gt.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(amp_enabled and device.type == "cuda")):
                coarse, fine = model(partial)
                loss_c = chamfer_L2(coarse, gt)
                loss_f = chamfer_L2(fine, gt)
                if use_dacd:
                    loss_f = loss_f + 0.2 * density_aware_cd(fine, gt)
                loss = (1.0 if epoch < warmup else 0.5) * loss_c + loss_f
                # 梯度累积
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            if it % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=float(loss) * grad_accum)

        scheduler.step()
        dur = time.time() - start_ep
        print(f"[INFO] epoch {epoch+1} finish in {dur/60:.1f} min")

        # —— 验证（降频），最后一个 epoch 也要验证 —— 
        if ((epoch + 1) % val_every != 0) and (epoch + 1 < epochs):
            continue

        model.eval()
        cd_meter = 0.0
        f_meter = 0.0
        n = 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(amp_enabled and device.type == "cuda")):
            for partial, gt in val_loader:
                partial, gt = partial.to(device, non_blocking=True), gt.to(device, non_blocking=True)
                _, fine = model(partial)
                cd = chamfer_L2(fine, gt).item()
                fs = fscore(fine, gt, tau=tau).item()
                cd_meter += cd
                f_meter += fs
                n += 1

        cd_meter /= max(n, 1)
        f_meter  /= max(n, 1)
        print(f"[VAL] CD={cd_meter:.6f}, F1@{tau}={f_meter:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        if cd_meter < best_cd:
            best_cd = cd_meter
            torch.save(model.state_dict(), "checkpoints/seedformer_lite_best.pth")
            print(f"[CKPT] Saved with CD={best_cd:.6f}")


if __name__ == "__main__":
    main()
