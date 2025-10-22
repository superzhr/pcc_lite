
# pcc_lite/data_tools/ply2h5.py  (robust, supports .pcd/.ply/.xyz/.pts)
import os, h5py, argparse, numpy as np, open3d as o3d

EXTS = {'.ply', '.pcd', '.xyz', '.pts'}

def load_pc(p):
    pc = o3d.io.read_point_cloud(p)
    pts = np.asarray(pc.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"{p} 不是 Nx3 点云")
    return pts

def walk_map(root):
    # Return dict: key=relative path without ext, val=absolute path
    out = {}
    for dp,_,fs in os.walk(root):
        for f in fs:
            ext = os.path.splitext(f)[1].lower()
            if ext in EXTS:
                full = os.path.join(dp, f)
                rel_noext = os.path.splitext(os.path.relpath(full, root))[0].replace("\\","/")
                out[rel_noext] = full
    return out

def convert_split(src_root, split, dst_root, partial_name, gt_name):
    split_root = os.path.join(src_root, split)
    out_split = 'val' if split.lower() in ['valid', 'validation'] else split
    out_dir = os.path.join(dst_root, out_split)
    os.makedirs(out_dir, exist_ok=True)

    pdir = os.path.join(split_root, partial_name)
    gdir = os.path.join(split_root, gt_name)
    if not (os.path.isdir(pdir) and os.path.isdir(gdir)):
        raise FileNotFoundError(f"未找到目录: {pdir} 或 {gdir}")

    pmap = walk_map(pdir)
    gmap = walk_map(gdir)
    keys = sorted(set(pmap.keys()) & set(gmap.keys()))
    if not keys:
        # fallback: match by filename only (ignore subfolders)
        p_by_name = {}
        for k, p in pmap.items():
            name = os.path.basename(k)
            p_by_name.setdefault(name, []).append(p)
        g_by_name = {}
        for k, g in gmap.items():
            name = os.path.basename(k)
            g_by_name.setdefault(name, []).append(g)
        keys_names = sorted(set(p_by_name.keys()) & set(g_by_name.keys()))
        if not keys_names:
            raise RuntimeError(f"未配对到任何文件，请确认 {pdir} 与 {gdir} 下的相对路径/文件名是一一对应的")
        pairs = []
        for name in keys_names:
            for i in range(min(len(p_by_name[name]), len(g_by_name[name]))):
                pairs.append((p_by_name[name][i], g_by_name[name][i]))
    else:
        pairs = [(pmap[k], gmap[k]) for k in keys]

    for i, (pp, gp) in enumerate(pairs):
        dst = os.path.join(out_dir, f"{out_split}_{i:06d}.h5")
        with h5py.File(dst, "w") as f:
            f.create_dataset('partial', data=load_pc(pp))
            f.create_dataset('gt', data=load_pc(gp))
    print(f"[OK] {split}->{out_split}: 写入 {len(pairs)} 个样本到 {out_dir}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', required=True, help='PCN 根目录（含 train/ valid/ test/ 等）')
    ap.add_argument('--dst_root', default='./data/PCN', help='输出到 pcc-lite 期望目录')
    ap.add_argument('--splits', default='train,valid,test', help='要转换的划分，逗号分隔')
    ap.add_argument('--partial_name', default='partial', help='不完整点云文件夹名（默认 partial）')
    ap.add_argument('--gt_name', default='complete', help='完整点云文件夹名（默认 complete；有的叫 gt）')
    args = ap.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)
    for sp in [s.strip() for s in args.splits.split(',') if s.strip()]:
        convert_split(args.src_root, sp, args.dst_root, args.partial_name, args.gt_name)
