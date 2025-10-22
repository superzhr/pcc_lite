# pcc_lite/data_tools/list2h5_fast.py
import os, argparse, h5py, numpy as np, open3d as o3d
from collections import defaultdict

EXTS = {".pcd", ".ply", ".xyz", ".pts"}
GT_DIRS = ["complete", "gt", "groundtruth"]
PT_DIRS = ["partial", "incomplete"]
SUF = ["", "_partial", "-partial", "_complete", "-complete"]

def load_pc(p):
    pc = o3d.io.read_point_cloud(p)
    pts = np.asarray(pc.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"{p} 不是 Nx3 点云")
    return pts

def canon_key(relpath):
    rel = relpath.replace("\\","/").lstrip("./")
    rel = os.path.splitext(rel)[0]
    for s in SUF:
        if rel.endswith(s): rel = rel[: -len(s)] or rel
    # 只保留末尾两级（类别/模型），可适配多层目录
    parts = rel.split("/")
    if len(parts) >= 2:
        rel2 = "/".join(parts[-2:])
    else:
        rel2 = rel
    return rel, rel2

def build_index(root):
    mp = {}
    for dp,_,fs in os.walk(root):
        for f in fs:
            ext = os.path.splitext(f)[1].lower()
            if ext in EXTS:
                full = os.path.join(dp,f)
                rel = os.path.relpath(full, root)
                k1,k2 = canon_key(rel)
                mp.setdefault(k1, full)
                mp.setdefault(k2, full)
    return mp

def detect_child(root, cand):
    names = {d.lower():d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))}
    for c in cand:
        if c in names: return names[c]
    return None

def convert_split(src_root, list_path, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    split = "train" if list_path.endswith("train.list") else \
            ("val" if list_path.endswith("valid.list") else \
            ("test_novel" if list_path.endswith("test_novel.list") else "test"))
    raw_split = "valid" if split=="val" else split
    split_root = os.path.join(src_root, raw_split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"找不到 {split_root}")

    pdir = detect_child(split_root, PT_DIRS)
    gdir = detect_child(split_root, GT_DIRS)
    if not pdir or not gdir:
        raise FileNotFoundError(f"在 {split_root} 下找不到 {PT_DIRS} 或 {GT_DIRS} 子目录")

    # 一次性建索引（快）
    print(f"[IDX] 建索引: {pdir} / {gdir}")
    pmap = build_index(os.path.join(split_root, pdir))
    gmap = build_index(os.path.join(split_root, gdir))
    print(f"[IDX] partial={len(pmap)}  complete={len(gmap)}")

    lines = [ln.strip() for ln in open(list_path,"r",encoding="utf-8") if ln.strip()]
    n_ok, n_skip = 0, 0
    for i, ln in enumerate(lines, 1):
        if i % 500 == 0:
            print(f"[PROG] {i}/{len(lines)} 已处理，已写入 {n_ok} 对")
        parts = ln.split()
        if len(parts) >= 2:
            p_rel, g_rel = parts[0].lstrip("./\\"), parts[1].lstrip("./\\")
            k1,_ = canon_key(p_rel); k2,_ = canon_key(g_rel)
            pp = pmap.get(k1) or pmap.get(canon_key(os.path.join(p_rel))[1])
            gg = gmap.get(k2) or gmap.get(canon_key(os.path.join(g_rel))[1])
        else:
            tok = parts[0].lstrip("./\\")
            k1,k2 = canon_key(tok)
            pp = pmap.get(k1) or pmap.get(k2)
            gg = gmap.get(k1) or gmap.get(k2)

        if not (pp and gg):
            n_skip += 1
            continue

        out = os.path.join(dst_dir, f"{split}_{n_ok:06d}.h5")
        with h5py.File(out, "w") as h5:
            h5.create_dataset("partial", data=load_pc(pp))
            h5.create_dataset("gt",      data=load_pc(gg))
        n_ok += 1

    print(f"[OK] {os.path.basename(list_path)} -> {dst_dir}  成功 {n_ok} 对，跳过 {n_skip} 条")

def main(src_root, dst_root, max_lines=None):
    os.makedirs(dst_root, exist_ok=True)
    for name in ["train.list", "valid.list", "test.list", "test_novel.list"]:
        lp = os.path.join(src_root, name)
        if not os.path.exists(lp): 
            print(f"[INFO] 跳过 {name}（不存在）"); 
            continue
        # 若只想小样本试跑，可临时截断为前 max_lines 行
        if max_lines:
            tmp = lp + ".tmp_firstN"
            with open(lp,"r",encoding="utf-8") as f, open(tmp,"w",encoding="utf-8") as g:
                for j,line in enumerate(f,1):
                    if j>max_lines: break
                    if line.strip(): g.write(line)
            convert_split(src_root, tmp, os.path.join(dst_root, "val" if name=="valid.list" else name.split(".")[0]))
            os.remove(tmp)
        else:
            convert_split(src_root, lp, os.path.join(dst_root, "val" if name=="valid.list" else name.split(".")[0]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--dst_root", default="./data/PCN")
    ap.add_argument("--max_lines", type=int, default=0, help="仅转换前 N 行（调试用）")
    args = ap.parse_args()
    main(args.src_root, args.dst_root, args.max_lines if args.max_lines>0 else None)
