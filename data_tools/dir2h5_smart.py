# pcc_lite/data_tools/dir2h5_smart.py
import os, argparse, h5py, numpy as np, open3d as o3d, re

EXTS = {".pcd", ".ply", ".xyz", ".pts"}
GT_DIR_CAND = ["complete", "gt", "groundtruth"]
PT_DIR_CAND = ["partial", "incomplete"]

def load_pc(p):
    pc = o3d.io.read_point_cloud(p)
    pts = np.asarray(pc.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"{p} 不是 Nx3 点云")
    return pts

def is_synset(s):        # 类别 ID：8 位数字（如 04225987）
    return len(s) == 8 and s.isdigit()

def tokens_from_rel(rel_noext):
    """从相对路径去扩展名后抽取关键信息（类别ID/模型ID）"""
    parts = rel_noext.replace("\\", "/").lstrip("./").split("/")
    synset = None
    # 1) 先找类别ID
    for i,p in enumerate(parts):
        if is_synset(p):
            synset = p
            # 2) 从类别ID往后找“最长的字母数字串”作为模型ID
            tail = parts[i+1:] if i+1 < len(parts) else []
            break
    else:
        # 没有类别ID：就全路径都算 tail
        tail = parts
    # 从 tail 和文件名里挑一个“最长的字母数字段”
    cand = []
    for t in tail:
        name = os.path.basename(t)
        name = os.path.splitext(name)[0]
        for seg in re.findall(r"[A-Za-z0-9]+", name):
            cand.append(seg)
    if not cand:
        # 兜底：用最后一段
        cand = [parts[-1]]
    model = max(cand, key=len)
    # 主键：带类别ID；备用键：只有模型ID（以防一侧缺类别ID）
    key_main = f"{synset}/{model}" if synset else model
    key_alt  = model
    return key_main, key_alt

def build_index(root):
    """把 root 下所有点云 -> {key: path}，key 同时登记 (synset/model) 与 (model) 两种"""
    mp = {}
    for dp,_,fs in os.walk(root):
        for f in fs:
            ext = os.path.splitext(f)[1].lower()
            if ext not in EXTS: 
                continue
            full = os.path.join(dp,f)
            rel = os.path.relpath(full, root)
            noext = os.path.splitext(rel)[0]
            k1, k2 = tokens_from_rel(noext)
            mp.setdefault(k1, full)
            mp.setdefault(k2, full)
    return mp

def detect_child(root, cand):
    names = {d.lower():d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))}
    for c in cand:
        if c in names: return names[c]
    return None

def convert_split(src_root, split, dst_root, pdir_name="auto", gdir_name="auto", limit=0, debug=False):
    in_split = "valid" if split=="val" else split
    split_root = os.path.join(src_root, in_split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"找不到目录：{split_root}")

    pdir = detect_child(split_root, PT_DIR_CAND) if pdir_name=="auto" else pdir_name
    gdir = detect_child(split_root, GT_DIR_CAND) if gdir_name=="auto" else gdir_name
    pdir = pdir or "partial"; gdir = gdir or "complete"
    p_root = os.path.join(split_root, pdir)
    g_root = os.path.join(split_root, gdir)
    if not (os.path.isdir(p_root) and os.path.isdir(g_root)):
        raise FileNotFoundError(f"未找到 {p_root} 或 {g_root}")

    if debug: print(f"[DBG] 使用目录：partial={p_root}  complete={g_root}")

    print(f"[IDX] 建索引（可能要几秒）…")
    pmap = build_index(p_root)
    gmap = build_index(g_root)
    if debug:
        print(f"[DBG] partial 索引样例：", list(pmap.items())[:3])
        print(f"[DBG] complete 索引样例：", list(gmap.items())[:3])

    keys = sorted(set(pmap.keys()) & set(gmap.keys()))
    if not keys:
        raise RuntimeError(f"{split} 没有配到任何 key：检查两侧层级或文件名是否一致")

    out_dir = os.path.join(dst_root, split)
    os.makedirs(out_dir, exist_ok=True)
    if limit > 0:
        keys = keys[:limit]
        print(f"[INFO] 仅转换前 {limit} 对（调试）")

    ok = 0
    for i, k in enumerate(keys):
        pp, gg = pmap[k], gmap[k]
        out = os.path.join(out_dir, f"{split}_{i:06d}.h5")
        with h5py.File(out, "w") as h5:
            h5.create_dataset("partial", data=load_pc(pp))
            h5.create_dataset("gt",      data=load_pc(gg))
        ok += 1
        if debug and ok <= 3:
            print(f"[DBG] pair[{ok}] -> {pp}  |  {gg}")
        if ok % 500 == 0:
            print(f"[PROG] {ok}/{len(keys)}")
    print(f"[OK] {split}: 成功 {ok} 对 -> {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="PCN 根目录（含 train/valid/test/...）")
    ap.add_argument("--dst_root", default="./data/PCN")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--partial_name", default="auto", help="默认自动检测，也可指定 partial/incomplete")
    ap.add_argument("--gt_name", default="auto", help="默认自动检测，也可指定 complete/gt/groundtruth")
    ap.add_argument("--limit", type=int, default=0, help="仅转换前 N 对（调试）")
    ap.add_argument("--debug", type=int, default=0, help="1 打印样例与配对信息")
    args = ap.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        convert_split(args.src_root, "val" if sp in ["valid","val","validation"] else sp,
                      args.dst_root, args.partial_name, args.gt_name, args.limit, bool(args.debug))
