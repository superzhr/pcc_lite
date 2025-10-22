"""Generate a tiny toy PCN-style dataset (.h5 files) to validate the pipeline.

Each sample is a simple shape (sphere/cube/cylinder) with random rotation and jitter.
Partial point clouds are created by slicing the full shape with a random plane.
"""
import os, h5py, argparse, numpy as np

def sample_sphere(n):
    u = np.random.rand(n)
    v = np.random.rand(n)
    theta = 2 * np.pi * u
    phi = np.arccos(2*v - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x,y,z], axis=1)

def sample_cube(n):
    pts = np.random.rand(n,3)*2-1
    # attract to surface
    face = np.random.randint(0,3, size=(n,))
    signs = np.random.choice([-1,1], size=(n,))
    for i in range(n):
        pts[i, face[i]] = signs[i]
    return pts

def sample_cylinder(n):
    theta = 2*np.pi*np.random.rand(n)
    z = np.random.rand(n)*2-1
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack([x,y,z], axis=1)

def random_shape(n):
    choice = np.random.choice(['sphere','cube','cyl'])
    if choice=='sphere': pts = sample_sphere(n)
    elif choice=='cube': pts = sample_cube(n)
    else: pts = sample_cylinder(n)
    # random rotation
    R = random_rotation()
    pts = pts @ R.T
    # jitter
    pts += np.random.normal(scale=0.005, size=pts.shape)
    return pts

def random_rotation():
    a,b,c = np.random.rand(3)*2*np.pi
    Rx = np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    Ry = np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    Rz = np.array([[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]])
    return Rz@Ry@Rx

def make_split(root, split, n_samples, n_full=10000, n_partial=3000):
    os.makedirs(os.path.join(root, split), exist_ok=True)
    for i in range(n_samples):
        gt = random_shape(n_full)
        # partial by slicing with a random plane n·x>t
        n = np.random.randn(3); n /= np.linalg.norm(n)+1e-8
        t = np.random.uniform(-0.2, 0.4)
        mask = (gt @ n) > t
        partial_full = gt[mask]
        if partial_full.shape[0] < n_partial:
            idx = np.random.choice(partial_full.shape[0], n_partial, replace=True)
        else:
            idx = np.random.choice(partial_full.shape[0], n_partial, replace=False)
        partial = partial_full[idx]
        path = os.path.join(root, split, f"toy_{i:04d}.h5")
        with h5py.File(path, 'w') as f:
            f.create_dataset('partial', data=partial.astype('float32'))
            f.create_dataset('gt', data=gt.astype('float32'))
    print(f"[OK] Wrote {n_samples} samples to {os.path.join(root, split)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='./data/PCN')
    ap.add_argument('--n_train', type=int, default=60)
    ap.add_argument('--n_val', type=int, default=10)
    ap.add_argument('--n_test', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    make_split(args.root, 'train', args.n_train)
    make_split(args.root, 'val', args.n_val)
    make_split(args.root, 'test', args.n_test)
    print('[DONE] Toy PCN created.')
