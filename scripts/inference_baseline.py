"""
Inference bridge: boundary polygon -> DiGress -> room layout graph (with post-processing)

Usage:
    python scripts/inference.py --ckpt last.ckpt
    python scripts/inference.py --ckpt last.ckpt --boundary "0,0 100,0 100,100 0,100"
    python scripts/inference.py --ckpt last.ckpt --num-samples 4
"""

import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DiGress', 'src'))

from diffusion_model_discrete import DiscreteDenoisingDiffusion
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from diffusion import diffusion_utils


ROOM_TYPES = [
    'LivingRoom', 'MasterRoom', 'Kitchen', 'Bathroom',
    'DiningRoom', 'ChildRoom', 'StudyRoom', 'SecondRoom',
    'GuestRoom', 'Balcony', 'Entrance', 'Storage', 'Wall'
]


# -------------------------
# Boundary encoding (TF)
# -------------------------
def compute_tf(boundary_xy, ndim=1000):
    b = np.array(boundary_xy, dtype=np.float64)
    b = np.concatenate((b, b[:1]))
    n = len(b) - 1
    v = b[1:] - b[:-1]
    L = np.linalg.norm(v, axis=1)
    perim = L.sum()
    if perim == 0:
        return np.zeros(ndim, dtype=np.float32)

    v = v / perim
    L = L / perim

    angles = np.zeros(n)
    for i in range(n):
        z = np.cross(v[i], v[(i + 1) % n])
        dot = np.clip(np.dot(v[i], v[(i + 1) % n]), -1.0, 1.0)
        angles[i] = np.arccos(dot) * (np.sign(z) if z != 0 else 1.0)

    x_vals = np.zeros(n + 1)
    y_vals = np.zeros(n + 1)
    s = 0.0
    for i in range(1, n + 1):
        x_vals[i] = L[i - 1] + x_vals[i - 1]
        y_vals[i - 1] = angles[i - 1] + s
        s = y_vals[i - 1]
    y_vals[-1] = s

    t = np.linspace(0, 1, ndim)
    return np.piecewise(t, [t >= xv for xv in x_vals], y_vals).astype(np.float32)


# -------------------------
# Model loading
# -------------------------
def load_model(checkpoint_path):
    print(f"Loading model from {checkpoint_path} ...")
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        checkpoint_path,
        train_metrics=TrainAbstractMetricsDiscrete(),
        sampling_metrics=TrainAbstractMetricsDiscrete(),
        map_location='cpu',
        strict=False
    )
    model.eval()
    print(f"Model loaded. y_data_dim={model.y_data_dim}, T={model.T}")
    return model


# -------------------------
# Sampling
# -------------------------
def sample_conditional(model, tf_vec, num_samples=4):
    device = torch.device('cpu')

    n_nodes = model.node_dist.sample_n(num_samples, device)
    n_max = torch.max(n_nodes).item()

    arange = torch.arange(n_max, device=device).unsqueeze(0).expand(num_samples, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    z_T = diffusion_utils.sample_discrete_feature_noise(
        limit_dist=model.limit_dist,
        node_mask=node_mask
    )

    X, E = z_T.X, z_T.E

    y = torch.tensor(tf_vec, dtype=torch.float32).unsqueeze(0).expand(num_samples, -1)

    with torch.no_grad():
        for s_int in reversed(range(0, model.T)):
            s_array = s_int * torch.ones((num_samples, 1))
            t_array = s_array + 1

            s_norm = s_array / model.T
            t_norm = t_array / model.T

            sampled_s, _ = model.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E = sampled_s.X, sampled_s.E

    results = []

    for i in range(num_samples):
        n = n_nodes[i].item()
        atom_types = X[i, :n].cpu()
        edge_types = E[i, :n, :n].cpu()

        rooms = [ROOM_TYPES[min(int(t), 12)] for t in atom_types]
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if edge_types[u, v] == 1]

        results.append({'rooms': rooms, 'edges': edges})

    return results


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--boundary', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=4)
    args = parser.parse_args()

    if args.boundary:
        pts = [list(map(float, p.split(','))) for p in args.boundary.strip().split()]
        boundary = np.array(pts)
    else:
        boundary = np.array([[0,0],[200,0],[200,150],[0,150]])
        print("Using default boundary.")

    tf_vec = compute_tf(boundary)
    model = load_model(args.ckpt)

    results = sample_conditional(model, tf_vec, args.num_samples)

    for i, r in enumerate(results):
        print(f"\n================ SAMPLE {i+1} ================\n")
        print(f"  Rooms ({len(r['rooms'])}): {r['rooms']}")
        print(f"  Edges ({len(r['edges'])}):")
        for u, v in r['edges']:
            print(f"    {r['rooms'][u]} -- {r['rooms'][v]}")
        print("\n===========================================\n")


if __name__ == '__main__':
    main()