import os
import sys
import numpy as np
import torch

_DIGRESS_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'DiGress', 'src')
)
if _DIGRESS_SRC not in sys.path:
    sys.path.insert(0, _DIGRESS_SRC)

ROOM_TYPES = [
    'LivingRoom', 'MasterRoom', 'Kitchen', 'Bathroom',
    'DiningRoom', 'ChildRoom', 'StudyRoom', 'SecondRoom',
    'GuestRoom', 'Balcony', 'Entrance', 'Storage', 'Wall',
]

_model = None


def compute_tf(boundary_xy, ndim=1000):
    b = np.array(boundary_xy, dtype=np.float64)
    if b.ndim == 2 and b.shape[1] > 2:
        b = b[:, :2]
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


def init_model(ckpt_path):
    global _model
    if not os.path.exists(ckpt_path):
        print(f'[DiGress] Checkpoint not found: {ckpt_path}')
        return False
    try:
        from diffusion_model_discrete import DiscreteDenoisingDiffusion
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        _model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            ckpt_path,
            train_metrics=TrainAbstractMetricsDiscrete(),
            sampling_metrics=TrainAbstractMetricsDiscrete(),
            map_location='cpu',
            strict=False,
        )
        _model.eval()
        print(f'[DiGress] Model loaded (T={_model.T})')
        return True
    except Exception as exc:
        print(f'[DiGress] Load failed: {exc}')
        return False


def is_loaded():
    return _model is not None


def sample_rooms(boundary, num_samples=1):
    if _model is None:
        raise RuntimeError('DiGress model not loaded')
    from diffusion import diffusion_utils

    device = torch.device('cpu')
    tf_vec = compute_tf(boundary)

    n_nodes = _model.node_dist.sample_n(num_samples, device)
    n_max = int(torch.max(n_nodes).item())
    arange = torch.arange(n_max, device=device).unsqueeze(0).expand(num_samples, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    z_T = diffusion_utils.sample_discrete_feature_noise(
        limit_dist=_model.limit_dist, node_mask=node_mask
    )
    X, E = z_T.X, z_T.E
    y = torch.tensor(tf_vec, dtype=torch.float32).unsqueeze(0).expand(num_samples, -1).clone()

    with torch.no_grad():
        for s_int in reversed(range(0, _model.T)):
            s_array = s_int * torch.ones((num_samples, 1))
            t_array = s_array + 1
            s_norm = s_array / _model.T
            t_norm = t_array / _model.T
            sampled_s, _ = _model.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E = sampled_s.X, sampled_s.E

    results = []
    for i in range(num_samples):
        n = int(n_nodes[i].item())
        rooms = [ROOM_TYPES[min(int(t), 12)] for t in X[i, :n].cpu()]
        edges = [
            [u, v]
            for u in range(n)
            for v in range(u + 1, n)
            if E[i, :n, :n].cpu()[u, v] == 1
        ]
        results.append({'rooms': rooms, 'edges': edges})
    return results
