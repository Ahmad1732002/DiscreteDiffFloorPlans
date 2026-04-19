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
ROOM_TYPE_IDX = {rt: i for i, rt in enumerate(ROOM_TYPES)}

TF_DIM         = 1000
COUNT_DIM      = 14   # 13 room types + 1 bedroom cluster
LOC_DIM        = 125  # zeros at inference
ADJ_DIM        = 25   # zeros at inference
BEDROOM_TYPE_INDICES = {1, 5, 6, 7, 8}  # Master, Child, Study, Second, Guest

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


def _build_y_vector(tf_vec, room_counts, y_data_dim):
    if y_data_dim <= TF_DIM:
        return tf_vec.astype(np.float32)

    counts = np.zeros(COUNT_DIM, dtype=np.float32)
    if room_counts:
        for rt_name, count in room_counts.items():
            if rt_name in ROOM_TYPE_IDX:
                idx = ROOM_TYPE_IDX[rt_name]
                counts[idx] = float(count)
                if idx in BEDROOM_TYPE_INDICES:
                    counts[13] += float(count)

    loc_masks = np.zeros(LOC_DIM, dtype=np.float32)
    adj       = np.zeros(ADJ_DIM, dtype=np.float32)
    return np.concatenate([tf_vec, counts, loc_masks, adj]).astype(np.float32)


def _post_process_graph(rooms, edges, room_counts=None):
    clean_edges = set()
    for u, v in edges:
        if u == v or u >= len(rooms) or v >= len(rooms):
            continue
        clean_edges.add(tuple(sorted((u, v))))
    edges = list(clean_edges)

    default_max = {'LivingRoom': 1, 'Kitchen': 1, 'Bathroom': 2, 'DiningRoom': 1}
    max_counts = dict(default_max)
    if room_counts:
        for rt, cnt in room_counts.items():
            max_counts[rt] = max(1, int(cnt))

    new_rooms = []
    mapping   = {}
    counts    = {}
    for i, r in enumerate(rooms):
        if r in max_counts:
            counts[r] = counts.get(r, 0) + 1
            if counts[r] > max_counts[r]:
                continue
        mapping[i] = len(new_rooms)
        new_rooms.append(r)

    edges = [(mapping[u], mapping[v]) for u, v in edges if u in mapping and v in mapping]
    rooms = new_rooms

    # Fill missing required rooms up to the requested count
    for required in ('LivingRoom', 'Bathroom'):
        target = (room_counts or {}).get(required, 1)
        current = sum(1 for r in rooms if r == required)
        for _ in range(target - current):
            rooms.append(required)

    max_edges = max(len(rooms) * 2, 4)
    if len(edges) > max_edges:
        edges = edges[:max_edges]

    if len(rooms) > 1:
        parent = list(range(len(rooms)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for u, v in edges:
            union(u, v)

        comps = {}
        for i in range(len(rooms)):
            comps.setdefault(find(i), []).append(i)

        comps = list(comps.values())
        if len(comps) > 1:
            living_idx = next((i for i, r in enumerate(rooms) if r == 'LivingRoom'), comps[0][0])
            for comp in comps:
                if living_idx not in comp:
                    edges.append((living_idx, comp[0]))

    return rooms, edges


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
        mode = 'constrained' if _model.y_data_dim > TF_DIM else 'baseline'
        print(f'[DiGress] Model loaded (mode={mode}, y_data_dim={_model.y_data_dim}, T={_model.T})')
        return True
    except Exception as exc:
        print(f'[DiGress] Load failed: {exc}')
        return False


def is_loaded():
    return _model is not None


def sample_rooms(boundary, num_samples=1, room_counts=None):
    """
    boundary:    numpy array of boundary vertices
    room_counts: dict {room_name: count} or None for unconstrained
    """
    if _model is None:
        raise RuntimeError('DiGress model not loaded')
    from diffusion import diffusion_utils

    device = torch.device('cpu')
    tf_vec = compute_tf(boundary)
    y_vec  = _build_y_vector(tf_vec, room_counts, _model.y_data_dim)

    n_nodes = _model.node_dist.sample_n(num_samples, device)
    n_max   = int(torch.max(n_nodes).item())
    arange  = torch.arange(n_max, device=device).unsqueeze(0).expand(num_samples, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    z_T = diffusion_utils.sample_discrete_feature_noise(
        limit_dist=_model.limit_dist, node_mask=node_mask
    )
    X, E = z_T.X, z_T.E
    y = torch.tensor(y_vec, dtype=torch.float32).unsqueeze(0).expand(num_samples, -1).clone()

    with torch.no_grad():
        for s_int in reversed(range(0, _model.T)):
            s_array = s_int * torch.ones((num_samples, 1))
            t_array = s_array + 1
            s_norm  = s_array / _model.T
            t_norm  = t_array / _model.T
            sampled_s, _ = _model.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E = sampled_s.X, sampled_s.E

    results = []
    for i in range(num_samples):
        n     = int(n_nodes[i].item())
        rooms = [ROOM_TYPES[min(int(t), len(ROOM_TYPES) - 1)] for t in X[i, :n].cpu()]
        edges = [
            [u, v]
            for u in range(n)
            for v in range(u + 1, n)
            if E[i, :n, :n].cpu()[u, v] == 1
        ]
        rooms, edges = _post_process_graph(rooms, edges, room_counts=room_counts)
        results.append({'rooms': rooms, 'edges': edges})
    return results
