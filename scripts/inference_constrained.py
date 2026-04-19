"""
Constrained inference: boundary polygon + optional room constraints -> DiGress -> room layout graph

Supports both the constrained model (y_data_dim=1164) and the baseline model (y_data_dim=1000).
Room constraints can be passed at inference; if omitted the model generates unconditionally
(it learned this via CFG dropout during training).

Usage:
    # With boundary and room count constraints
    python scripts/inference_constrained.py --ckpt last.ckpt \
        --boundary "0,0 200,0 200,150 0,150" \
        --rooms "LivingRoom:1,Kitchen:1,Bathroom:2,MasterRoom:1"

    # Unconstrained (boundary only)
    python scripts/inference_constrained.py --ckpt last.ckpt \
        --boundary "0,0 200,0 200,150 0,150"

    # Debug mode: prints per-step stats and raw logits summary
    python scripts/inference_constrained.py --ckpt last.ckpt --debug
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

# ── Room type vocabulary (must match training) ────────────────────────────────
ROOM_TYPES = [
    'LivingRoom', 'MasterRoom', 'Kitchen', 'Bathroom',
    'DiningRoom', 'ChildRoom', 'StudyRoom', 'SecondRoom',
    'GuestRoom', 'Balcony', 'Entrance', 'Storage', 'Wall'
]
ROOM_TYPE_IDX = {rt: i for i, rt in enumerate(ROOM_TYPES)}

# Constraint vector layout (must match floorplan_constrained_dataset.py)
TF_DIM         = 1000
COUNT_DIM      = 14   # 13 room types + 1 bedroom cluster
LOC_DIM        = 125  # 5 constrained types × 25 grid cells
ADJ_DIM        = 25   # 5×5 adjacency
CONSTRAINT_DIM = COUNT_DIM + LOC_DIM + ADJ_DIM   # 164
Y_DIM_CONSTRAINED = TF_DIM + CONSTRAINT_DIM       # 1164

BEDROOM_TYPE_INDICES = {1, 5, 6, 7, 8}  # Master, Child, Study, Second, Guest


# ── Boundary encoding (Turning Function) ─────────────────────────────────────
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


# ── Constraint vector builder ─────────────────────────────────────────────────
def build_y_vector(tf_vec, room_counts=None, y_data_dim=None):
    """
    Build the conditioning vector y for the model.

    tf_vec:       (1000,) TF boundary descriptor
    room_counts:  dict {"RoomType": count, ...} or None for unconstrained
    y_data_dim:   total y dimension from the loaded model

    For the constrained model (y_data_dim=1164):
      y = [TF(1000) | counts(14) | loc_masks(125, zeros) | adj(25, zeros)]
    For the baseline model (y_data_dim=1000):
      y = TF(1000)  (room_counts ignored)
    """
    if y_data_dim is None or y_data_dim <= TF_DIM:
        return tf_vec.astype(np.float32)

    counts = np.zeros(COUNT_DIM, dtype=np.float32)
    if room_counts is not None:
        for rt_name, count in room_counts.items():
            if rt_name in ROOM_TYPE_IDX:
                idx = ROOM_TYPE_IDX[rt_name]
                counts[idx] = float(count)
                if idx in BEDROOM_TYPE_INDICES:
                    counts[13] += float(count)  # bedroom cluster

    loc_masks = np.zeros(LOC_DIM, dtype=np.float32)  # unknown at inference
    adj       = np.zeros(ADJ_DIM, dtype=np.float32)  # unknown at inference
    constraint_part = np.concatenate([counts, loc_masks, adj])
    return np.concatenate([tf_vec, constraint_part]).astype(np.float32)


# ── Post-processing ───────────────────────────────────────────────────────────
def post_process_graph(rooms, edges, room_counts=None, debug=False):
    """
    Clean the raw diffusion output into a valid layout graph.

    room_counts: optional dict of target room counts (from user constraints).
                 If provided, enforces those counts; otherwise uses safe defaults.
    """
    if debug:
        print(f'  [post] raw: {len(rooms)} rooms, {len(edges)} edges')
        from collections import Counter
        print(f'  [post] raw room types: {dict(Counter(rooms))}')

    # 1. Remove self-loops and duplicate edges
    clean_edges = set()
    for u, v in edges:
        if u == v or u >= len(rooms) or v >= len(rooms):
            continue
        clean_edges.add(tuple(sorted((u, v))))
    edges = list(clean_edges)

    # 2. Enforce room count limits
    # Use user-supplied counts as upper bounds when provided; else use safe defaults
    default_max = {'LivingRoom': 1, 'Kitchen': 1, 'Bathroom': 2, 'DiningRoom': 1}
    max_counts = dict(default_max)
    if room_counts is not None:
        for rt, cnt in room_counts.items():
            max_counts[rt] = max(1, int(cnt))  # at least 1

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

    edges = [(mapping[u], mapping[v])
             for u, v in edges if u in mapping and v in mapping]
    rooms = new_rooms

    if debug:
        print(f'  [post] after count filter: {len(rooms)} rooms, {len(edges)} edges')

    # 3. Ensure required rooms are present
    for required in ('LivingRoom', 'Bathroom'):
        if required not in rooms:
            rooms.append(required)
            if debug:
                print(f'  [post] added missing required room: {required}')

    # 4. Limit edge density BEFORE connectivity (connectivity edges are never pruned)
    max_edges = max(len(rooms) * 2, 4)
    if len(edges) > max_edges:
        edges = edges[:max_edges]
        if debug:
            print(f'  [post] trimmed edges to {max_edges}')

    # 5. Ensure connectivity using Union-Find
    if len(rooms) > 1:
        parent = list(range(len(rooms)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
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
            # Connect components via LivingRoom if present, else via first node of each comp
            living_idx = next((i for i, r in enumerate(rooms) if r == 'LivingRoom'), comps[0][0])
            for comp in comps:
                if living_idx not in comp:
                    edges.append((living_idx, comp[0]))
            if debug:
                print(f'  [post] connected {len(comps)} components via LivingRoom')

    if debug:
        print(f'  [post] final: {len(rooms)} rooms, {len(edges)} edges')

    return rooms, edges


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path, debug=False):
    if debug:
        print(f'Loading model from {checkpoint_path} ...')
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        checkpoint_path,
        train_metrics=TrainAbstractMetricsDiscrete(),
        sampling_metrics=TrainAbstractMetricsDiscrete(),
        map_location='cpu',
        strict=False
    )
    model.eval()
    y_dim = model.y_data_dim
    mode = 'constrained (CFG)' if y_dim > TF_DIM else 'baseline (TF only)'
    print(f'Model loaded. mode={mode}, y_data_dim={y_dim}, T={model.T}')
    return model


# ── Sampling ──────────────────────────────────────────────────────────────────
def sample_conditional(model, y_vec, num_samples=4, room_counts=None, debug=False):
    """
    Run reverse diffusion conditioned on y_vec.

    y_vec:       (y_data_dim,) conditioning vector
    room_counts: dict of user-supplied counts, forwarded to post-processing
    """
    device = torch.device('cpu')

    n_nodes = model.node_dist.sample_n(num_samples, device)
    n_max   = torch.max(n_nodes).item()

    arange    = torch.arange(n_max, device=device).unsqueeze(0).expand(num_samples, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    z_T = diffusion_utils.sample_discrete_feature_noise(
        limit_dist=model.limit_dist, node_mask=node_mask
    )
    X, E = z_T.X, z_T.E

    y = torch.tensor(y_vec, dtype=torch.float32).unsqueeze(0).expand(num_samples, -1)

    if debug:
        print(f'\n[sample] n_nodes per sample: {n_nodes.tolist()}')
        print(f'[sample] y vector: TF_norm={float(y[0, :TF_DIM].norm()):.3f}', end='')
        if y.shape[1] > TF_DIM:
            counts_part = y[0, TF_DIM:TF_DIM + COUNT_DIM]
            nonzero = {ROOM_TYPES[i]: int(counts_part[i].item())
                       for i in range(13) if counts_part[i] > 0}
            print(f', constraint counts: {nonzero}', end='')
        print()

    with torch.no_grad():
        for s_int in reversed(range(0, model.T)):
            s_array = s_int * torch.ones((num_samples, 1))
            t_array = s_array + 1
            s_norm  = s_array / model.T
            t_norm  = t_array / model.T

            sampled_s, _ = model.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            if debug and s_int % 100 == 0:
                # Compute fraction of non-zero edges at this step
                edge_frac = E[node_mask].float().mean().item()
                print(f'  [t={model.T - s_int:4d}/{model.T}] edge density={edge_frac:.3f}')

    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E = sampled_s.X, sampled_s.E

    results = []
    for i in range(num_samples):
        n          = n_nodes[i].item()
        atom_types = X[i, :n].cpu()
        edge_types = E[i, :n, :n].cpu()

        rooms = [ROOM_TYPES[min(int(t), len(ROOM_TYPES) - 1)] for t in atom_types]
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)
                 if edge_types[u, v].item() == 1]

        if debug:
            print(f'\n[sample {i + 1}] raw: rooms={rooms}')
            print(f'[sample {i + 1}] raw: edges={edges}')

        rooms_pp, edges_pp = post_process_graph(rooms, edges, room_counts=room_counts, debug=debug)

        results.append({
            'raw':       {'rooms': rooms, 'edges': edges},
            'processed': {'rooms': rooms_pp, 'edges': edges_pp},
        })

    return results


# ── Pretty printing ───────────────────────────────────────────────────────────
def print_results(results):
    for i, r in enumerate(results):
        print(f'\n{"=" * 48}  SAMPLE {i + 1}  {"=" * 48}\n')

        raw = r['raw']
        print(f'RAW GRAPH  ({len(raw["rooms"])} rooms, {len(raw["edges"])} edges)')
        print(f'  Rooms: {raw["rooms"]}')
        for u, v in raw['edges']:
            print(f'  {raw["rooms"][u]}  --  {raw["rooms"][v]}')

        proc = r['processed']
        print(f'\nPOST-PROCESSED  ({len(proc["rooms"])} rooms, {len(proc["edges"])} edges)')
        print(f'  Rooms: {proc["rooms"]}')
        for u, v in proc['edges']:
            print(f'  {proc["rooms"][u]}  --  {proc["rooms"][v]}')

        print()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        required=True,
                        help='Path to .ckpt checkpoint file')
    parser.add_argument('--boundary',    type=str, default=None,
                        help='Space-separated "x,y" vertices, e.g. "0,0 200,0 200,150 0,150"')
    parser.add_argument('--rooms',       type=str, default=None,
                        help='Comma-separated "Type:count" pairs, e.g. "LivingRoom:1,Kitchen:1,Bathroom:2"')
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--debug',       action='store_true',
                        help='Print per-step stats and raw logit summaries')
    args = parser.parse_args()

    # Parse boundary
    if args.boundary:
        pts      = [list(map(float, p.split(','))) for p in args.boundary.strip().split()]
        boundary = np.array(pts)
    else:
        boundary = np.array([[0, 0], [200, 0], [200, 150], [0, 150]])
        print('Using default rectangular boundary.')

    # Parse room constraints
    room_counts = None
    if args.rooms:
        room_counts = {}
        for token in args.rooms.split(','):
            name, cnt = token.strip().split(':')
            room_counts[name.strip()] = int(cnt.strip())
        print(f'Room constraints: {room_counts}')
    else:
        print('No room constraints — unconstrained inference.')

    tf_vec = compute_tf(boundary)
    model  = load_model(args.ckpt, debug=args.debug)

    y_vec  = build_y_vector(tf_vec, room_counts=room_counts, y_data_dim=model.y_data_dim)
    print(f'y vector shape: {y_vec.shape}  '
          f'(TF={TF_DIM}, constraints={y_vec.shape[0] - TF_DIM})')

    results = sample_conditional(
        model, y_vec,
        num_samples=args.num_samples,
        room_counts=room_counts,
        debug=args.debug,
    )

    print_results(results)


if __name__ == '__main__':
    main()
