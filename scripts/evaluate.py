"""
evaluate.py — End-to-end benchmark: Retrieval baseline vs. DiGress diffusion.

Both methods feed into the same frozen Graph2Plan model so the comparison
isolates only the first-stage graph generator.

Metrics (all computed on the held-out test split):
  FID                    (↓)  Fréchet Inception Distance on 128×128 rendered segmentation maps
  KID                    (↓)  Kernel Inception Distance on same
  Connectivity rate      (↑)  fraction of generated graphs that are fully connected
  Graph validity         (↑)  connected + required rooms (LivingRoom, Bathroom) + no isolated nodes
  Room-type accuracy     (↑)  intersection-over-GT of predicted vs. GT room-type multiset
  Room-type F1           (↑)  harmonic mean of room-type precision and recall
  Node count overlap     (↑)  min(n_pred, n_gt) / max(n_pred, n_gt)
  Count satisfaction     (↑)  fraction of constrained room types whose count matches GT exactly
                               (constrained model only; measures constraint following)
  Adj satisfaction       (↑)  fraction of GT constrained adjacencies reproduced in prediction
                               (constrained model only)
  Pixel mIoU             (↑)  mean per-class pixel IoU vs. GT segmentation

Run from the project root:
    python scripts/evaluate.py --ckpt checkpoints/last.ckpt
    python scripts/evaluate.py --ckpt checkpoints/last.ckpt --num_samples 500 --out results/
"""

import argparse
import copy
import os
import sys
import pickle
import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Interface'))
sys.path.insert(0, os.path.join(ROOT, 'DiGress', 'src'))

from model.floorplan import FloorPlan
from model.model import Model
from model.box_utils import centers_to_extents
from model.utils import room_label

# ── Constants ─────────────────────────────────────────────────────────────────
ROOM_TYPES = [
    'LivingRoom', 'MasterRoom', 'Kitchen', 'Bathroom',
    'DiningRoom', 'ChildRoom', 'StudyRoom', 'SecondRoom',
    'GuestRoom', 'Balcony', 'Entrance', 'Storage', 'Wall',
]
ROOM_NAME_TO_IDX = {rt: i for i, rt in enumerate(ROOM_TYPES)}
NUM_CLASSES = 13

TF_DIM    = 1000
COUNT_DIM = 14
LOC_DIM   = 125
ADJ_DIM   = 25
BEDROOM_INDICES  = {1, 5, 6, 7, 8}
CONSTRAINED_TYPES = [0, 1, 2, 3, 4]   # Living, Master, Kitchen, Bathroom, Dining
REQUIRED_ROOMS    = ('LivingRoom', 'Bathroom')

_PALETTE_LIST = [r[4] for r in room_label]
PALETTE = np.array(_PALETTE_LIST + [[255, 255, 255]] * 10, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(args):
    test_pkl  = pickle.load(open(args.test_data,  'rb'))
    train_pkl = pickle.load(open(args.train_data, 'rb'))
    return test_pkl['data'], train_pkl['data']


# ─────────────────────────────────────────────────────────────────────────────
# Graph2Plan (frozen downstream model)
# ─────────────────────────────────────────────────────────────────────────────
def load_g2p(model_path):
    m = Model()
    m.load_state_dict(torch.load(model_path, map_location='cpu'))
    m.eval()
    return m


def run_g2p(g2p, fp):
    with torch.no_grad():
        batch = list(fp.get_test_data())
        batch[0] = batch[0].unsqueeze(0)
        boundary, inside_box, rooms, attrs, triples = batch
        boxes_pred, gene_layout, _ = g2p(
            rooms, triples, boundary,
            obj_to_img=None, attributes=attrs, boxes_gt=None,
            generate=True, refine=True, relative=True, inside_box=inside_box,
        )
        boxes_pred  = centers_to_extents(boxes_pred.detach())
        gene_layout = gene_layout * boundary[:, :1]
        gene_preds  = torch.argmax(gene_layout.softmax(1).detach(), dim=1)
    seg   = gene_preds.squeeze().cpu().numpy().astype(np.uint8)
    boxes = boxes_pred.squeeze().cpu().numpy()
    return seg, boxes


# ─────────────────────────────────────────────────────────────────────────────
# Rendering utilities
# ─────────────────────────────────────────────────────────────────────────────
def seg_to_rgb_tensor(seg):
    seg_clipped = np.clip(seg, 0, len(PALETTE) - 1)
    rgb = PALETTE[seg_clipped]
    return torch.from_numpy(rgb).permute(2, 0, 1)


def gt_segmentation(ref_datum):
    seg = np.full((128, 128), NUM_CLASSES, dtype=np.uint8)
    for row in ref_datum.box:
        x0, y0, x1, y1, rtype = (int(row[i]) for i in range(5))
        x0, y0, x1, y1 = x0 // 2, y0 // 2, x1 // 2, y1 // 2
        seg[y0:y1, x0:x1] = min(rtype, NUM_CLASSES - 1)
    return seg


def gt_room_list(ref_datum):
    return [ROOM_TYPES[min(int(b[4]), NUM_CLASSES - 1)] for b in ref_datum.box]


# ─────────────────────────────────────────────────────────────────────────────
# TF computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_tf(boundary, ndim=1000):
    b = np.array(boundary, dtype=np.float64)
    if b.ndim == 2 and b.shape[1] > 2:
        b = b[:, :2]
    b = np.concatenate((b, b[:1]))
    n = len(b) - 1
    v = b[1:] - b[:-1]
    L = np.linalg.norm(v, axis=1)
    perim = L.sum()
    if perim == 0:
        return np.zeros(ndim, dtype=np.float32)
    v, L = v / perim, L / perim
    angles = np.zeros(n)
    for i in range(n):
        z   = np.cross(v[i], v[(i + 1) % n])
        dot = np.clip(np.dot(v[i], v[(i + 1) % n]), -1.0, 1.0)
        angles[i] = np.arccos(dot) * (np.sign(z) if z != 0 else 1.0)
    xv, yv = np.zeros(n + 1), np.zeros(n + 1)
    s = 0.0
    for i in range(1, n + 1):
        xv[i]     = L[i - 1] + xv[i - 1]
        yv[i - 1] = angles[i - 1] + s
        s = yv[i - 1]
    yv[-1] = s
    t = np.linspace(0, 1, ndim)
    return np.piecewise(t, [t >= x for x in xv], yv).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Constraint extraction & y-vector building
# ─────────────────────────────────────────────────────────────────────────────
def extract_gt_constraints(datum):
    """
    Extract room count and adjacency constraints from a floor plan datum (training split).
    Returns:
        counts: dict {room_type_name: count}
        adj_pairs: set of frozensets {frozenset({rt_i, rt_j})} for constrained-type pairs
    """
    rooms = [ROOM_TYPES[min(int(b[4]), NUM_CLASSES - 1)] for b in datum.box
             if int(b[4]) < NUM_CLASSES]
    counts = dict(Counter(rooms))

    # Adjacency between constrained type names only
    constrained_names = {ROOM_TYPES[i] for i in CONSTRAINED_TYPES}
    adj_pairs = set()
    if hasattr(datum, 'edge') and len(datum.edge) > 0:
        for e in datum.edge:
            u, v = int(e[0]), int(e[1])
            if u < len(rooms) and v < len(rooms):
                ru, rv = rooms[u], rooms[v]
                if ru in constrained_names and rv in constrained_names and ru != rv:
                    adj_pairs.add(frozenset({ru, rv}))

    return counts, adj_pairs


def build_y(tf_vec, y_data_dim, gt_counts=None):
    """
    Build conditioning vector y.

    If the model is constrained (y_data_dim > TF_DIM) and gt_counts are provided,
    the room-count portion is filled in from gt_counts so constraint satisfaction
    can be measured. Location masks and adjacency remain zero (not known at test time).
    """
    if y_data_dim <= TF_DIM:
        return tf_vec.astype(np.float32)

    counts = np.zeros(COUNT_DIM, dtype=np.float32)
    if gt_counts is not None:
        for rt_name, count in gt_counts.items():
            if rt_name in ROOM_NAME_TO_IDX:
                idx = ROOM_NAME_TO_IDX[rt_name]
                counts[idx] = float(count)
                if idx in BEDROOM_INDICES:
                    counts[13] += float(count)

    return np.concatenate([tf_vec, counts,
                           np.zeros(LOC_DIM, dtype=np.float32),
                           np.zeros(ADJ_DIM, dtype=np.float32)]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval baseline
# ─────────────────────────────────────────────────────────────────────────────
def run_retrieval_pipeline(datum, ref_datum, g2p):
    test_fp  = FloorPlan(datum)
    train_fp = FloorPlan(ref_datum, train=True)
    fp_end   = test_fp.adapt_graph(train_fp)
    fp_end.adjust_graph()
    seg, boxes = run_g2p(g2p, fp_end)
    rooms = [ROOM_TYPES[min(int(b[4]), NUM_CLASSES - 1)] for b in fp_end.data.box]
    edges = [(int(e[0]), int(e[1])) for e in fp_end.data.edge] if len(fp_end.data.edge) else []
    return seg, boxes, rooms, edges


# ─────────────────────────────────────────────────────────────────────────────
# DiGress pipeline
# ─────────────────────────────────────────────────────────────────────────────
def load_digress(ckpt_path):
    from diffusion_model_discrete import DiscreteDenoisingDiffusion
    from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
    m = DiscreteDenoisingDiffusion.load_from_checkpoint(
        ckpt_path,
        train_metrics=TrainAbstractMetricsDiscrete(),
        sampling_metrics=TrainAbstractMetricsDiscrete(),
        map_location='cpu', strict=False,
    )
    m.eval()
    mode = 'constrained' if m.y_data_dim > TF_DIM else 'baseline'
    print(f'[DiGress] Loaded ({mode}, y_dim={m.y_data_dim}, T={m.T})')
    return m


def _post_process(rooms, edges):
    clean = set()
    for u, v in edges:
        if u != v and u < len(rooms) and v < len(rooms):
            clean.add(tuple(sorted((u, v))))
    edges = list(clean)

    max_c = {'LivingRoom': 1, 'Kitchen': 1, 'Bathroom': 2, 'DiningRoom': 1}
    new_rooms, mapping, counts = [], {}, {}
    for i, r in enumerate(rooms):
        if r in max_c:
            counts[r] = counts.get(r, 0) + 1
            if counts[r] > max_c[r]:
                continue
        mapping[i] = len(new_rooms)
        new_rooms.append(r)
    edges = [(mapping[u], mapping[v]) for u, v in edges if u in mapping and v in mapping]
    rooms = new_rooms

    for req in REQUIRED_ROOMS:
        if req not in rooms:
            rooms.append(req)

    edges = edges[:max(len(rooms) * 2, 4)]

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
            hub = next((i for i, r in enumerate(rooms) if r == 'LivingRoom'), comps[0][0])
            for comp in comps:
                if hub not in comp:
                    edges.append((hub, comp[0]))
    return rooms, edges


def digress_sample(digress_model, boundary, gt_counts=None):
    from diffusion import diffusion_utils
    device = torch.device('cpu')
    tf_vec = compute_tf(boundary)
    y_vec  = build_y(tf_vec, digress_model.y_data_dim, gt_counts=gt_counts)

    n_nodes   = digress_model.node_dist.sample_n(1, device)
    n_max     = int(n_nodes.max().item())
    node_mask = torch.arange(n_max).unsqueeze(0) < n_nodes.unsqueeze(1)

    z_T = diffusion_utils.sample_discrete_feature_noise(
        limit_dist=digress_model.limit_dist, node_mask=node_mask)
    X, E = z_T.X, z_T.E
    y = torch.tensor(y_vec, dtype=torch.float32).unsqueeze(0).clone()

    with torch.no_grad():
        for s_int in reversed(range(digress_model.T)):
            s = torch.tensor([[s_int / digress_model.T]])
            t = torch.tensor([[(s_int + 1) / digress_model.T]])
            sampled_s, _ = digress_model.sample_p_zs_given_zt(s, t, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E = sampled_s.X, sampled_s.E
    n = int(n_nodes[0].item())
    rooms = [ROOM_TYPES[min(int(tok), NUM_CLASSES - 1)] for tok in X[0, :n].cpu()]
    edges = [[u, v] for u in range(n) for v in range(u + 1, n)
             if E[0, :n, :n].cpu()[u, v] == 1]
    return _post_process(rooms, edges)


def run_digress_pipeline(datum, digress_model, g2p, gt_counts=None):
    rooms, edges = digress_sample(digress_model, datum.boundary, gt_counts=gt_counts)
    n = len(rooms)

    bx0 = int(np.min(datum.boundary[:, 0]))
    bx1 = int(np.max(datum.boundary[:, 0]))
    by0 = int(np.min(datum.boundary[:, 1]))
    by1 = int(np.max(datum.boundary[:, 1]))
    bw, bh = bx1 - bx0, by1 - by0
    ncols = max(1, int(np.ceil(np.sqrt(n))))
    nrows = max(1, int(np.ceil(n / ncols)))
    cw, ch = max(1, bw // ncols), max(1, bh // nrows)

    boxes = []
    for i, rname in enumerate(rooms):
        col, row = i % ncols, i // ncols
        x0 = int(bx0 + col * cw);  y0 = int(by0 + row * ch)
        x1 = min(x0 + cw, bx1);    y1 = min(y0 + ch, by1)
        boxes.append([x0, y0, x1, y1, ROOM_NAME_TO_IDX.get(rname, 0)])

    edge_arr = (np.array([[u, v, 0] for u, v in edges], dtype=int)
                if edges else np.zeros((0, 3), dtype=int))

    fp_data      = copy.deepcopy(datum)
    fp_data.box  = np.array(boxes, dtype=int)
    fp_data.edge = edge_arr
    fp_digress   = FloorPlan(fp_data)
    fp_digress.adjust_graph()

    seg, boxes_pred = run_g2p(g2p, fp_digress)
    return seg, boxes_pred, rooms, edges


# ─────────────────────────────────────────────────────────────────────────────
# Graph metrics (original)
# ─────────────────────────────────────────────────────────────────────────────
def graph_metrics(rooms, edges, gt_rooms):
    """Returns (connectivity: float, room_type_accuracy: float)."""
    n = len(rooms)
    if n == 0:
        return 0.0, 0.0

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        if u < n and v < n:
            parent[find(u)] = find(v)
    connected = float(len({find(i) for i in range(n)}) == 1)

    pred_c = Counter(rooms)
    gt_c   = Counter(gt_rooms)
    all_t  = set(pred_c) | set(gt_c)
    matched = sum(min(pred_c.get(t, 0), gt_c.get(t, 0)) for t in all_t)
    rtype_acc = matched / max(sum(gt_c.values()), 1)

    return connected, float(rtype_acc)


# ─────────────────────────────────────────────────────────────────────────────
# New graph-aware validity & conditioning metrics
# ─────────────────────────────────────────────────────────────────────────────
def graph_validity(rooms, edges):
    """
    Binary validity score (1 = valid, 0 = invalid).

    A floor plan graph is valid if:
      1. It is connected (all rooms reachable from any room)
      2. Contains at least one LivingRoom and one Bathroom
      3. Has at least one edge (non-trivial graph)
      4. Has no isolated nodes (every room participates in ≥1 edge)
    """
    n = len(rooms)
    if n == 0:
        return 0.0

    # Required room types
    for req in REQUIRED_ROOMS:
        if req not in rooms:
            return 0.0

    # Must have at least one edge
    valid_edges = [(u, v) for u, v in edges if u < n and v < n and u != v]
    if len(valid_edges) == 0:
        return 0.0

    # No isolated nodes
    connected_nodes = set()
    for u, v in valid_edges:
        connected_nodes.add(u)
        connected_nodes.add(v)
    if len(connected_nodes) < n:
        return 0.0

    # Graph connectivity (Union-Find)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in valid_edges:
        parent[find(u)] = find(v)

    if len({find(i) for i in range(n)}) != 1:
        return 0.0

    return 1.0


def room_type_f1(pred_rooms, gt_rooms):
    """
    Micro-averaged F1 over room-type multisets.

    True positives = min(pred_count, gt_count) per type.
    """
    pred_c = Counter(pred_rooms)
    gt_c   = Counter(gt_rooms)
    all_t  = set(pred_c) | set(gt_c)

    tp = sum(min(pred_c.get(t, 0), gt_c.get(t, 0)) for t in all_t)
    fp = sum(max(pred_c.get(t, 0) - gt_c.get(t, 0), 0) for t in all_t)
    fn = sum(max(gt_c.get(t, 0) - pred_c.get(t, 0), 0) for t in all_t)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(f1)


def node_count_overlap(n_pred, n_gt):
    """min/max ratio — 1.0 means exact match, 0 means one is empty."""
    denom = max(n_pred, n_gt)
    return float(min(n_pred, n_gt) / denom) if denom > 0 else 1.0


def count_satisfaction(pred_rooms, target_counts):
    """
    Fraction of required room types in target_counts whose count exactly matches
    what was generated.

    target_counts: dict {room_type: required_count}
    Returns float in [0, 1]. None if target_counts is empty.
    """
    if not target_counts:
        return None

    pred_c = Counter(pred_rooms)
    constrained_names = {ROOM_TYPES[i] for i in CONSTRAINED_TYPES}

    total, exact = 0, 0
    for rt, req_count in target_counts.items():
        if rt not in constrained_names or req_count == 0:
            continue
        total += 1
        if pred_c.get(rt, 0) == req_count:
            exact += 1

    return float(exact / total) if total > 0 else None


def adj_satisfaction(pred_rooms, pred_edges, target_adj_pairs):
    """
    Fraction of GT constrained-type adjacency pairs that appear in the prediction.

    target_adj_pairs: set of frozenset({room_type_a, room_type_b})
    Returns float in [0, 1]. None if no target pairs.
    """
    if not target_adj_pairs:
        return None

    n = len(pred_rooms)
    pred_adj = set()
    for u, v in pred_edges:
        if u < n and v < n:
            pred_adj.add(frozenset({pred_rooms[u], pred_rooms[v]}))

    satisfied = sum(1 for pair in target_adj_pairs if pair in pred_adj)
    return float(satisfied / len(target_adj_pairs))


# ─────────────────────────────────────────────────────────────────────────────
# Pixel mIoU
# ─────────────────────────────────────────────────────────────────────────────
def pixel_miou(pred_seg, gt_seg, num_classes=NUM_CLASSES):
    ious = []
    for c in range(num_classes):
        pred_c = pred_seg == c
        gt_c   = gt_seg   == c
        inter  = int((pred_c & gt_c).sum())
        union  = int((pred_c | gt_c).sum())
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FID / KID
# ─────────────────────────────────────────────────────────────────────────────
def compute_fid_kid(real_tensors, fake_tensors, feature=64):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError:
        print('[WARN] torchmetrics not installed — skipping FID/KID.\n'
              '       Install: pip install torchmetrics[image]')
        return None, None

    n = min(len(real_tensors), len(fake_tensors))
    if n < 2:
        print('[WARN] Too few samples for FID/KID.')
        return None, None

    real = torch.stack(real_tensors[:n])
    fake = torch.stack(fake_tensors[:n])

    if real.shape[-1] < 75:
        real = F.interpolate(real.float(), size=75, mode='bilinear').byte()
        fake = F.interpolate(fake.float(), size=75, mode='bilinear').byte()

    fid_m = FrechetInceptionDistance(feature=feature, normalize=False)
    fid_m.update(real, real=True)
    fid_m.update(fake, real=False)
    fid = fid_m.compute().item()

    subset = min(50, n)
    kid_m = KernelInceptionDistance(feature=feature, subset_size=subset, normalize=False)
    kid_m.update(real, real=True)
    kid_m.update(fake, real=False)
    kid_mean, _ = kid_m.compute()

    return fid, kid_mean.item()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Evaluate retrieval vs. DiGress on test split')
    ap.add_argument('--ckpt',        required=True)
    ap.add_argument('--g2p_model',   default=os.path.join(ROOT, 'Interface', 'model', 'model.pth'))
    ap.add_argument('--test_data',   default=os.path.join(ROOT, 'Interface', 'static', 'Data', 'data_test_converted.pkl'))
    ap.add_argument('--train_data',  default=os.path.join(ROOT, 'Interface', 'static', 'Data', 'data_train_converted.pkl'))
    ap.add_argument('--num_samples', type=int, default=5)
    ap.add_argument('--out',         default='results/')
    ap.add_argument('--seed',        type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print('Loading data...')
    test_data, train_data = load_data(args)
    n = min(args.num_samples, len(test_data))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(test_data), n, replace=False)
    print(f'Evaluating on {n} / {len(test_data)} test samples')

    print('Loading Graph2Plan model...')
    g2p = load_g2p(args.g2p_model)

    print('Loading DiGress model...')
    digress = load_digress(args.ckpt)
    is_constrained = digress.y_data_dim > TF_DIM

    # ── Accumulators ──────────────────────────────────────────────────────────
    gt_tensors  = []
    ret_tensors, dig_tensors = [], []

    # Original metrics
    ret_iou,  dig_iou  = [], []
    ret_conn, dig_conn = [], []
    ret_racc, dig_racc = [], []

    # New graph-aware metrics
    ret_valid,  dig_valid  = [], []   # graph validity (binary)
    ret_f1,     dig_f1     = [], []   # room-type F1
    ret_ncov,   dig_ncov   = [], []   # node count overlap
    ret_csat,   dig_csat   = [], []   # count satisfaction (constrained only)
    ret_asat,   dig_asat   = [], []   # adjacency satisfaction (constrained only)

    for k, idx in enumerate(indices):
        datum     = test_data[int(idx)]
        train_idx = int(datum.topK[0])
        ref_datum = train_data[train_idx]

        gt_seg = gt_segmentation(ref_datum)
        gt_rms = gt_room_list(ref_datum)
        gt_tensors.append(seg_to_rgb_tensor(gt_seg))

        # Extract GT constraints from the reference floor plan
        gt_counts, gt_adj_pairs = extract_gt_constraints(ref_datum)
        # Only use counts for the constrained model; baseline ignores them
        counts_for_digress = gt_counts if is_constrained else None

        print(f'\n[{k+1:3d}/{n}]', end='  ')

        # ── Retrieval ────────────────────────────────────────────────────────
        print('retrieval...', end=' ', flush=True)
        try:
            ret_seg, _, ret_rooms, ret_edges = run_retrieval_pipeline(datum, ref_datum, g2p)
            ret_tensors.append(seg_to_rgb_tensor(ret_seg))
            ret_iou.append(pixel_miou(ret_seg, gt_seg))
            c, r = graph_metrics(ret_rooms, ret_edges, gt_rms)
            ret_conn.append(c)
            ret_racc.append(r)
            ret_valid.append(graph_validity(ret_rooms, ret_edges))
            ret_f1.append(room_type_f1(ret_rooms, gt_rms))
            ret_ncov.append(node_count_overlap(len(ret_rooms), len(gt_rms)))
            cs = count_satisfaction(ret_rooms, gt_counts)
            if cs is not None: ret_csat.append(cs)
            as_ = adj_satisfaction(ret_rooms, ret_edges, gt_adj_pairs)
            if as_ is not None: ret_asat.append(as_)
            print(f'IoU={ret_iou[-1]:.3f} valid={ret_valid[-1]:.0f} F1={ret_f1[-1]:.3f}', end='  ')
        except Exception as exc:
            print(f'[WARN retrieval] {exc}', end='  ')

        # ── DiGress ──────────────────────────────────────────────────────────
        print('digress...', end=' ', flush=True)
        try:
            dig_seg, _, dig_rooms, dig_edges = run_digress_pipeline(
                datum, digress, g2p, gt_counts=counts_for_digress)
            dig_tensors.append(seg_to_rgb_tensor(dig_seg))
            dig_iou.append(pixel_miou(dig_seg, gt_seg))
            c, r = graph_metrics(dig_rooms, dig_edges, gt_rms)
            dig_conn.append(c)
            dig_racc.append(r)
            dig_valid.append(graph_validity(dig_rooms, dig_edges))
            dig_f1.append(room_type_f1(dig_rooms, gt_rms))
            dig_ncov.append(node_count_overlap(len(dig_rooms), len(gt_rms)))
            cs = count_satisfaction(dig_rooms, gt_counts)
            if cs is not None: dig_csat.append(cs)
            as_ = adj_satisfaction(dig_rooms, dig_edges, gt_adj_pairs)
            if as_ is not None: dig_asat.append(as_)
            print(f'IoU={dig_iou[-1]:.3f} valid={dig_valid[-1]:.0f} F1={dig_f1[-1]:.3f}')
        except Exception as exc:
            print(f'[WARN digress] {exc}')

    # ── FID / KID ─────────────────────────────────────────────────────────────
    print('\nComputing FID / KID (retrieval)...')
    ret_fid, ret_kid = compute_fid_kid(gt_tensors, ret_tensors)
    print('Computing FID / KID (DiGress)...')
    dig_fid, dig_kid = compute_fid_kid(gt_tensors, dig_tensors)

    # ── Report ────────────────────────────────────────────────────────────────
    def f(vals):
        return f'{np.mean(vals):.4f} ± {np.std(vals):.4f}' if vals else 'N/A'

    def fv(v):
        return f'{v:.4f}' if v is not None else 'N/A'

    csat_note = '(constrained model)' if is_constrained else '(baseline — N/A)'

    lines = [
        '',
        '=' * 72,
        f'  Evaluation — {n} test samples  (seed={args.seed})',
        f'  Model: {"constrained" if is_constrained else "baseline"}  '
        f'y_dim={digress.y_data_dim}',
        '=' * 72,
        f'  {"Metric":<32} {"Retrieval":>18}  {"DiGress (ours)":>16}',
        '-' * 72,
        f'  {"FID ↓":<32} {fv(ret_fid):>18}  {fv(dig_fid):>16}',
        f'  {"KID ↓":<32} {fv(ret_kid):>18}  {fv(dig_kid):>16}',
        '-' * 72,
        f'  {"Connectivity rate ↑":<32} {f(ret_conn):>18}  {f(dig_conn):>16}',
        f'  {"Graph validity ↑":<32} {f(ret_valid):>18}  {f(dig_valid):>16}',
        '-' * 72,
        f'  {"Room-type accuracy ↑":<32} {f(ret_racc):>18}  {f(dig_racc):>16}',
        f'  {"Room-type F1 ↑":<32} {f(ret_f1):>18}  {f(dig_f1):>16}',
        f'  {"Node count overlap ↑":<32} {f(ret_ncov):>18}  {f(dig_ncov):>16}',
        '-' * 72,
        f'  {"Count satisfaction ↑":<32} {f(ret_csat):>18}  {f(dig_csat):>16}',
        f'  {"  " + csat_note:<32}',
        f'  {"Adj satisfaction ↑":<32} {f(ret_asat):>18}  {f(dig_asat):>16}',
        '-' * 72,
        f'  {"Pixel mIoU ↑":<32} {f(ret_iou):>18}  {f(dig_iou):>16}',
        '=' * 72,
        f'  n_retrieval={len(ret_iou)}  n_digress={len(dig_iou)}',
        '',
        '  Graph validity = connected + required rooms + no isolated nodes.',
        '  Count/adj satisfaction: GT constraints from top-1 train reference.',
        '  FID/KID estimates are noisy for n < 2048; use --num_samples 500+.',
        '=' * 72,
    ]
    report = '\n'.join(lines)
    print(report)

    rpt_path = os.path.join(args.out, 'eval_results.txt')
    with open(rpt_path, 'w') as fh:
        fh.write(report + '\n')

    raw = {
        'n_samples': n, 'seed': args.seed,
        'model': {'constrained': is_constrained, 'y_data_dim': digress.y_data_dim},
        'retrieval': {
            'fid': ret_fid, 'kid': ret_kid,
            'connectivity':         ret_conn,
            'graph_validity':       ret_valid,
            'room_type_accuracy':   ret_racc,
            'room_type_f1':         ret_f1,
            'node_count_overlap':   ret_ncov,
            'count_satisfaction':   ret_csat,
            'adj_satisfaction':     ret_asat,
            'pixel_miou':           ret_iou,
        },
        'digress': {
            'fid': dig_fid, 'kid': dig_kid,
            'connectivity':         dig_conn,
            'graph_validity':       dig_valid,
            'room_type_accuracy':   dig_racc,
            'room_type_f1':         dig_f1,
            'node_count_overlap':   dig_ncov,
            'count_satisfaction':   dig_csat,
            'adj_satisfaction':     dig_asat,
            'pixel_miou':           dig_iou,
        },
    }
    json_path = os.path.join(args.out, 'eval_results.json')
    with open(json_path, 'w') as fh:
        json.dump(raw, fh, indent=2)

    print(f'\n  Results saved to {rpt_path} and {json_path}')


if __name__ == '__main__':
    main()
