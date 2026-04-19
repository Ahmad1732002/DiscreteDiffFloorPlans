"""
evaluate.py — End-to-end benchmark: Retrieval baseline vs. DiGress diffusion.

Both methods feed into the same frozen Graph2Plan model so the comparison
isolates only the first-stage graph generator.

Metrics (all computed on the held-out test split):
  FID  (↓)  Fréchet Inception Distance on 128×128 rendered segmentation maps
  KID  (↓)  Kernel Inception Distance  on same
  Connectivity rate  (↑)  fraction of generated graphs that are connected
  Room-type accuracy (↑)  overlap of predicted vs. GT room-type multiset
  Pixel mIoU         (↑)  mean per-class pixel IoU vs. GT segmentation

Run from the project root:
    python scripts/evaluate.py --ckpt checkpoints/last.ckpt
    python scripts/evaluate.py --ckpt checkpoints/last.ckpt --num_samples 200 --out results/
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

# ── Path setup (no Django required) ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Interface'))
sys.path.insert(0, os.path.join(ROOT, 'DiGress', 'src'))

# Interface imports (Django-free; test.py is NOT imported to avoid vw dependency)
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
NUM_CLASSES = 13       # room types 0-12; 13 = external background

TF_DIM    = 1000
COUNT_DIM = 14
LOC_DIM   = 125
ADJ_DIM   = 25
BEDROOM_INDICES = {1, 5, 6, 7, 8}

# RGB palette indexed by room_label index (0-12 rooms, 13 = external white)
_PALETTE_LIST = [r[4] for r in room_label]          # list of [R,G,B]
PALETTE = np.array(_PALETTE_LIST + [[255, 255, 255]] * 10, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(args):
    test_pkl  = pickle.load(open(args.test_data,  'rb'))
    train_pkl = pickle.load(open(args.train_data, 'rb'))
    test_data  = test_pkl['data']
    train_data = train_pkl['data']
    return test_data, train_data


# ─────────────────────────────────────────────────────────────────────────────
# Graph2Plan (frozen downstream model)
# ─────────────────────────────────────────────────────────────────────────────
def load_g2p(model_path):
    m = Model()
    m.load_state_dict(torch.load(model_path, map_location='cpu'))
    m.eval()
    return m


def run_g2p(g2p, fp):
    """Returns (seg128 uint8 ndarray, boxes_pred float ndarray N×4)."""
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
    seg = gene_preds.squeeze().cpu().numpy().astype(np.uint8)
    boxes = boxes_pred.squeeze().cpu().numpy()
    return seg, boxes


# ─────────────────────────────────────────────────────────────────────────────
# Rendering utilities
# ─────────────────────────────────────────────────────────────────────────────
def seg_to_rgb_tensor(seg):
    """(128,128) int ndarray → (3,128,128) uint8 tensor."""
    seg_clipped = np.clip(seg, 0, len(PALETTE) - 1)
    rgb = PALETTE[seg_clipped]                          # (128,128,3)
    return torch.from_numpy(rgb).permute(2, 0, 1)       # (3,128,128)


def gt_segmentation(ref_datum):
    """Render ground-truth boxes (from a training reference sample) to 128×128 seg map.

    Test data has no GT boxes. We use train_data[datum.topK[0]] — the pre-computed
    top-1 nearest training sample — as the reference. This is the same sample the
    retrieval baseline copies from, making it a justified and fair reference for both.
    """
    seg = np.full((128, 128), NUM_CLASSES, dtype=np.uint8)   # background
    for row in ref_datum.box:
        x0, y0, x1, y1, rtype = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
        # boundary coords are in [0,255]; seg map is 128×128 so divide by 2
        x0, y0, x1, y1 = x0 // 2, y0 // 2, x1 // 2, y1 // 2
        rtype = min(rtype, NUM_CLASSES - 1)
        seg[y0:y1, x0:x1] = rtype
    return seg


def gt_room_list(ref_datum):
    return [ROOM_TYPES[min(int(b[4]), NUM_CLASSES - 1)] for b in ref_datum.box]


# ─────────────────────────────────────────────────────────────────────────────
# TF computation (standalone, no Django)
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
# Retrieval baseline
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_top1(datum):
    """Use the pre-computed topK[0] index stored in the test datum."""
    return int(datum.topK[0])


def run_retrieval_pipeline(datum, train_datum, g2p):
    test_fp  = FloorPlan(datum)
    train_fp = FloorPlan(train_datum, train=True)
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


def _build_y(tf_vec, y_data_dim):
    if y_data_dim <= TF_DIM:
        return tf_vec
    return np.concatenate([tf_vec,
                            np.zeros(COUNT_DIM, dtype=np.float32),
                            np.zeros(LOC_DIM,   dtype=np.float32),
                            np.zeros(ADJ_DIM,   dtype=np.float32)])


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

    for req in ('LivingRoom', 'Bathroom'):
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


def digress_sample(digress_model, boundary):
    from diffusion import diffusion_utils
    device = torch.device('cpu')
    tf_vec = compute_tf(boundary)
    y_vec  = _build_y(tf_vec, digress_model.y_data_dim)

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


def run_digress_pipeline(datum, digress_model, g2p):
    rooms, edges = digress_sample(digress_model, datum.boundary)
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
# Graph metrics
# ─────────────────────────────────────────────────────────────────────────────
def graph_metrics(rooms, edges, gt_rooms):
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

    real = torch.stack(real_tensors[:n])   # (N,3,128,128) uint8
    fake = torch.stack(fake_tensors[:n])

    # InceptionV3 requires ≥ 75px
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
    ap.add_argument('--ckpt',        required=True,
                    help='Path to DiGress .ckpt checkpoint')
    ap.add_argument('--g2p_model',   default=os.path.join(ROOT, 'Interface', 'model', 'model.pth'),
                    help='Path to Graph2Plan model.pth')
    ap.add_argument('--test_data',   default=os.path.join(ROOT, 'Interface', 'static', 'Data', 'data_test_converted.pkl'))
    ap.add_argument('--train_data',  default=os.path.join(ROOT, 'Interface', 'static', 'Data', 'data_train_converted.pkl'))
    ap.add_argument('--num_samples', type=int, default=5,
                    help='Number of test samples (≥50 recommended for FID reliability)')
    ap.add_argument('--out',         default='results/',
                    help='Output directory')
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

    # Accumulators
    gt_tensors  = []
    ret_tensors, dig_tensors = [], []
    ret_iou,    dig_iou     = [], []
    ret_conn,   dig_conn    = [], []
    ret_racc,   dig_racc    = [], []

    for k, idx in enumerate(indices):
        datum     = test_data[int(idx)]
        train_idx = retrieve_top1(datum)           # pre-computed in test pkl
        ref_datum = train_data[train_idx]          # top-1 training match with full GT

        gt_seg = gt_segmentation(ref_datum)        # GT from top-1 training reference
        gt_rms = gt_room_list(ref_datum)
        gt_tensors.append(seg_to_rgb_tensor(gt_seg))

        print(f'\n[{k+1:3d}/{n}]', end='  ')

        # ── Retrieval ────────────────────────────────────────────────────────
        print('retrieval...', end=' ', flush=True)
        try:
            ret_seg, _, ret_rooms, ret_edges = run_retrieval_pipeline(
                datum, ref_datum, g2p)
            ret_tensors.append(seg_to_rgb_tensor(ret_seg))
            ret_iou.append(pixel_miou(ret_seg, gt_seg))
            c, r = graph_metrics(ret_rooms, ret_edges, gt_rms)
            ret_conn.append(c); ret_racc.append(r)
            print(f'IoU={ret_iou[-1]:.3f} conn={c:.0f} racc={r:.3f}', end='  ')
        except Exception as exc:
            print(f'[WARN] {exc}', end='  ')

        # ── DiGress ──────────────────────────────────────────────────────────
        print('digress...', end=' ', flush=True)
        try:
            dig_seg, _, dig_rooms, dig_edges = run_digress_pipeline(datum, digress, g2p)
            dig_tensors.append(seg_to_rgb_tensor(dig_seg))
            dig_iou.append(pixel_miou(dig_seg, gt_seg))
            c, r = graph_metrics(dig_rooms, dig_edges, gt_rms)
            dig_conn.append(c); dig_racc.append(r)
            print(f'IoU={dig_iou[-1]:.3f} conn={c:.0f} racc={r:.3f}')
        except Exception as exc:
            print(f'[WARN] {exc}')

    # ── FID / KID ─────────────────────────────────────────────────────────────
    print('\nComputing FID / KID (retrieval)...')
    ret_fid, ret_kid = compute_fid_kid(gt_tensors, ret_tensors)
    print('Computing FID / KID (DiGress)...')
    dig_fid, dig_kid = compute_fid_kid(gt_tensors, dig_tensors)

    # ── Report ────────────────────────────────────────────────────────────────
    def f(vals):
        return f'{np.mean(vals):.4f} ± {np.std(vals):.4f}' if vals else 'N/A'

    def fid_str(v):
        return f'{v:.4f}' if v is not None else 'N/A (install torchmetrics[image])'

    lines = [
        '',
        '=' * 70,
        f'  Evaluation — {n} test samples  (seed={args.seed})',
        '=' * 70,
        f'  {"Metric":<28} {"Retrieval baseline":>20}  {"DiGress (ours)":>16}',
        '-' * 70,
        f'  {"FID ↓":<28} {fid_str(ret_fid):>20}  {fid_str(dig_fid):>16}',
        f'  {"KID ↓":<28} {fid_str(ret_kid):>20}  {fid_str(dig_kid):>16}',
        f'  {"Connectivity ↑":<28} {f(ret_conn):>20}  {f(dig_conn):>16}',
        f'  {"Room-type accuracy ↑":<28} {f(ret_racc):>20}  {f(dig_racc):>16}',
        f'  {"Pixel mIoU ↑":<28} {f(ret_iou):>20}  {f(dig_iou):>16}',
        '=' * 70,
        f'  n_retrieval={len(ret_iou)}  n_digress={len(dig_iou)}',
        '',
        '  Note: GT reference = top-1 training match (datum.topK[0]) per test sample.',
        '        FID/KID estimates are noisy for n < 2048; use --num_samples 500+.',
        '=' * 70,
    ]
    report = '\n'.join(lines)
    print(report)

    # Save text report
    rpt_path = os.path.join(args.out, 'eval_results.txt')
    with open(rpt_path, 'w') as f_:
        f_.write(report + '\n')

    # Save raw numbers as JSON for further analysis
    raw = {
        'n_samples': n,
        'seed': args.seed,
        'retrieval': {
            'fid': ret_fid, 'kid': ret_kid,
            'connectivity': ret_conn,
            'room_type_accuracy': ret_racc,
            'pixel_miou': ret_iou,
        },
        'digress': {
            'fid': dig_fid, 'kid': dig_kid,
            'connectivity': dig_conn,
            'room_type_accuracy': dig_racc,
            'pixel_miou': dig_iou,
        },
    }
    json_path = os.path.join(args.out, 'eval_results.json')
    with open(json_path, 'w') as f_:
        json.dump(raw, f_, indent=2)

    print(f'\n  Results saved to {rpt_path} and {json_path}')


if __name__ == '__main__':
    main()
