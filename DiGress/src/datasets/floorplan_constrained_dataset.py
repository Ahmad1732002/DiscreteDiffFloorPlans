import os
import pathlib
import pickle

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from datasets.floorplan_dataset import (
    ROOM_TYPES, NUM_ROOM_TYPES, GRID_SIZE, NUM_LOC, NUM_SIZE,
    NODE_FEAT_DIM, NUM_EDGE_TYPES, SPLIT_FRACS, SPLIT_SEED,
    _load_pkl, _get_split_indices, _encode_node_features,
)

# ── Constraint encoding (Graph2Plan Section 4) ────────────────────────────────
# 5 room types that support location + adjacency constraints
CONSTRAINED_TYPES = [0, 1, 2, 3, 4]   # Living, Master, Kitchen, Bathroom, Dining
NUM_CONSTRAINED   = len(CONSTRAINED_TYPES)

# Bedroom cluster: all bedroom-like types merged into one count
BEDROOM_TYPES = {1, 5, 6, 7, 8}   # Master, Child, Study, Second, Guest

# y vector layout:
#   [0   : 1000] TF boundary descriptor
#   [1000: 1013] per-type room counts (13)
#   [1013: 1014] bedroom cluster count (1)
#   [1014: 1139] location masks 5×25 (125)
#   [1139: 1164] adjacency 5×5 (25)
# Total y dim: 1164
TF_DIM         = 1000
COUNT_DIM      = NUM_ROOM_TYPES + 1   # 14
LOC_DIM        = NUM_CONSTRAINED * NUM_LOC   # 125
ADJ_DIM        = NUM_CONSTRAINED * NUM_CONSTRAINED  # 25
CONSTRAINT_DIM = COUNT_DIM + LOC_DIM + ADJ_DIM  # 164
Y_DIM          = TF_DIM + CONSTRAINT_DIM  # 1164
# ──────────────────────────────────────────────────────────────────────────────


def _encode_constraints(boxes, edges, boundary):
    """
    Encode layout constraints as described in Graph2Plan Section 4:
      - Room type counts (13) + bedroom cluster (1)          = 14 dims
      - Location masks for 5 constrained types × 25 cells    = 125 dims
      - Adjacency matrix between 5 constrained types (5×5)   = 25 dims
    Returns: (164,) float32 array
    """
    # Room counts
    counts = np.zeros(COUNT_DIM, dtype=np.float32)
    for box in boxes:
        rt = int(box[4])
        if 0 <= rt < NUM_ROOM_TYPES:
            counts[rt] += 1.0
            if rt in BEDROOM_TYPES:
                counts[NUM_ROOM_TYPES] += 1.0

    # Boundary bbox for normalisation
    ext = boundary[:, :2]
    bx0, bx1 = ext[:, 0].min(), ext[:, 0].max()
    by0, by1 = ext[:, 1].min(), ext[:, 1].max()
    bw = max(float(bx1 - bx0), 1.0)
    bh = max(float(by1 - by0), 1.0)

    gbins = np.linspace(0, 1, GRID_SIZE + 1)
    gbins[0], gbins[-1] = -np.inf, np.inf

    # Location masks: for each of 5 constrained types, which grid cells it occupies
    loc_masks = np.zeros((NUM_CONSTRAINED, NUM_LOC), dtype=np.float32)
    for box in boxes:
        rt = int(box[4])
        if rt not in CONSTRAINED_TYPES:
            continue
        ti = CONSTRAINED_TYPES.index(rt)
        x0, y0, x1, y1 = box[:4]
        cx_norm = (((x0 + x1) / 2.0) - bx0) / bw
        cy_norm = (((y0 + y1) / 2.0) - by0) / bh
        cx_bin  = int(np.digitize(cx_norm, gbins)) - 1
        cy_bin  = int(np.digitize(cy_norm, gbins)) - 1
        loc_masks[ti, cx_bin * GRID_SIZE + cy_bin] = 1.0

    # Adjacency between constrained types
    adj = np.zeros((NUM_CONSTRAINED, NUM_CONSTRAINED), dtype=np.float32)
    room_rts = {i: int(boxes[i, 4]) for i in range(len(boxes))}
    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        if u >= len(boxes) or v >= len(boxes):
            continue
        rt_u = room_rts.get(u, -1)
        rt_v = room_rts.get(v, -1)
        if rt_u in CONSTRAINED_TYPES and rt_v in CONSTRAINED_TYPES:
            i = CONSTRAINED_TYPES.index(rt_u)
            j = CONSTRAINED_TYPES.index(rt_v)
            adj[i, j] = adj[j, i] = 1.0

    return np.concatenate([counts, loc_masks.flatten(), adj.flatten()])


def _build_pyg_constrained(floor_plan, tf_vec=None):
    """
    Like _build_pyg but y = [TF(1000) | constraints(164)] = 1164 dims.
    constraints = [room counts(14) | location masks(125) | adjacency(25)]
    """
    boxes    = floor_plan.box
    edges    = floor_plan.edge
    boundary = floor_plan.boundary

    valid_rooms = boxes[:, 4].astype(int) < NUM_ROOM_TYPES
    boxes       = boxes[valid_rooms]
    old_to_new  = np.cumsum(valid_rooms) - 1
    num_rooms   = len(boxes)

    if num_rooms == 0:
        raise ValueError('No valid rooms after filtering External type')

    x = torch.tensor(_encode_node_features(boxes, boundary), dtype=torch.float)

    if len(edges) > 0:
        src = old_to_new[edges[:, 0].astype(int)]
        dst = old_to_new[edges[:, 1].astype(int)]
        valid_e = (edges[:, 0].astype(int) < len(valid_rooms)) & \
                  valid_rooms[edges[:, 0].astype(int)] & \
                  valid_rooms[edges[:, 1].astype(int)]
        src, dst = src[valid_e], dst[valid_e]
        valid_e2 = (src < num_rooms) & (dst < num_rooms)
        src, dst = src[valid_e2], dst[valid_e2]
        if len(src) > 0:
            edge_index = torch.tensor(
                np.stack([np.concatenate([src, dst]),
                          np.concatenate([dst, src])]),
                dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    num_edges = edge_index.shape[1]
    edge_attr = torch.zeros(num_edges, NUM_EDGE_TYPES, dtype=torch.float)
    if num_edges > 0:
        edge_attr[:, 1] = 1.0

    constraint_vec = _encode_constraints(boxes, edges, boundary)
    tf_part = tf_vec if tf_vec is not None else np.zeros(TF_DIM, dtype=np.float32)
    y = torch.tensor(np.concatenate([tf_part, constraint_vec]),
                     dtype=torch.float).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=y, n_nodes=torch.tensor([num_rooms], dtype=torch.long))


class FloorplanConstrainedDataset(InMemoryDataset):
    def __init__(self, split, root, use_tf_conditioning=True,
                 transform=None, pre_transform=None, pre_filter=None):
        assert split in ('train', 'val', 'test')
        self.split               = split
        self.use_tf_conditioning = use_tf_conditioning
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data_train_converted.pkl', 'tf_train.npy']

    @property
    def processed_file_names(self):
        return [f'floorplan_constrained_{self.split}.pt']

    def download(self):
        raise RuntimeError(
            f"Copy data_train_converted.pkl and tf_train.npy into {self.raw_dir}"
        )

    def process(self):
        all_data = _load_pkl(
            os.path.join(self.raw_dir, 'data_train_converted.pkl'))['data']
        tf_all = np.load(os.path.join(self.raw_dir, 'tf_train.npy')) \
            if self.use_tf_conditioning else None

        idx         = _get_split_indices(len(all_data))[self.split]
        floor_plans = [all_data[i] for i in idx]
        tfs         = tf_all[idx] if tf_all is not None else None

        data_list = []
        for i, fp in enumerate(floor_plans):
            try:
                data = _build_pyg_constrained(fp, tfs[i] if tfs is not None else None)
            except Exception:
                continue
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        print(f'[FloorplanConstrainedDataset] {self.split}: {len(data_list)} graphs')
        torch.save(self.collate(data_list), self.processed_paths[0])


class FloorplanConstrainedDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg   = cfg
        base_path  = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path  = os.path.join(base_path, cfg.dataset.datadir)
        use_tf     = getattr(cfg.dataset, 'use_tf_conditioning', True)

        datasets = {
            'train': FloorplanConstrainedDataset('train', root_path, use_tf),
            'val':   FloorplanConstrainedDataset('val',   root_path, use_tf),
            'test':  FloorplanConstrainedDataset('test',  root_path, use_tf),
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class FloorplanConstrainedDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.name        = 'floorplan_constrained'
        self.n_nodes     = datamodule.node_counts()
        self.node_types  = datamodule.node_types()
        self.edge_types  = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
