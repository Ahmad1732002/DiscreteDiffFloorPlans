import os
import pathlib
import pickle

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# 13 room types used in Graph2Plan layout graphs (External/index-13 is not a room node)
ROOM_TYPES = [
    'LivingRoom', 'MasterRoom', 'Kitchen', 'Bathroom',
    'DiningRoom', 'ChildRoom', 'StudyRoom', 'SecondRoom',
    'GuestRoom', 'Balcony', 'Entrance', 'Storage', 'Wall'
]
NUM_ROOM_TYPES = len(ROOM_TYPES)   # 13

# Location: 5x5 grid over boundary bbox (Section 4.1, K=5)
GRID_SIZE  = 5
NUM_LOC    = GRID_SIZE * GRID_SIZE  # 25

# Size: 10 area-ratio bins (Section 5.1, d3=10)
NUM_SIZE   = 10

# Node feature layout matching Graph2Plan's get_attributes():
#   [room_type (13) | location 5x5 (25) | size (10)] = 48 dims
NODE_FEAT_DIM = NUM_ROOM_TYPES + NUM_LOC + NUM_SIZE  # 48

# Edge types: 0=no edge, 1-10=spatial relations matching Graph2Plan's 10 types
# For diffusion we use binary (present / not present)
NUM_EDGE_TYPES = 2   # 0=no edge, 1=adjacent

# 70/15/15 split confirmed in Graph2Plan paper Section 6.1
SPLIT_FRACS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SPLIT_SEED  = 42


def _load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _get_split_indices(n):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(n)
    n_train = int(n * SPLIT_FRACS['train'])
    n_val   = int(n * SPLIT_FRACS['val'])
    return {
        'train': idx[:n_train],
        'val':   idx[n_train:n_train + n_val],
        'test':  idx[n_train + n_val:],
    }


def _encode_node_features(boxes, boundary):
    """
    Encode room nodes as [room_type (13) | location 5x5 (25) | size (10)],
    matching exactly the node representation in Graph2Plan (Section 5.1).

    boxes:    (M, 5)  x0 y0 x1 y1 room_type  (0-255 pixel coords)
    boundary: (N, 4)  x y dir isNew
    Returns:  (M, 48) float tensor
    """
    ext  = boundary[:, :2]
    bx0, bx1 = ext[:, 0].min(), ext[:, 0].max()
    by0, by1 = ext[:, 1].min(), ext[:, 1].max()
    bw   = max(float(bx1 - bx0), 1.0)
    bh   = max(float(by1 - by0), 1.0)
    area = bw * bh

    M    = len(boxes)
    feat = np.zeros((M, NODE_FEAT_DIM), dtype=np.float32)

    gbins = np.linspace(0, 1, GRID_SIZE + 1)
    gbins[0], gbins[-1] = -np.inf, np.inf

    abins = np.linspace(0, 1, NUM_SIZE + 1)
    abins[0], abins[-1] = -np.inf, np.inf

    for i, box in enumerate(boxes):
        x0, y0, x1, y1, rt = box
        rt = int(rt)

        # Room type (skip External = index 13)
        if 0 <= rt < NUM_ROOM_TYPES:
            feat[i, rt] = 1.0

        # Normalized center relative to boundary bbox
        cx_norm = (((x0 + x1) / 2.0) - bx0) / bw
        cy_norm = (((y0 + y1) / 2.0) - by0) / bh

        # 5x5 grid location (row-major: cx_bin * 5 + cy_bin)
        cx_bin  = int(np.digitize(cx_norm, gbins)) - 1
        cy_bin  = int(np.digitize(cy_norm, gbins)) - 1
        loc_idx = NUM_ROOM_TYPES + cx_bin * GRID_SIZE + cy_bin
        feat[i, loc_idx] = 1.0

        # Area ratio bin
        room_area = max(float((x1 - x0) * (y1 - y0)) / area, 0.0)
        size_idx  = NUM_ROOM_TYPES + NUM_LOC + (int(np.digitize(room_area, abins)) - 1)
        feat[i, size_idx] = 1.0

    return feat


def _build_pyg(floor_plan, tf_vec=None):
    """
    Convert one FloorPlan object to a PyG Data object.

    Node features (48-dim) match Graph2Plan's node encoding exactly:
        [room_type (13) | 5x5 location (25) | area size (10)]

    Edge attr (2-dim): binary [no_edge, adjacent]

    y (1, 1000): boundary TF descriptor for conditioning
    """
    boxes    = floor_plan.box       # (M, 5): x0 y0 x1 y1 room_type
    edges    = floor_plan.edge      # (E, 3): u v relation
    boundary = floor_plan.boundary  # (N, 4): x y dir isNew

    # Filter out External rooms (type 13) — not part of layout graph
    valid_rooms = boxes[:, 4].astype(int) < NUM_ROOM_TYPES
    boxes       = boxes[valid_rooms]
    # Remap edge indices after filtering
    old_to_new  = np.cumsum(valid_rooms) - 1
    num_rooms   = len(boxes)

    if num_rooms == 0:
        raise ValueError('No valid rooms after filtering External type')

    x = torch.tensor(_encode_node_features(boxes, boundary), dtype=torch.float)

    # Build bidirectional edges with remapped indices
    if len(edges) > 0:
        src = old_to_new[edges[:, 0].astype(int)]
        dst = old_to_new[edges[:, 1].astype(int)]
        # keep only edges where both endpoints are valid rooms
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

    # Conditioning: 1000-dim TF boundary descriptor
    y = torch.tensor(tf_vec, dtype=torch.float).unsqueeze(0) \
        if tf_vec is not None else torch.zeros((1, 0), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=y, n_nodes=torch.tensor([num_rooms], dtype=torch.long))


class FloorplanDataset(InMemoryDataset):
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
        return [f'floorplan_{self.split}.pt']

    def download(self):
        raise RuntimeError(
            f"Copy data_train_converted.pkl and tf_train.npy into {self.raw_dir}"
        )

    def process(self):
        # All 74995 samples have full layouts. Split 70/15/15 — same indices
        # used by both DiGress and the retrieval baseline for a fair comparison.
        all_data = _load_pkl(
            os.path.join(self.raw_dir, 'data_train_converted.pkl'))['data']
        tf_all = np.load(os.path.join(self.raw_dir, 'tf_train.npy')) \
            if self.use_tf_conditioning else None

        idx         = _get_split_indices(len(all_data))[self.split]
        # Support both list and numpy-array storage formats
        floor_plans = [all_data[i] for i in idx]
        tfs         = tf_all[idx] if tf_all is not None else None

        data_list = []
        for i, fp in enumerate(floor_plans):
            try:
                data = _build_pyg(fp, tfs[i] if tfs is not None else None)
            except Exception:
                continue
            if self.pre_filter  is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        print(f'[FloorplanDataset] {self.split}: {len(data_list)} graphs')
        torch.save(self.collate(data_list), self.processed_paths[0])


class FloorplanDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg   = cfg
        base_path  = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path  = os.path.join(base_path, cfg.dataset.datadir)
        use_tf     = getattr(cfg.dataset, 'use_tf_conditioning', True)

        datasets = {
            'train': FloorplanDataset('train', root_path, use_tf),
            'val':   FloorplanDataset('val',   root_path, use_tf),
            'test':  FloorplanDataset('test',  root_path, use_tf),
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class FloorplanDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.name        = 'floorplan'
        self.n_nodes     = datamodule.node_counts()
        self.node_types  = datamodule.node_types()
        self.edge_types  = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
