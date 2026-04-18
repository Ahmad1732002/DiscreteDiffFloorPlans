"""Generate tf_train.npy from data_train_converted.pkl."""
import argparse
import pickle
import sys
import numpy as np

sys.path.insert(0, '/app/DiGress/src')


def compute_tf(b):
    if b.shape[1] > 2:
        b = b[:, :2]
    b = np.concatenate((b, b[:1]))
    n = len(b) - 1
    v = b[1:] - b[:-1]
    L = np.linalg.norm(v, axis=1)
    perim = L.sum()
    v = v / perim
    L = L / perim
    angles = np.zeros(n)
    for i in range(n):
        z = np.cross(v[i], v[(i + 1) % n])
        angles[i] = np.arccos(np.clip(np.dot(v[i], v[(i + 1) % n]), -1, 1)) * np.sign(z)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    s = 0
    for i in range(1, n + 1):
        x[i] = L[i - 1] + x[i - 1]
        y[i - 1] = angles[i - 1] + s
        s = y[i - 1]
    y[-1] = s
    return x, y


def sample_tf(x, y, ndim=1000):
    t = np.linspace(0, 1, ndim)
    return np.piecewise(t, [t >= xx for xx in x], y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    train_data = data['data']
    print(f'Computing TF for {len(train_data)} samples ...')

    tf_list = []
    for i, d in enumerate(train_data):
        x, y = compute_tf(d.boundary)
        tf_list.append(sample_tf(x, y, 1000))
        if i % 5000 == 0:
            print(f'  {i}/{len(train_data)}')

    tf_train = np.array(tf_list)
    np.save(args.out, tf_train)
    print(f'Saved {args.out} with shape {tf_train.shape}')
