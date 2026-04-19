"""
Generate data_test_converted.pkl by matching the images in static/Data/Img/
against data_train_converted.pkl using the nameList IDs.

Usage:
    python scripts/generate_test_pkl.py \
        --train Interface/static/Data/data_train_converted.pkl \
        --img   Interface/static/Data/Img \
        --out   Interface/static/Data/data_test_converted.pkl
"""

import argparse
import os
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--img',   required=True,
                        help='Directory containing test boundary images (*.png)')
    parser.add_argument('--out',   required=True)
    args = parser.parse_args()

    img_ids = {os.path.splitext(f)[0] for f in os.listdir(args.img) if f.endswith('.png')}
    print(f'Images in Img folder: {len(img_ids)}')

    print(f'Loading {args.train} ...')
    with open(args.train, 'rb') as f:
        train_pkl = pickle.load(f)

    all_data  = train_pkl['data']
    all_names = [str(n) for n in train_pkl['nameList']]
    print(f'Total samples in train pkl: {len(all_names)}')

    test_data  = []
    test_names = []
    for i, name in enumerate(all_names):
        if name in img_ids:
            test_data.append(all_data[i])
            test_names.append(name)

    print(f'Matched test samples: {len(test_data)}')

    train_names = [n for n in all_names if n not in img_ids]

    out_pkl = {
        'data':          test_data,
        'testNameList':  test_names,
        'trainNameList': train_names,
    }

    with open(args.out, 'wb') as f:
        pickle.dump(out_pkl, f)

    print(f'Done. Written to {args.out}')


if __name__ == '__main__':
    main()
