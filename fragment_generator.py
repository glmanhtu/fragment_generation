import argparse
import glob
import json
import os
import random
from typing import List

import cv2
import numpy as np
import tqdm
from torch.utils.data import Dataset

from utils.fragmentize import FragmentizeStrategy, Fragment1v1Rotate90, Fragment1v05Rotate90
from utils.utils import seed_everything, compute_white_percentage, visualise_fragments


class ImageData(Dataset):
    def __init__(self, dataset_dir, output_dir, fragmentize_strategies: List[FragmentizeStrategy]):
        self.dataset = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True))
        self.working_dir = output_dir
        self.fragmentize_strategies = fragmentize_strategies
        self.min_size = 224

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        np_img = cv2.imread(img_path)
        h, w = np_img.shape[:2]
        if h < self.min_size or w < self.min_size:
            return {}
        results = {}
        for fragmentize_strategy in self.fragmentize_strategies:
            images, labels = fragmentize_strategy.split(np_img)
            visualise_fragments(images, labels, degree=0).show()
            patch_dir = os.path.join(self.working_dir, img_name, fragmentize_strategy.name())
            if len(images) == 0:
                continue

            os.makedirs(patch_dir, exist_ok=True)

            # Combine the two lists into a list of tuples using zip
            combined_lists = list(zip(images, labels))

            # Shuffle the combined list
            random.shuffle(combined_lists)
            for idx, (img, label) in enumerate(combined_lists):
                file_path = os.path.join(patch_dir, f'{idx}.jpg')
                cv2.imwrite(file_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                label['im_path'] = file_path.replace(self.working_dir + os.path.sep, '')
                label['white_percentage'] = round(compute_white_percentage(img), 3)

            random.shuffle(labels)
            results.setdefault(img_name, {})[fragmentize_strategy.name()] = labels
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dataset generator", add_help=True)
    parser.add_argument("--dataset-dir", required=True, help="Path to cleaned papyrus images dataset")
    parser.add_argument("--edges-json", required=True, help="Path to edges json file")
    parser.add_argument("--n-workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output dataset",
    )

    args = parser.parse_args()
    seed_everything(12345)

    with open(args.edges_json) as f:
        edge_def = json.load(f)
    edges_dir = os.path.dirname(args.edges_json)
    edges = edge_def['edges']
    for item in edges:
        item['img_path'] = os.path.join(edges_dir, item['img_path'])

    fragmentize_strategies = [
        Fragment1v1Rotate90(edges, args.patch_size),
        Fragment1v05Rotate90(edges, args.patch_size)
    ]

    dataset = ImageData(args.dataset_dir, args.output_dir, fragmentize_strategies)
    gt = {}
    for sample_gt in tqdm.tqdm(dataset):
        gt = {**gt, **sample_gt}

    img_names = list(gt.keys())
    random.shuffle(img_names)

    # Split the dataset to 3 parts with their ratios are 0.7, 0.15, 0.15
    indicies_for_splitting = [int(len(img_names) * 0.7), int(len(img_names) * (0.7 + 0.15))]
    train, val, test = np.split(img_names, indicies_for_splitting)

    train_gt = {key: gt[key] for key in train}
    val_gt = {key: gt[key] for key in val}
    test_gt = {key: gt[key] for key in test}

    with open(os.path.join(args.output_dir, 'train.json'), 'w') as f:
        json.dump(train_gt, f)
    with open(os.path.join(args.output_dir, 'val.json'), 'w') as f:
        json.dump(val_gt, f)
    with open(os.path.join(args.output_dir, 'test.json'), 'w') as f:
        json.dump(test_gt, f)


    print('Finished!')