import argparse
import json
import os

import cv2
import numpy as np

from utils.segmentation import crop_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dataset generator", add_help=True)
    parser.add_argument("--dataset-dir", required=True, help="Path to cleaned papyrus images dataset")
    parser.add_argument("--edge-json", required=True, help="Path to edges json file")
    parser.add_argument("--n-workers", type=int, default=0, help="Number of workers")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output dataset",
    )

    args = parser.parse_args()

    with open(args.edge_json) as f:
        edge_def = json.load(f)

    patches = []
    for image_id in edge_def['assets']:
        image_def = edge_def['assets'][image_id]
        image_path = os.path.join(args.dataset_dir, image_def['asset']['name'])
        np_img = cv2.imread(image_path)
        for idx, line_def in enumerate(image_def['regions']):
            if 'Broken' in line_def['tags']:
                continue
            bb = line_def['boundingBox']
            l, t, h, w = (round(bb['left']), round(bb['top']), round(bb['height']), round(bb['width']))
            line_im = np_img[t:t + h, l:l + w].copy()
            line_im_mask = (line_im != 255).astype(np.uint8) * 255
            line_im_mask = crop_image(line_im_mask)
            line_im_mask = crop_image(line_im_mask, pixel_value=255)
            line_im_mask = line_im_mask[:, :, 0]
            rel_path = os.path.join(image_id, f'{idx}.jpg')
            img_path = os.path.join(args.output_dir, rel_path)
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            cv2.imwrite(img_path, line_im_mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
            patches.append({'img_path': rel_path, 'width': line_im_mask.shape[1], 'height': line_im_mask.shape[0]})

    summary = {'edges': patches}
    summary_file = os.path.join(args.output_dir, 'edges.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f)
