import math
import os
import random
import re

import cv2
import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def remove_small_artifacts(np_img, kernel_size=2):
    # Create a kernel for the morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion and dilation to remove small artifacts
    filtered_image = cv2.erode(np_img, kernel, iterations=1)
    filtered_image = cv2.dilate(filtered_image, kernel, iterations=1)
    return filtered_image


def paste_image_to_center(target_image, source_image):
    # Get the dimensions of the target and source images
    target_height, target_width = target_image.shape[:2]
    source_height, source_width = source_image.shape[:2]

    # Calculate the position to paste the source image in the center of the target image
    paste_x = (target_width - source_width) // 2
    paste_y = (target_height - source_height) // 2

    # Create a copy of the target image to avoid modifying the original
    result_image = np.copy(target_image)

    # Paste the source image into the calculated position in the result image
    result_image[paste_y:paste_y + source_height, paste_x:paste_x + source_width] = source_image

    return result_image


def compute_white_percentage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_pixel_count = np.sum(gray > 250)
    total_pixels = gray.shape[0] * gray.shape[1]
    return white_pixel_count / total_pixels


def grouping(val):
    part, group = math.modf(val)
    return int(group), int(part * 10)


def center_crop(np_img, h, w):
    center = np_img.shape
    x = center[1] / 2 - w / 2
    y = center[0] / 2 - h / 2

    crop_img = np_img[int(y):int(y + h), int(x):int(x + w)]
    return crop_img


def visualise_fragments(images, labels, degree=0):
    img_vis, label_vis = [], []
    if len(images) == 0:
        return
    small_im_group = {}

    for image, label in zip(images, labels):
        if label['degree'] == degree:
            if isinstance(label['col'], float):
                j, part_j = grouping(label['col'])
                i, part_i = grouping(label['row'])
                part_label = {'col': part_j, 'row': part_i, 'degree': degree}
                part_im = image
                small_im_group.setdefault(f'j{j}i{i}', []).append((part_im, part_label))
            else:
                img_vis.append(image)
                label_vis.append(label)

    for im_group_id in small_im_group:
        group_imgs = [x[0] for x in small_im_group[im_group_id]]
        group_labels = [x[1] for x in small_im_group[im_group_id]]
        im_group = visualise_fragments(group_imgs, group_labels)
        img_vis.append(cv2.cvtColor(np.asarray(im_group), cv2.COLOR_RGB2BGR))
        pattern = re.search(r"j(\d+)i(\d+)", im_group_id)
        label_vis.append({'col': int(pattern.group(1)), 'row': int(pattern.group(2))})

    n_cols = max([x['col'] for x in label_vis]) + 1
    n_rows = max([x['row'] for x in label_vis]) + 1

    final_width = n_cols * images[0].shape[1]
    final_height = n_rows * images[0].shape[0]

    final_image = Image.new('RGB', (final_width, final_height), (255, 255, 255))

    for fragment, label in zip(img_vis, label_vis):
        col, row = label['col'], label['row']
        x = col * fragment.shape[1]
        y = row * fragment.shape[0]

        # Convert the numpy array to a PIL Image
        fragment_image = Image.fromarray(cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB))

        # Paste the fragment onto the final image
        final_image.paste(fragment_image, (x, y))

    # Display or save the final image
    return final_image
