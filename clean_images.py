import argparse
import glob
import os

import cv2
import numpy as np
import tqdm
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

from utils.segmentation import thresholdSegmentation, crop_image


def remove_small_artifacts(np_img, kernel_size=2):
    # Create a kernel for the morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion and dilation to remove small artifacts
    filtered_image = cv2.erode(np_img, kernel, iterations=1)
    filtered_image = cv2.dilate(filtered_image, kernel, iterations=1)
    return filtered_image


def remove_background(np_img, blur_size=11, ellipse_size=60, foreground_proportion=0.2):

    # Add padding, since it seems to make the segmentation works easier
    p_size = 20
    np_img = cv2.copyMakeBorder(np_img, p_size, p_size, p_size, p_size, cv2.BORDER_REFLECT_101)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    shape_ext = thresholdSegmentation(gray, blur_size, ellipse_size)

    mask = np.zeros_like(np_img)
    mask[:, :, 0] = shape_ext
    mask[:, :, 1] = shape_ext
    mask[:, :, 2] = shape_ext

    background_area = cv2.bitwise_not(mask)
    # Apply the mask to the image
    masked_background = cv2.bitwise_and(np_img, np_img, mask=background_area[:, :, 0])
    # Convert the isolated region to a NumPy array
    masked_background = masked_background.reshape(-1, 3)  # Reshape the isolated region into a 2D array

    # Filter out black pixels (pixel value 0)
    background_pixels = masked_background[masked_background[:, 0] != 0]

    masked_foreground = cv2.bitwise_and(np_img, np_img, mask=mask[:, :, 0])
    masked_foreground = masked_foreground.reshape(-1, 3)  # Reshape the isolated region into a 2D array

    # Filter out black pixels (pixel value 0)
    foreground_pixels = masked_foreground[masked_foreground[:, 0] != 0]
    n_items = min(len(foreground_pixels), int(foreground_proportion * len(background_pixels)))
    foreground_pixels_portion = foreground_pixels[:n_items]

    # Use K-means to cluster the colors
    num_clusters = 2  # one for foreground colours, one for background colours
    kmeans = KMeans(n_clusters=num_clusters, n_init=5)
    kmeans.fit(np.concatenate([background_pixels, foreground_pixels_portion], axis=0))

    item_count = np.bincount(kmeans.labels_)
    arg_sort = np.argsort(item_count)

    # Get the colors in the dominant cluster
    # arg_sort[0] should contain the background colour range, since it has the highest number of pixels
    dominant_colors = background_pixels[kmeans.labels_[:len(background_pixels)] != arg_sort[0]]

    # Calculate the range of colors within the dominant cluster
    min_values = np.min(dominant_colors, axis=0)

    # We assume that maximum colour for background is always a white colour!
    max_values = np.array([255, 255, 255], dtype=np.uint8)

    # Create a binary mask for pixels within the color range
    color_range_mask = cv2.inRange(np_img, min_values, max_values)

    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(np_img, np_img, mask=cv2.bitwise_not(color_range_mask))
    filtered_image = remove_small_artifacts(filtered_image)
    w, h = filtered_image.shape[:2]

    # Remove the padding added
    filtered_image = filtered_image[p_size:w-p_size, p_size:h-p_size]
    filtered_image = crop_image(filtered_image)
    filtered_image[filtered_image == 0] = 255
    return filtered_image


class ImageData(Dataset):
    def __init__(self, dataset_dir, output_dir):
        self.dataset = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True))
        self.working_dir = output_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        with Image.open(img_path) as f:
            img = f.convert("RGB")
        np_img = np.asarray(img)
        try:
            clean_img = remove_background(np_img)
            output_file = os.path.join(self.working_dir, os.path.basename(img_path))
            cv2.imwrite(output_file, cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
        except Exception as e:
            print(f'Unable to clean image {img_path}')
            raise e
        return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dataset generator", add_help=True)
    parser.add_argument("--dataset-dir", required=True, metavar="FILE", help="Path to papyrus images dataset")
    parser.add_argument("--n-workers", type=int, default=0, help="Number of workers")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output dataset",
    )

    args = parser.parse_args()
    dataset = ImageData(args.dataset_dir, args.output_dir)
    dataloader = DataLoader(dataset, batch_size=args.n_workers + 1, num_workers=args.n_workers)
    print('Starting to clean up dataset...')
    for idxs in tqdm.tqdm(dataloader):
        a = 1

    print('Finished!')