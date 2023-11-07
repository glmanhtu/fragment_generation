import random
from typing import Dict, List, Tuple

import cv2
import numpy as np

from scipy.ndimage import rotate
from utils.segmentation import crop_image
from utils.utils import remove_small_artifacts, paste_image_to_center


class FragmentizeStrategy:
    def __init__(self, edges, patch_size):
        self.edges = edges
        self.patch_size = patch_size

    def name(self):
        return self.__class__.__name__

    def split(self, np_im: np.array) -> Tuple[List[np.array], List[Dict]]:
        raise NotImplementedError()

    def is_edge_up(self, np_img):
        """
        Detect the current edge is facing up
        @param np_img: numpy image
        @return: boolean
        """
        return np.sum(np_img[0, :]) < np.sum(np_img[-1, :])

    def get_edge(self, width, rescale=1):
        length = 0
        img = None
        while length < width:
            edge = random.choice(self.edges)
            img = cv2.imread(edge['img_path'])[:, :, 0]
            if img.shape[1] < img.shape[0]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if rescale != 1:
                dim = int(img.shape[1] * rescale), int(img.shape[0] * rescale)
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                img = (img > 10).astype(np.uint8) * 255

            length = img.shape[1]

        if self.is_edge_up(img):
            img = cv2.rotate(img, cv2.ROTATE_180)

        cut_from = random.randint(0, img.shape[1] - width)
        edge = img[:, cut_from:cut_from + width].copy()
        edge = crop_image(edge, is_gray=True)
        height = edge.shape[0]

        out_img = np.zeros((height, width), dtype=np.uint8)
        eh, ew = edge.shape[:2]
        out_img[:, 0:min(width, ew)] = edge[:, 0:min(width, ew)]
        return out_img / 255

    def fragmentize(self, patch, rescale=1):
        p_h, p_w = patch.shape[:2]
        top_edge = self.get_edge(p_w, rescale)
        top_edge = cv2.rotate(top_edge, cv2.ROTATE_180)
        bottom_edge = self.get_edge(p_w, rescale)
        left_edge = self.get_edge(p_h, rescale)
        left_edge = cv2.rotate(left_edge, cv2.ROTATE_90_CLOCKWISE)
        right_edge = self.get_edge(p_h, rescale)
        right_edge = cv2.rotate(right_edge, cv2.ROTATE_90_COUNTERCLOCKWISE)

        curr_w, curr_h = 0, 0
        mask = np.ones_like(patch, dtype=np.float32)[:, :, 0]
        mask[curr_h: curr_h + top_edge.shape[0], curr_w: curr_w + p_w] = top_edge
        mask[curr_h:curr_h + p_h, curr_w + p_w - right_edge.shape[1]: curr_w + p_w] *= right_edge
        mask[curr_h + p_h - bottom_edge.shape[0]: curr_h + p_h, curr_w: curr_w + p_w] *= bottom_edge
        mask[curr_h:curr_h + p_h, curr_w: curr_w + left_edge.shape[1]] *= left_edge
        fragment = patch * np.expand_dims(mask, -1)
        fragment = crop_image(fragment.astype(np.uint8))
        fragment = remove_small_artifacts(fragment, kernel_size=3)

        out_size = max(patch.shape)
        output = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        fragment = paste_image_to_center(output, fragment)
        fragment[fragment == 0] = 255
        return fragment


class Fragment1v1RotateFree(FragmentizeStrategy):

    def __init__(self, edges, patch_size):
        super().__init__(edges, patch_size)

    def split_degree(self, np_im: np.array, degree: int):
        im = np_im.copy()
        h, w = im.shape[:2]
        p_size = self.patch_size
        if h < p_size * 1.5 and w < p_size * 1.5:
            return [], []

        if h < p_size * 0.8 or w < p_size * 0.8:
            return [], []

        # Crop the image to make it fit with the patch size
        h, w = round(im.shape[0] / p_size) * p_size, round(im.shape[1] / p_size) * p_size
        output = np.ones((h, w, 3), dtype=np.uint8) * 255
        im = im[0: min(h, im.shape[0]), 0:min(w, im.shape[1])]
        im = paste_image_to_center(output, im)

        images, labels = [], []
        i = 0
        while (i + 1) * p_size <= h:
            j = 0
            curr_h = i * p_size
            while (j + 1) * p_size <= w:
                curr_w = j * p_size
                patch = im[curr_h:curr_h + p_size, curr_w:curr_w + p_size].copy()
                fragment = self.fragmentize(patch)
                rotated_fragment = rotate(fragment, degree, mode='constant', cval=255, reshape=False)
                images.append(rotated_fragment)
                labels.append({'col': j, 'row': i, 'degree': degree})
                j += 1
            i += 1
        return images, labels

    def split(self, np_im: np.array) -> Tuple[List[np.array], List[Dict]]:
        pass


class Fragment1v1Rotate90(Fragment1v1RotateFree):

    def __init__(self, edges, patch_size):
        super().__init__(edges, patch_size)

    def split(self, np_im: np.array) -> Tuple[List[np.array], List[Dict]]:
        images, labels = [], []
        for deg in [0, 90, 180, 270]:
            deg_imgs, deg_lbs = self.split_degree(np_im, deg)
            images += deg_imgs
            labels += deg_lbs

        return images, labels


class Fragment1v05RotateFree(FragmentizeStrategy):

    def __init__(self, edges, patch_size):
        super().__init__(edges, patch_size)

    def split_degree(self, np_im: np.array, degree: int, p_size):
        im = np_im.copy()
        h, w = im.shape[:2]
        if h < p_size * 1.5 and w < p_size * 1.5:
            return [], []

        if h < p_size * 0.8 or w < p_size * 0.8:
            return [], []

        # Crop the image to make it fit with the patch size
        h, w = round(im.shape[0] / p_size) * p_size, round(im.shape[1] / p_size) * p_size
        output = np.ones((h, w, 3), dtype=np.uint8) * 255
        im = im[0: min(h, im.shape[0]), 0:min(w, im.shape[1])]
        im = paste_image_to_center(output, im)

        images, labels = [], []
        i = 0
        while (i + 1) * p_size <= h:
            j = 0
            curr_h = i * p_size
            while (j + 1) * p_size <= w:
                curr_w = j * p_size
                patch = im[curr_h:curr_h + p_size, curr_w:curr_w + p_size].copy()
                odd_even = (i % 2 == 0 and j % 2 != 0) or (i % 2 != 0 and j % 2 == 0)
                if p_size == self.patch_size and odd_even:
                    frag_imgs, frag_labels = self.split_degree(patch, degree, self.patch_size // 2)
                    for img, label in zip(frag_imgs, frag_labels):
                        label['col'] = float(f'{j}.{label["col"]}')
                        label['row'] = float(f'{i}.{label["row"]}')
                        labels.append(label)

                        frag_im = np.ones((p_size, p_size, 3), dtype=np.uint8) * 255
                        frag_im = paste_image_to_center(frag_im, img)
                        images.append(frag_im)
                else:
                    fragment = self.fragmentize(patch, rescale=p_size / self.patch_size)
                    rotated_fragment = rotate(fragment, degree, mode='constant', cval=255, reshape=False)
                    images.append(rotated_fragment)
                    labels.append({'col': j, 'row': i, 'degree': degree})
                j += 1
            i += 1
        return images, labels

    def split(self, np_im: np.array) -> Tuple[List[np.array], List[Dict]]:
        pass


class Fragment1v05Rotate90(Fragment1v05RotateFree):

    def __init__(self, edges, patch_size):
        super().__init__(edges, patch_size)

    def split(self, np_im: np.array) -> Tuple[List[np.array], List[Dict]]:
        images, labels = [], []
        for deg in [0, 90, 180, 270]:
            deg_imgs, deg_lbs = self.split_degree(np_im, deg, self.patch_size)
            images += deg_imgs
            labels += deg_lbs

        return images, labels
