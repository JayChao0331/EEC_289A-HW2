import os
import argparse
import cv2
import numpy as np
import random
from PIL import Image

import torch
import torchvision
from torchvision import transforms


EIGHT_CONNECTED_NEIGHBOR_KERNEL = np.array([[1., 1., 1.],
                                            [1., 0., 1.],
                                            [1., 1., 1.]], dtype=np.float64)
SIGMA_COEFF = 6.4
ERROR_THRESHOLD = 0.1


def normalized_ssd(sample, window, mask):
    wh, ww = window.shape
    sh, sw = sample.shape

    strided_sample = np.lib.stride_tricks.as_strided(sample, shape=((sh-wh+1), (sw-ww+1), wh, ww), 
                        strides=(sample.strides[0], sample.strides[1], sample.strides[0], sample.strides[1]))
    strided_sample = strided_sample.reshape(-1, wh, ww)

    strided_window = np.lib.stride_tricks.as_strided(window, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, window.strides[0], window.strides[1]))
    strided_window = strided_window.reshape(-1, wh, ww)

    strided_mask = np.lib.stride_tricks.as_strided(mask, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, mask.strides[0], mask.strides[1]))
    strided_mask = strided_mask.reshape(-1, wh, ww)

    sigma = wh / SIGMA_COEFF
    kernel = cv2.getGaussianKernel(ksize=wh, sigma=sigma)
    kernel_2d = kernel * kernel.T

    strided_kernel = np.lib.stride_tricks.as_strided(kernel_2d, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, kernel_2d.strides[0], kernel_2d.strides[1]))
    strided_kernel = strided_kernel.reshape(-1, wh, ww)

    squared_differences = ((strided_sample - strided_window)**2) * strided_kernel * strided_mask
    ssd = np.sum(squared_differences, axis=(1,2))
    ssd = ssd.reshape(sh-wh+1, sw-ww+1)

    total_ssd = np.sum(mask * kernel_2d)
    normalized_ssd = ssd / total_ssd

    return normalized_ssd


def get_candidate_indices(normalized_ssd, error_threshold=ERROR_THRESHOLD):
    min_ssd = np.min(normalized_ssd)
    min_threshold = min_ssd * (1. + error_threshold)
    indices = np.where(normalized_ssd <= min_threshold)
    return indices


def select_pixel_index(normalized_ssd, indices, method='uniform'):
    N = indices[0].shape[0]

    if method == 'uniform':
        weights = np.ones(N) / float(N)
    else:
        weights = normalized_ssd[indices]
        weights = weights / np.sum(weights)

    selection = np.random.choice(np.arange(N), size=1, p=weights)
    selected_index = (indices[0][selection], indices[1][selection])
    
    return selected_index


def get_neighboring_pixel_indices(pixel_mask):
    kernel = np.ones((3,3))
    dilated_mask = cv2.dilate(pixel_mask, kernel, iterations=1)
    neighbors = dilated_mask - pixel_mask

    # Recover the indices of the mask frontier.
    neighbor_indices = np.nonzero(neighbors)

    return neighbor_indices


def permute_neighbors(pixel_mask, neighbors):
    N = neighbors[0].shape[0]

    permuted_indices = np.random.permutation(np.arange(N))
    permuted_neighbors = (neighbors[0][permuted_indices], neighbors[1][permuted_indices])

    neighbor_count = cv2.filter2D(pixel_mask, ddepth=-1, kernel=EIGHT_CONNECTED_NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT)

    permuted_neighbor_counts = neighbor_count[permuted_neighbors]

    sorted_order = np.argsort(permuted_neighbor_counts)[::-1]
    permuted_neighbors = (permuted_neighbors[0][sorted_order], permuted_neighbors[1][sorted_order])

    return permuted_neighbors


def texture_can_be_synthesized(mask):
    mh, mw = mask.shape[:2]
    num_completed = np.count_nonzero(mask)
    num_incomplete = (mh * mw) - num_completed
    
    return num_incomplete > 0


def initialize_texture_synthesis(original_sample, window_size, kernel_size):
    sample = original_sample
    
    sample = sample.astype(np.float64)
    sample = sample / 255.

    window = np.zeros(window_size, dtype=np.float64)

    if original_sample.ndim == 2:
        result_window = np.zeros_like(window, dtype=np.uint8)
    else:
        result_window = np.zeros(window_size + (3,), dtype=np.uint8)

    h, w = window.shape
    mask = np.zeros((h, w), dtype=np.float64)

    sh, sw = original_sample.shape[:2]
    ih = (sh // 2) - 2
    iw = (sw // 2) - 2
    seed = sample[ih:ih+3, iw:iw+3]

    ph, pw = (h//2)-1, (w//2)-1
    window[ph:ph+3, pw:pw+3] = seed
    mask[ph:ph+3, pw:pw+3] = 1
    result_window[ph:ph+3, pw:pw+3] = original_sample[ih:ih+3, iw:iw+3]

    win = kernel_size//2
    padded_window = cv2.copyMakeBorder(window, 
                                       top=win, bottom=win, left=win, right=win, borderType=cv2.BORDER_CONSTANT, value=0.)
    padded_mask = cv2.copyMakeBorder(mask,
                                     top=win, bottom=win, left=win, right=win, borderType=cv2.BORDER_CONSTANT, value=0.)
    
    window = padded_window[win:-win, win:-win]
    mask = padded_mask[win:-win, win:-win]

    return sample, window, mask, padded_window, padded_mask, result_window


def synthesize_texture(original_sample, window_size, kernel_size):
    global gif_count
    (sample, window, mask, padded_window, 
        padded_mask, result_window) = initialize_texture_synthesis(original_sample, window_size, kernel_size)

    while texture_can_be_synthesized(mask):
        neighboring_indices = get_neighboring_pixel_indices(mask)

        neighboring_indices = permute_neighbors(mask, neighboring_indices)
        
        for ch, cw in zip(neighboring_indices[0], neighboring_indices[1]):

            window_slice = padded_window[ch:ch+kernel_size, cw:cw+kernel_size]
            mask_slice = padded_mask[ch:ch+kernel_size, cw:cw+kernel_size]

            ssd = normalized_ssd(sample, window_slice, mask_slice)
            indices = get_candidate_indices(ssd)
            selected_index = select_pixel_index(ssd, indices)

            selected_index = (selected_index[0] + kernel_size // 2, selected_index[1] + kernel_size // 2)

            window[ch, cw] = sample[selected_index]
            mask[ch, cw] = 1
            result_window[ch, cw] = original_sample[selected_index[0], selected_index[1]]

    return result_window


def validate_args(args):
    wh, ww = args.window_height, args.window_width
    if wh < 3 or ww < 3:
        raise ValueError('window_size must be greater than or equal to (3,3).')

    if args.kernel_size <= 1:
        raise ValueError('kernel size must be greater than 1.')

    if args.kernel_size % 2 == 0:
        raise ValueError('kernel size must be odd.')

    if args.kernel_size > min(wh, ww):
        raise ValueError('kernel size must be less than or equal to the smaller window_size dimension.')
    

def generate_mnist():
    print('Reading MNIST images...')
    image_set = torchvision.datasets.MNIST(root='./', train=False, download=True)

    image_data = []
    for image, _ in image_set:
        image_data.append(np.array(image))
    
    indices = list(range(len(image_data)))
    random.shuffle(indices)
    select_images = [image_data[i] for i in indices[:100]]

    concat_img = np.zeros((280, 280), dtype=np.uint8)
    pointer = 0
    for row_id in range(0, 280, 28):
        for col_id in range(0, 280, 28):
            concat_img[row_id:row_id+28, col_id:col_id+28] = select_images[pointer]
            pointer += 1
    
    im = Image.fromarray(concat_img)
    im.save('./mnist_cat.png')


def parse_args():
    parser = argparse.ArgumentParser(description='Perform texture synthesis')
    parser.add_argument('--window_height', type=int,  required=False, default=28, help='Height of the synthesis window')
    parser.add_argument('--window_width', type=int, required=False, default=28, help='Width of the synthesis window')
    parser.add_argument('--kernel_size', type=int, required=False, default=11, help='One dimension of the square synthesis kernel')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    validate_args(args)

    generate_mnist()

    sample = cv2.imread('./mnist_cat.png', cv2.IMREAD_GRAYSCALE)
    sample = cv2.resize(sample, (128,128), interpolation= cv2.INTER_LINEAR)

    synthesized_texture = synthesize_texture(original_sample=sample, 
                                             window_size=(args.window_height, args.window_width), 
                                             kernel_size=args.kernel_size)
    
    cv2.imwrite('./mnist_cat_synthesized.png', synthesized_texture)




if __name__ == '__main__':
    main()
