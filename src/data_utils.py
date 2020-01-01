import os

import cv2
import numpy as np
from keras_preprocessing.image import load_img, ImageEnhance


def normalize_rgb_values(x, color_scale=255):
    """
    Normalizes rgb values based on color scale (usually 255).
    :param x: The pixel, or numpy array that contains the pixels. The operation is applied to each pixel
    :param color_scale: The color scale, usually 255.
    :return: The normalized pixel values.
    """
    return x / (color_scale / 2)


def denormalize_rgb_values(x, color_scale=255):
    """
    Denormalizes rgb values based on color scale (usually 255).
    :param x: The pixel, or numpy array that contains the pixels. The operation is applied to each pixel
    :param color_scale: The color scale, usually 255.
    :return: The denormalized pixel values.
    """
    return x * (color_scale / 2)


def center_pixels(x, dataset_rgb_mean):
    """
    Normalizes pixel values after substracting the dataset's RGB mean values.
    :param x: The pixel, or numpy array that contains the pixels. The operation is applied to each pixel
    :param dataset_rgb_mean: The dataset's RGB mean.
    :return: The centered pixel value or array with centered pixel values.
    """
    return normalize_rgb_values(x - dataset_rgb_mean)


def decenter_pixels(x, dataset_rgb_mean):
    """
    Denormalizes pixel values and adds the dataset's RGB mean values.
    :param x: The pixel, or numpy array that contains the pixels. The operation is applied to each pixel
    :param dataset_rgb_mean: The dataset's RGB mean.
    :return: The decentered pixel array with decentered pixel values as uint8 (only up to 255 values needed).
    """
    return np.array(denormalize_rgb_values(x) + dataset_rgb_mean).astype(np.uint8)


def convert_pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image


def get_closest_element_to_value(array, value):
    """
    Given an array, and a value, returns an index j such that ``value`` is between array[j]
    and array[j+1]. array must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that value is out of range below and above respectively.
    """
    array = sorted(array)
    n = len(array)
    if value < array[0]:
        return array[0]
    elif value > array[n - 1]:
        return array[n - 1]
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.  # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return array[0]
    elif value == array[n - 1]:  # and top
        return array[n - 1]
    else:
        return array[jl]


def generate_data(n_epochs, training=True, generate_y=True, generate_x=True, scale=2, bicubic=True, rgb_mean=(114.21, 111.42, 103.03), n_training_files=800, n_val_files=100, correction=False):
    """
    Tailored generator of DIV2K data for batch-training and data augmentation. Created for use with keras models'/tensorflow's fit_generator method.
    Can be used to augment data with horizontal and vertical flipping, lowering and raising brightness. Those can be combined for further data augmentation.
    It is impossible for most computers/ GPUs to handle tensor allocation when using a lot of data, so usually augmenting it with 1 flip and 1 brightness change is
    the most we can do. This generator can only be used with the DIV2K dataset, or any other that follows the same naming schema.
    :param n_epochs: Number of epochs the generator has to run for. Equal to the number of epochs the fit_generator method will run for.
    :param training: Whether to generate training data (True) or validation data (False).
    :param generate_y: Whether to generate a y batch or just X.
    :param generate_x: Whether to generate an X batch or just y (or nothing for some obscure reason?).
    :param scale: The scaling factor of the DIV2K dataset.
    :param bicubic: Whether to generate data from the bicubic dataset or the unknown.
    :param rgb_mean: The dataset's RGB mean, used for centering pixels during data generation.
    :param n_training_files: The number of training files to generate per epoch, has to be at most all the images in the training folder.
    :param n_val_files: The number of validation files to generate per epoch,  has to be at most all the images in the validaiton folder.
    :param correction: Whether to generate training/validation data for the corrector model.
    :return: Yields a "batch" of the (X,y) pairs, the first being the original X,y pair for the current file count, and the rest being the augmented images.
    """
    if correction:
        if training:
            path_x = f"../data/res_upscale_x{scale}"
            path_y = "../data/DIV2K_train_HR/"
        else:
            path_x = f"../data/res_upscale_x{scale}_valid"
            path_y = "../data/DIV2K_valid_HR/"
    elif bicubic:
        if training:
            path_x = f"../data/DIV2K_train_LR_bicubic/X{scale}"
            path_y = f"../data/DIV2K_train_HR/"
        else:
            path_x = f"../data/DIV2K_valid_LR_bicubic/X{scale}"
            path_y = f"../data/DIV2K_valid_HR/"
    else:
        if training:
            path_x = f"../data/DIV2K_train_LR_unknown/X{scale}"
            path_y = f"../data/DIV2K_train_HR/"
        else:
            path_x = f"../data/DIV2K_valid_LR_unknown/X{scale}"
            path_y = f"../data/DIV2K_valid_HR/"

    batch_size = 1
    high_res_list = []
    for z in range(n_epochs):
        batch_index = 0
        if training:
            for i in range(n_training_files):
                x_train = []
                y_train = []
                if generate_x:
                    for root, dirs, files in os.walk(path_x):
                        high_res_list = []
                        for file_count in range(batch_index, batch_index + batch_size):
                            low_res_image = load_img(os.path.join(root, files[file_count]))
                            high_res_list.append(files[file_count].replace(f"x{scale}", "").replace("_res", ""))
                            # Appending them to existing batch
                            enhancer = ImageEnhance.Brightness(low_res_image)
                            x_train.append(center_pixels(np.array(low_res_image), rgb_mean))
                            # Apply image augmentation techniques
                            x_train.append(center_pixels(np.fliplr(np.array(low_res_image)), rgb_mean))
                            x_train.append(center_pixels(np.flipud(np.array(low_res_image)), rgb_mean))
                            x_train.append(center_pixels(np.array(enhancer.enhance(1.5)), rgb_mean))
                            x_train.append(center_pixels(np.array(enhancer.enhance(0.5)), rgb_mean))
                if generate_y:
                    for root, dirs, files in os.walk(path_y):
                        for file_name in high_res_list:
                            high_res_image = load_img(os.path.join(root, file_name))
                            enhancer = ImageEnhance.Brightness(high_res_image)
                            y_train.append(center_pixels(np.array(high_res_image), rgb_mean))
                            # Apply image augmentation techniques
                            y_train.append(center_pixels(np.fliplr(np.array(high_res_image)), rgb_mean))
                            y_train.append(center_pixels(np.flipud(np.array(high_res_image)), rgb_mean))
                            y_train.append(center_pixels(np.array(enhancer.enhance(1.5)), rgb_mean))
                            y_train.append(center_pixels(np.array(enhancer.enhance(0.5)), rgb_mean))

                batch_index += batch_size
                yield np.array(x_train), np.array(y_train)
        else:
            for i in range(n_val_files):
                x_train = []
                y_train = []
                if generate_x:
                    for root, dirs, files in os.walk(path_x):
                        high_res_list = []
                        for file_count in range(batch_index, batch_index + batch_size):
                            low_res_image = load_img(os.path.join(root, files[file_count]))
                            enhancer = ImageEnhance.Brightness(low_res_image)
                            high_res_list.append(files[file_count].replace(f"x{scale}", "").replace("_res", ""))
                            # Appending them to existing batch
                            x_train.append(center_pixels(np.array(low_res_image), rgb_mean))
                            # Apply image augmentation techniques
                            x_train.append(center_pixels(np.fliplr(np.array(low_res_image)), rgb_mean))
                            x_train.append(center_pixels(np.flipud(np.array(low_res_image)), rgb_mean))
                            x_train.append(center_pixels(np.array(enhancer.enhance(1.5)), rgb_mean))
                            x_train.append(center_pixels(np.array(enhancer.enhance(0.5)), rgb_mean))
                if generate_y:
                    for root, dirs, files in os.walk(path_y):
                        for file_name in high_res_list:
                            high_res_image = load_img(os.path.join(root, file_name))
                            enhancer = ImageEnhance.Brightness(high_res_image)
                            y_train.append(center_pixels(np.array(high_res_image), rgb_mean))
                            # Apply image augmentation techniques
                            y_train.append(center_pixels(np.fliplr(np.array(high_res_image)), rgb_mean))
                            y_train.append(center_pixels(np.flipud(np.array(high_res_image)), rgb_mean))
                            y_train.append(center_pixels(np.array(enhancer.enhance(1.5)), rgb_mean))
                            y_train.append(center_pixels(np.array(enhancer.enhance(0.5)), rgb_mean))

                batch_index += batch_size
                yield np.array(x_train), np.array(y_train)


def calculate_rgb_mean():
    gen = generate_data(n_epochs=1, training=True, generate_y=False)
    rgb_mean = np.array([0., 0., 0.])
    ctr = 0
    try:
        while True:
            ctr += 1
            res = next(gen)
            x = res[0][0]
            x_mean = np.mean(x, axis=(0, 1))
            rgb_mean += x_mean
    except StopIteration:
        rgb_mean /= ctr
    return rgb_mean
