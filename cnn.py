import os

import numpy as np
import tensorflow as tf

from keras.layers import Add, Conv2D, Input, Lambda
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator, load_img


class SuperResolutionModelTrainer:
    def __init__(self, model=None, mean=None, scaling=2):
        # Raise error if scaling not part of training set
        if scaling not in [2, 3, 4]:
            raise ValueError("Scaling factor not supported.")
        self.RGB_SCALE = 255
        if not mean:
            self.dataset_rgb_mean = [0, 0, 0]
        else:
            self.dataset_rgb_mean = mean
        if not model:
            model = self.build_sr_cnn_model()

    def build_sr_cnn_model(self, convolution_width=64, network_depth=8, scaling=2):
        # Build network
        input_layer = Input(shape=(None, None, 3), name="Input")
        pixel_centering_layer = Lambda(function=lambda x: self.center_pixels(x), name="Pixel_Centering")(input_layer)

        # Entry layer for residual blocks
        conv_1 = residual_block = Conv2D(convolution_width, 3, padding='same', name="Convolution_Initial")(pixel_centering_layer)

        # Create specified number of residual blocks
        for i in range(network_depth):
            residual_block = self.create_residual_block(residual_block, convolution_width, i)

        # Create convolutional layer before upsampling
        conv_2 = Conv2D(convolution_width, 3, padding='same', name="Convolution_Pre_Upsampling")(residual_block)
        add_layer = Add(name="Addition_1")([conv_1, conv_2])

        # Upsampling
        if scaling in [2, 3]:
            upsample_layer = Conv2D(convolution_width * (scaling ** 2), 3, padding='same', name="Upsampling_1_x"+str(scaling))(add_layer)
            shuffle_layer = Lambda(function=lambda x: tf.nn.depth_to_space(x, scaling), name="Pixel_Shuffle_1")(upsample_layer)
        else:
            upsample_layer = Conv2D(convolution_width * (scaling ** 2), 3, padding='same', name="Upsampling_1_x"+str(scaling))(add_layer)
            shuffle_layer = Lambda(function=lambda x: tf.nn.depth_to_space(x, scaling), name="Pixel_Shuffle_1")(upsample_layer)
            upsample_layer = Conv2D(convolution_width * (scaling ** 2), 3, padding='same', name="Upsampling_2_x"+str(scaling))(shuffle_layer)
            shuffle_layer = Lambda(function=lambda x: tf.nn.depth_to_space(x, scaling), name="Pixel_Shuffle_2")(upsample_layer)

        # Final convolutional layer for output
        conv_3 = Conv2D(3, 3, padding='same', name="Convolution_Post_Upsampling")(shuffle_layer)

        output = Lambda(function=lambda x: self.decenter_pixels(x), name="Pixel_Decentering")(conv_3)
        return Model(inputs=input_layer, outputs=output, name="EDSR_Super_Resolution_CNN")

    @staticmethod
    def create_residual_block(block_input, convolution_width, block_count):
        sub_conv_1 = Conv2D(convolution_width, 3, padding='same', activation='relu', name="Residual_Block_" + str(block_count) + "_Convolutional_1")(block_input)
        sub_conv_2 = Conv2D(convolution_width, 3, padding='same', name="Residual_Block_" + str(block_count) + "_Convolutional_2")(sub_conv_1)
        sub_add_1 = Add(name="Residual_Block_" + str(block_count) + "_Add")([block_input, sub_conv_2])
        return sub_add_1

    def center_pixels(self, x):
        return self.normalize_rgb_values(x - self.dataset_rgb_mean)

    def decenter_pixels(self, x):
        return self.denormalize_rgb_values(x) + self.dataset_rgb_mean

    def normalize_rgb_values(self, x):
        return x/(self.RGB_SCALE/2)

    def denormalize_rgb_values(self, x):
        return x*(self.RGB_SCALE/2)


def generate_data_for_training(batch_index, batch_size=20):
    x_train = []
    y_train = []

    for root, dirs, files in os.walk("../data/DIV2K_train_LR_bicubic/X4/train"):
        for file_count in range(batch_index, batch_index+batch_size):
            print(file_count)
            low_res_image = load_img(os.path.join(root, files[file_count]))
            # Appending them to existing batch
            x_train.append(np.array(low_res_image))

    for root, dirs, files in os.walk("../data/DIV2K_train_HR/train"):
        for file_count in range(batch_index, batch_index+batch_size):
            print(file_count)
            high_res_image = load_img(os.path.join(root, files[file_count]))
            y_train.append(np.array(high_res_image))

    yield (x_train, y_train, batch_index+batch_size)


if __name__ == "__main__":
    SuperResolutionModelTrainer()
    start_index = 0

    x, y, index = generate_data_for_training(start_index)

