import os
import numpy as np
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.layers import Add, Conv2D, Input, Lambda, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import load_img


class SuperResolutionModelTrainer:
    def __init__(self, scaling, model=None, rgb_mean=None):
        # Image and network parameters
        self.color_scale = 255
        self.image_channels = 3
        self.scaling = scaling
        self.convolution_width = 16
        self.network_depth = 4
        # Raise error if scaling not part of training set
        if scaling not in [2, 3, 4]:
            raise ValueError("Scaling factor not supported.")
        if not rgb_mean:
            self.dataset_rgb_mean = (114.21, 111.42, 103.03)
        else:
            self.dataset_rgb_mean = rgb_mean
        if not model:
            self.model = self.build_sr_cnn_model(channels=self.image_channels, scaling=self.scaling, convolution_width=self.convolution_width, network_depth=self.network_depth)
            self.model.summary(line_length=200)

    def build_sr_cnn_model(self, channels=3, scaling=2, convolution_width=1, network_depth=1):
        # Build network
        input_layer = Input(shape=(None, None, channels), name="Input")

        # Entry layer for residual blocks
        conv_1 = residual_block = Conv2D(convolution_width, channels, padding='same', name="Convolution_Initial")(input_layer)

        # Create specified number of residual blocks
        for i in range(network_depth):
            residual_block = self.create_residual_block(residual_block, convolution_width, i)

        # Create convolutional layer before upsampling
        conv_2 = Conv2D(convolution_width, channels, padding='same', name="Convolution_Pre_Upsampling")(residual_block)
        add_layer = Add(name="Addition_1")([conv_1, conv_2])

        # Upsampling
        if scaling in [2, 3]:
            # One-layer upsampling for x2 or x3
            upsample_layer = Conv2D(convolution_width * (scaling ** 2), channels, padding='same', name="Upsampling_1_x" + str(scaling))(add_layer)
            shuffle_layer = Lambda(function=lambda x: tf.nn.depth_to_space(x, scaling), name="Pixel_Shuffle_1")(upsample_layer)
        else:
            # Two-layer upsampling for x4 (x2 -> shuffle -> x2 -> shuffle)
            scaling //= 2
            upsample_layer = Conv2D(convolution_width * (scaling ** 2), channels, padding='same', name="Upsampling_1_x" + str(scaling))(add_layer)
            shuffle_layer = Lambda(function=lambda x: tf.nn.depth_to_space(x, scaling), name="Pixel_Shuffle_1")(upsample_layer)
            upsample_layer = Conv2D(convolution_width * (scaling ** 2), channels, padding='same', name="Upsampling_2_x" + str(scaling))(shuffle_layer)
            shuffle_layer = Lambda(function=lambda x: tf.nn.depth_to_space(x, scaling), name="Pixel_Shuffle_2")(upsample_layer)

        # Final convolutional layer for output
        conv_3 = Conv2D(3, channels, padding='same', name="Convolution_Post_Upsampling")(shuffle_layer)

        model = Model(inputs=input_layer, outputs=conv_3, name="EDSR_Super_Resolution_CNN")

        opt = Adam(lr=10 ** -3)
        model.compile(optimizer=opt, loss="mean_squared_error", metrics=['acc'])
        return model

    @staticmethod
    def create_residual_block(block_input, convolution_width, block_count):
        sub_conv_1 = Conv2D(convolution_width, 3, padding='same', activation='relu', name="Residual_Block_" + str(block_count) + "_Convolutional_1")(block_input)
        sub_conv_2 = Conv2D(convolution_width, 3, padding='same', name="Residual_Block_" + str(block_count) + "_Convolutional_2")(sub_conv_1)
        sub_conv_mult = Lambda(function=lambda x: x * 0.1, name="Residual_Block_" + str(block_count) + "_Multiply")(sub_conv_2)
        sub_add_1 = Add(name="Residual_Block_" + str(block_count) + "_Add")([block_input, sub_conv_mult])
        return sub_add_1

    def center_pixels(self, x):
        print(x)
        return self.normalize_rgb_values(x - self.dataset_rgb_mean)

    def decenter_pixels(self, x):
        print(x)
        return self.denormalize_rgb_values(x) + self.dataset_rgb_mean

    def normalize_rgb_values(self, x):
        return x / (self.color_scale / 2)

    def denormalize_rgb_values(self, x):
        return x * (self.color_scale / 2)


def generate_data(training=True, generate_y=True, scale=2, bicubic=True):
    if bicubic:
        if scale == 2:
            if training:
                path_x = "../data/DIV2K_train_LR_bicubic/X2"
                path_y = "../data/DIV2K_train_HR/train"
            else:
                path_x = "../data/DIV2K_valid_LR_bicubic/X2"
                path_y = "../data/DIV2K_valid_HR/"
        elif scale == 3:
            if training:
                path_x = "../data/DIV2K_train_LR_bicubic/X3"
                path_y = "../data/DIV2K_train_HR/train"
            else:
                path_x = "../data/DIV2K_valid_LR_bicubic/X3"
                path_y = "../data/DIV2K_valid_HR/"
        else:
            if training:
                path_x = "../data/DIV2K_train_LR_bicubic/X4/train"
                path_y = "../data/DIV2K_train_HR/train"
            else:
                path_x = "../data/DIV2K_valid_LR_bicubic/X4"
                path_y = "../data/DIV2K_valid_HR/"
    else:
        if scale == 2:
            if training:
                path_x = "../data/DIV2K_train_LR_unknown/X2"
                path_y = "../data/DIV2K_train_HR/train"
            else:
                path_x = "../data/DIV2K_valid_LR_unknown/X2"
                path_y = "../data/DIV2K_valid_HR/"
        elif scale == 3:
            if training:
                path_x = "../data/DIV2K_train_LR_unknown/X3"
                path_y = "../data/DIV2K_train_HR/train"
            else:
                path_x = "../data/DIV2K_valid_LR_unknown/X3"
                path_y = "../data/DIV2K_valid_HR/"
        else:
            if training:
                path_x = "../data/DIV2K_train_LR_unknown/X4"
                path_y = "../data/DIV2K_train_HR/train"
            else:
                path_x = "../data/DIV2K_valid_LR_unknown/X4"
                path_y = "../data/DIV2K_valid_HR/"

    batch_index = 0
    batch_size = 1
    for i in range(800):
        x_train = []
        y_train = []
        for root, dirs, files in os.walk(path_x):
            for file_count in range(batch_index, batch_index + batch_size):
                low_res_image = load_img(os.path.join(root, files[file_count]))
                # Appending them to existing batch
                x_train.append(np.array(low_res_image))
        if generate_y:
            for root, dirs, files in os.walk(path_y):
                for file_count in range(batch_index, batch_index + batch_size):
                    high_res_image = load_img(os.path.join(root, files[file_count]))
                    y_train.append(np.array(high_res_image))

        batch_index += batch_size
        yield np.array(x_train), np.array(y_train)


def calculate_mean():
    gen = generate_data(training=True, generate_y=False)
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


if __name__ == "__main__":
    start_index = 0
    training = True
    scaling = 4

    srmt = SuperResolutionModelTrainer(scaling)
    if training:
        training_generator = generate_data(training=True, scale=scaling)
        validation_generator = generate_data(training=False, scale=scaling)
        filepath = "./sr_model.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        srmt.model.fit_generator(training_generator, validation_data=validation_generator, steps_per_epoch=800, epochs=25, callbacks=callbacks_list, validation_steps=100)
    else:
        mean = calculate_mean()
        print("CALCULATED MEAN:", mean)
