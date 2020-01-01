import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.layers import Add, Conv2D, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

from src.data_utils import calculate_rgb_mean, center_pixels, decenter_pixels


class SuperResolutionModelTrainer:
    def __init__(self, scaling, saved_model=None, rgb_mean=None, recalulcate_mean=False, conv_width=16, network_depth=4, show_model_architecture=False):
        """
        Initializes the training, testing and usage interface of the super-resolution model described by the "Enhanced Deep Residual Networks for Single
        Image Super-Resolution" paper that can be found at: https://arxiv.org/pdf/1707.02921.pdf.
        Default settings are set to a model architecture of 16 residual blocks and a convolution layer width of 64, but can be modified to fit the needs
        of the application.
        :param scaling: The model's scaling parameter. Can be either 2, 3 or 4 for a x2, x3 or x4 times size augmentation.
        :param saved_model: The parameter used for loading a saved model, can be either a Keras saved model or a path to a weights file.
        :param rgb_mean: The pre-computed RGB mean of the training dataset. If none given, computes it from local dataset.
        :param recalulcate_mean: Whether to force recalculate the RGB mean or not.
        :param conv_width: The number of output filters in the convolutional layers.
        :param network_depth: The number of residual blocks to be built in the network initialization.
        :param show_model_architecture: Whether to print the model architecture or not.
        """
        # Image and network parameters
        self.color_scale = 255
        self.image_channels = 3
        self.scaling = scaling
        self.convolution_width = conv_width  # Prioritize increasing width instead of depth as it can maximize model capacity with limited resources
        self.network_depth = network_depth

        # Raise error if scaling not part of training set
        if scaling not in [2, 3, 4]:
            raise ValueError("Scaling factor not supported.")

        # Mean used for pixel centering, can be recalculated
        if not rgb_mean and not recalulcate_mean:
            self.dataset_rgb_mean = (114.21, 111.42, 103.03)
        elif not rgb_mean and recalulcate_mean:
            self.dataset_rgb_mean = calculate_rgb_mean()
        else:
            self.dataset_rgb_mean = rgb_mean

        # If no given model, create one. If saved weights file path given, load weights from path, if model given directly, load model
        if not saved_model:
            self.model = self.build_sr_model(channels=self.image_channels, scaling=self.scaling, convolution_width=self.convolution_width, network_depth=self.network_depth)
        elif isinstance(saved_model, str):
            self.model = self.build_sr_model(channels=self.image_channels, scaling=self.scaling, convolution_width=self.convolution_width, network_depth=self.network_depth)
            try:
                self.model.load_weights(saved_model)
            except FileNotFoundError:
                print("Given model weights file does not exist. Continuing with default initialization.")
        elif isinstance(saved_model, Model):
            self.model = saved_model

        # Display model architecture if enabled
        if show_model_architecture:
            self.model.summary(line_length=200)

    def build_sr_model(self, channels, scaling, convolution_width, network_depth):
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

        # Model creation from connected layers
        model = Model(inputs=input_layer, outputs=conv_3, name="Super_Resolution_Network")

        # Model compilation
        opt = Adam(learning_rate=10 ** -4, beta_1=0.9, beta_2=0.999, epsilon=10 ** -8)  # Described in paper section 4.2
        model.compile(optimizer=opt, loss="huber_loss", metrics=['acc', self.psnr_wrapper])
        return model

    @staticmethod
    def create_residual_block(block_input, convolution_width, block_count):
        sub_conv_1 = Conv2D(convolution_width, 3, padding='same', activation='relu', name="Residual_Block_" + str(block_count) + "_Convolutional_1")(block_input)
        sub_conv_2 = Conv2D(convolution_width, 3, padding='same', name="Residual_Block_" + str(block_count) + "_Convolutional_2")(sub_conv_1)
        sub_conv_mult = Lambda(function=lambda x: x * 0.1, name="Residual_Block_" + str(block_count) + "_Multiply")(sub_conv_2)
        sub_add_1 = Add(name="Residual_Block_" + str(block_count) + "_Add")([block_input, sub_conv_mult])
        return sub_add_1

    def psnr_wrapper(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, self.color_scale)

    def fit_on_generators(self, train_gen, val_gen, n_epochs, filepath_save):
        checkpoint = ModelCheckpoint(filepath_save, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.model.fit_generator(train_gen, validation_data=val_gen, steps_per_epoch=800, epochs=n_epochs, callbacks=callbacks_list, validation_steps=100, verbose=1)

    def evaluate_on_generator(self, eval_gen):
        return self.model.evaluate_generator(eval_gen, steps=100, verbose=1)

    def upscale_image(self, lr_image):
        if isinstance(lr_image, str):
            lr_image = Image.open(lr_image)
        if isinstance(lr_image, Image.Image):
            lr_image = np.array([np.array(lr_image)])
        high_scale_image = self.model.predict(center_pixels(lr_image, self.dataset_rgb_mean))
        high_scale_image = decenter_pixels(high_scale_image, self.dataset_rgb_mean)
        return Image.fromarray(high_scale_image[0])

    def correct_image(self, hr_image, lr_image):
        if isinstance(hr_image, Image.Image):
            hr_image = cv2.cvtColor(np.array(hr_image), cv2.COLOR_RGB2BGR)
        if isinstance(lr_image, Image.Image):
            lr_image = lr_image.resize((lr_image.size[0] * self.scaling, lr_image.size[1] * self.scaling), resample=Image.BILINEAR)
            lr_image = cv2.cvtColor(np.array(lr_image), cv2.COLOR_RGB2BGR)
        new_image = hr_image.copy()
        nrow, ncol = hr_image.shape[0], hr_image.shape[1]
        for i in range(nrow):
            for j in range(ncol):
                for c in range(self.image_channels):
                    if hr_image[i, j, c] >= 220 or hr_image[i, j, c] <= 30:
                        new_image[i, j, c] = lr_image[i, j, c]
        return Image.fromarray(cv2.cvtColor(new_image.copy(), cv2.COLOR_BGR2RGB))
