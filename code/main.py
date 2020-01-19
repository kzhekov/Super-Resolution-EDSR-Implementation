from PIL import Image

from code.image_processing import test_prediction, build_model_and_plot_psnr_results_for_validation_dataset
from code.super_resolution_model import SuperResolutionModelTrainer


def test_model_training():
    """
    Initialize and train the model on the DIV2K dataset, which is not included in the submitted files. If you wish to use the training,
    you would have to put the files in the data folder as follows:
        -	data/DIV2K_train_LR_bicubic/X2 for the low-resolution bicubic images to be upscaled x2, a.k.a. “X”.
        -	data/DIV2K_train_HR/ for the high-resolution training images, a.k.a. “y”.
    :return: None
    """
    scaling = 2
    model_width = 32
    model_depth = 3
    srmt = SuperResolutionModelTrainer(scaling=scaling, conv_width=model_width, network_depth=model_depth, saved_model="../data/binaries/sr_model_d3_w32_x2_aug_3.hdf5", show_model_architecture=True)
    srmt.train_network()


def test_psnr_on_all_images():
    """
    Test the upscaling on all images in the data/pictures/X2 folder that have a corresponding ground truth image in the data/pictures/val folder.
    Saves the PSNR plot in the data/results folder.
    :return: None
    """
    build_model_and_plot_psnr_results_for_validation_dataset()


def test_on_one_image(image_file_name, validation_path="../data/pictures/val", low_res_path="../data/pictures"):
    """
    Test the upscaling on a single image, given a string that is the comparison image's name without the extension.
    The low-resolution image to be used for the upscaling has to have the name of the high-resolution one, followed by x2, x3 or x4 based on its scaling.
    :param low_res_path: The path containing the low-resolution images. The images have to be in a subfolder x2, x3 or x4 based on their scaling.
    :param validation_path: The path containing the ground truth (high resolution) images.
    :param image_file_name: The name of the high resolution image, without the extension.
    :return: None
    """
    scaling = 2
    model_width = 32
    model_depth = 3
    srmt = SuperResolutionModelTrainer(scaling=scaling, conv_width=model_width, network_depth=model_depth, saved_model="../data/binaries/sr_model_d3_w32_x2_aug_3.hdf5", show_model_architecture=True)
    low_res_image = Image.open(f"{low_res_path}/X{scaling}/{image_file_name}x2.png")
    comp_image = Image.open(f"{validation_path}/{image_file_name}.png")
    psnr = test_prediction(lr_image=low_res_image, comparison_image=comp_image, model=srmt, display_res=True, image_nr=image_file_name, model_scaling=scaling)
    print(f"Model upscaling PSNR: {psnr[0]} dB")
    print(f"Bilinear prediction PSNR: {psnr[1]} dB")
    print(f"Corrected upscaling PSNR: {psnr[2]} dB")


if __name__ == "__main__":
    # Test upscaling on one image
    test_on_one_image("0802")

    # Plot PSNR comparison on multiple images
    test_psnr_on_all_images()
