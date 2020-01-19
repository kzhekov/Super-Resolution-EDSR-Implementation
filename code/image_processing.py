import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras_preprocessing.image import load_img, save_img

from code.super_resolution_model import SuperResolutionModelTrainer


def threaded_display(im):
    """
    Used to display an image using threads, non-blocking.
    :param im: The image to display.
    :return: None
    """
    im.show()


def psnr_comparison_all_images(model_scaling=2, conv_width=32, network_depth=3, load_path="../data/binaries/sr_model_d3_w32_x4_aug_2.hdf5", low_res_path="../data/pictures", val_path="../data/pictures/val"):
    """
    Instantiates the specified model architecture and loads the weights from the specified path. Calculates the PSNR for every image in the DIV2K validation dataset.
    :param val_path: The path to the validation dataset.
    :param low_res_path: The path to the low-resolution images of the validation dataset.
    :param model_scaling: The scaling factor of the model
    :param conv_width: The number of output filters in the convolution
    :param network_depth: The number of resblocks in the network
    :param load_path: The path where the saved weights of the model are contained.
    :return: The PSNR results of the model upscaling, bilinear upscaling and corrected model upscaling.
    """
    srmt = SuperResolutionModelTrainer(model_scaling, load_path, conv_width=conv_width, network_depth=network_depth)
    iteration_list = []
    psnr_results_model = []
    psnr_results_bilinear = []
    psnr_results_correction = []
    for nbr in range(801, 811, 1):
        x = list("0000")[:-len(str(nbr))]
        file_n = list(str(nbr))
        x.extend(file_n)
        iteration_list.append("".join(x))
    for image_nr in iteration_list:
        lr_image = load_img(f"{low_res_path}/X{model_scaling}/{image_nr}x{model_scaling}.png")
        comparison_image = load_img(f"{val_path}/{image_nr}.png")
        test_res = test_prediction(lr_image, srmt, model_scaling, conv_width, network_depth, load_path, comparison_image, image_nr=image_nr)
        psnr_results_model.append(test_res[0])
        psnr_results_bilinear.append(test_res[1])
        psnr_results_correction.append(test_res[2])

    return psnr_results_model, psnr_results_bilinear, psnr_results_correction


def test_prediction(lr_image, model=None, model_scaling=2, conv_width=32, network_depth=3, load_path="../data/binaries/sr_model_d16_w64_x4_aug_2.hdf5",
                    comparison_image=None, display_res=True, image_nr="None"):
    """
    Uses an instantiated model or instantiates a model to test the upscaling on a given low-resolution PIL image.
    :param lr_image: The PIL low-resolution image.
    :param model: The instantiated model, if using an already instantiated one.
    :param model_scaling: The scaling of the model.
    :param conv_width: The number of output filters for the convolutional layers. Unnecessary if already instantiated.
    :param network_depth: The number of resblocks in the network. Unnecessary if already instantiated.
    :param load_path: The path to the file from which to load the weights. Unnecessary if already instantiated.
    :param comparison_image: The image to use for the PSNR metric. Unnecessary if no comparison is wanted.
    :param display_res: Whether to save the three (bilinear, model, corrected) upscaled images.
    :param image_nr: The image number that is being processed, used for saving the files.
    :return: Only if comparison image given, the PSNR metric for the model upscaled, bilinear upscaled and corrected upscaled images.
    """
    if not model:
        # Create model trainer and initialize model
        srmt = SuperResolutionModelTrainer(model_scaling, load_path, conv_width=conv_width, network_depth=network_depth)
    else:
        srmt = model
    hr_image = srmt.upscale_image(lr_image)
    bilinear_hr_image = lr_image.resize((lr_image.size[0] * model_scaling, lr_image.size[1] * model_scaling), resample=Image.BILINEAR)
    corrected_hr_image = srmt.correct_image(hr_image, lr_image)

    # Image saving parameters
    dpi = 100
    margin = 0.05  # (5% of the width/height of the figure...)
    xpixels, ypixels = hr_image.size[0], hr_image.size[1] * 3 + 200
    img_size = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi

    if display_res:
        fig, axs = plt.subplots(nrows=3, ncols=1, subplot_kw={'xticks': [], 'yticks': []}, figsize=img_size)
        for ax, img, title in zip(axs, [bilinear_hr_image, hr_image, corrected_hr_image], ["Bilinear upscaling", "Model upscaling", "Model + correction upscaling"]):
            ax.imshow(img, aspect='equal', interpolation='none')
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f"../data/pictures/results/x{model_scaling}_{image_nr}.png")
        plt.close(fig)

    if comparison_image:
        return srmt.psnr_wrapper(np.array(hr_image), np.array(comparison_image)).numpy(), srmt.psnr_wrapper(np.array(bilinear_hr_image), np.array(comparison_image)).numpy(), srmt.psnr_wrapper(np.array(corrected_hr_image), np.array(comparison_image)).numpy()


def plot_psnr_comparison_results(psnr_results_model, psnr_results_bilinear, psnr_results_corrector, model_scaling):
    """
    Used to plot all the different PSNR results for the model, bilinear and corrected upscalings.
    :param psnr_results_model: The PSNR results for the model upscaling
    :param psnr_results_bilinear: The PSNR results for the model upscaling
    :param psnr_results_corrector: The PSNR results for the model upscaling
    :param model_scaling: The model scaling parameter, either 2, 3 or 4.
    :return: None
    """
    fig, ax = plt.subplots()
    plt.ylim(10, 50)
    ax.scatter(range(1, len(psnr_results_model) + 1), psnr_results_model, label='Model upscaling PSNR')
    ax.scatter(range(1, len(psnr_results_bilinear) + 1), psnr_results_bilinear, label='Bilinear upscaling PSNR')
    ax.scatter(range(1, len(psnr_results_corrector) + 1), psnr_results_corrector, label='Model with correction upscaling PSNR')
    ax.plot(range(1, len(psnr_results_model) + 1), [np.mean(psnr_results_model) for _ in range(1, len(psnr_results_model) + 1)], label='Model upscaling PSNR mean')
    ax.plot(range(1, len(psnr_results_bilinear) + 1), [np.mean(psnr_results_bilinear) for _ in range(1, len(psnr_results_bilinear) + 1)], label='Bilinear upscaling PSNR mean')
    ax.plot(range(1, len(psnr_results_corrector) + 1), [np.mean(psnr_results_corrector) for _ in range(1, len(psnr_results_corrector) + 1)], label='Model with correction upscaling PSNR mean')
    fig.suptitle(f"PSNR Model vs Bilinear vs Corrected with scaling: x{model_scaling}")
    ax.legend()
    plt.savefig(f"../data/plots/PSNRx{model_scaling}.png")
    plt.show()
    print(f"Model upscaling: {np.mean(psnr_results_model)}, Corrected model upscaling: {np.mean(psnr_results_corrector)}, Bilinear upscaling: {np.mean(psnr_results_bilinear)}")


def build_model_and_plot_psnr_results_for_validation_dataset():
    psnr_results_model_x2, psnr_results_bilinear_x2, psnr_results_corrector_x2 = psnr_comparison_all_images(model_scaling=2, conv_width=32, network_depth=3, load_path="../data/binaries/sr_model_d3_w32_x2_aug_3.hdf5")
    plot_psnr_comparison_results(psnr_results_model_x2, psnr_results_bilinear_x2, psnr_results_corrector_x2, 2)
