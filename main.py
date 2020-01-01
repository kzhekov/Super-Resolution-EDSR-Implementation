import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras_preprocessing.image import load_img, save_img

from src.data_utils import generate_data
from src.super_resolution_model import SuperResolutionModelTrainer


def train_network(model_scaling=4, conv_width=64, network_depth=16, n_epochs=25, checkpoint_save_path="../data/binaries/sr_checkpoint.hdf5"):
    # Create model trainer and initialize model
    srmt = SuperResolutionModelTrainer(model_scaling, conv_width=conv_width, network_depth=network_depth, show_model_architecture=True)
    # Generate data for training and validation
    training_generator = generate_data(n_epochs=n_epochs, training=True, scale=model_scaling)
    validation_generator = generate_data(n_epochs=n_epochs, training=False, scale=model_scaling)
    # Fit the model on both generators
    srmt.fit_on_generators(train_gen=training_generator, val_gen=validation_generator, n_epochs=n_epochs, filepath_save=checkpoint_save_path)


def evaluate_network(model_scaling=4, n_epochs=25):
    # Create model trainer and initialize model
    srmt = SuperResolutionModelTrainer(model_scaling, "../data/binaries/sr_model_d4_w16_x4.hdf5")
    # Generate data for evaluation
    evaluation_generator = generate_data(n_epochs=n_epochs, training=False, scale=model_scaling)
    # Evaluate and print results
    eval_res = srmt.evaluate_on_generator(evaluation_generator)
    print(eval_res)


def threaded_display(im):
    im.show()


def process_all_images_and_save(model_scaling=4, conv_width=64, network_depth=16, load_path="../data/binaries/sr_model_d16_w64_x4_aug_2.hdf5"):
    srmt = SuperResolutionModelTrainer(model_scaling, load_path, conv_width=conv_width, network_depth=network_depth)
    iteration_list = []
    for nbr in range(1, 801, 1):
        x = list("0000")[:-len(str(nbr))]
        file_n = list(str(nbr))
        x.extend(file_n)
        iteration_list.append("".join(x))
    for image_nr in iteration_list:
        lr_image = load_img(f"../data/DIV2K_valid_LR_bicubic/X{model_scaling}/{image_nr}x{model_scaling}.png")
        hr_image = srmt.upscale_image(lr_image)
        save_img(f"../data/res_upscale_x{model_scaling}/{image_nr}x{model_scaling}_res.png", hr_image)


def psnr_comparison_all_images(model_scaling=4, conv_width=64, network_depth=16, load_path="../data/binaries/sr_model_d16_w64_x4_aug_2.hdf5"):
    srmt = SuperResolutionModelTrainer(model_scaling, load_path, conv_width=conv_width, network_depth=network_depth)
    iteration_list = []
    psnr_results_model = []
    psnr_results_bilinear = []
    psnr_results_correction = []
    for nbr in range(801, 901, 1):
        x = list("0000")[:-len(str(nbr))]
        file_n = list(str(nbr))
        x.extend(file_n)
        iteration_list.append("".join(x))
    for image_nr in iteration_list:
        lr_image = load_img(f"../data/DIV2K_valid_LR_bicubic/X{model_scaling}/{image_nr}x{model_scaling}.png")
        comparison_image = load_img(f"../data/DIV2K_valid_HR/{image_nr}.png")
        test_res = test_prediction(lr_image, model_scaling, conv_width, network_depth, load_path, comparison_image, model=srmt, image_nr=image_nr)
        psnr_results_model.append(test_res[0])
        psnr_results_bilinear.append(test_res[1])
        psnr_results_correction.append(test_res[2])

    return psnr_results_model, psnr_results_bilinear, psnr_results_correction


def test_prediction(lr_image, model_scaling=4, conv_width=64, network_depth=16, load_path="../data/binaries/sr_model_d16_w64_x4_aug_2.hdf5", comparison_image=None, corrector_load_path="../data/binaries/corrector_checkpoint_8.hdf5", display_res=True, model=None, image_nr="None"):
    if not model:
        # Create model trainer and initialize model
        srmt = SuperResolutionModelTrainer(model_scaling, load_path, conv_width=conv_width, network_depth=network_depth, saved_corrector_model=corrector_load_path)
    else:
        srmt = model
    hr_image = srmt.upscale_image(lr_image)
    bilinear_hr_image = lr_image.resize((lr_image.size[0] * model_scaling, lr_image.size[1] * model_scaling), resample=Image.BILINEAR)
    corrected_hr_image = srmt.correct_image(hr_image, lr_image)

    if display_res:
        dpi = 100
        margin = 0.05  # (5% of the width/height of the figure...)
        xpixels, ypixels = hr_image.size[0], hr_image.size[1] * 3 + 200
        img_size = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi
        fig, axs = plt.subplots(nrows=3, ncols=1, subplot_kw={'xticks': [], 'yticks': []}, figsize=img_size)
        for ax, img, title in zip(axs, [bilinear_hr_image, hr_image, corrected_hr_image], ["Bilinear upscaling", "Model upscaling", "Model + correction upscaling"]):
            ax.imshow(img, aspect='equal', interpolation='none')
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f"../data/pictures/x{model_scaling}/{image_nr}.png")
        plt.close(fig)

    if comparison_image:
        return srmt.psnr_wrapper(np.array(hr_image), np.array(comparison_image)).numpy(), srmt.psnr_wrapper(np.array(bilinear_hr_image), np.array(comparison_image)).numpy(), srmt.psnr_wrapper(np.array(corrected_hr_image), np.array(comparison_image)).numpy()


def plot_psnr_comparison_results(psnr_results_model, psnr_results_bilinear, psnr_results_corrector, model_scaling):
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
    print(f"model:{np.mean(psnr_results_model)}, corrector:{np.mean(psnr_results_corrector)}, bicubic:{np.mean(psnr_results_bilinear)}")


def main():
    psnr_results_model_x2, psnr_results_bilinear_x2, psnr_results_corrector_x2 = psnr_comparison_all_images(model_scaling=2, conv_width=32, network_depth=3, load_path="../data/binaries/sr_model_d3_w32_x2_aug_3.hdf5")
    plot_psnr_comparison_results(psnr_results_model_x2, psnr_results_bilinear_x2, psnr_results_corrector_x2, 2)
    psnr_results_model_x3, psnr_results_bilinear_x3, psnr_results_corrector_x3 = psnr_comparison_all_images(model_scaling=3, conv_width=32, network_depth=3, load_path="../data/binaries/sr_model_d3_w32_x3_aug_3.hdf5")
    plot_psnr_comparison_results(psnr_results_model_x3, psnr_results_bilinear_x3, psnr_results_corrector_x3, 3)
    psnr_results_model_x4, psnr_results_bilinear_x4, psnr_results_corrector_x4 = psnr_comparison_all_images()
    plot_psnr_comparison_results(psnr_results_model_x4, psnr_results_bilinear_x4, psnr_results_corrector_x4, 4)


if __name__ == "__main__":
    srmt = SuperResolutionModelTrainer(scaling=2, conv_width=32, network_depth=3, saved_model=r"E:\University\Project\data\binaries\sr_model_d3_w32_x2_aug_3.hdf5")
