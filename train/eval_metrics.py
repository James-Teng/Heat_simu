from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(image_true, image_test, data_range=2.):
    return peak_signal_noise_ratio(
        image_true.cpu().numpy(),
        image_test.cpu().numpy(),
        data_range=data_range,
    )


def ssim(image_true, image_test, data_range=2.):
    return structural_similarity(
        image_true.cpu().numpy(),
        image_test.cpu().numpy(),
        data_range=data_range
    )

