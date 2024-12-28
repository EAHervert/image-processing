import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_openml
import os
import scipy

def metrics_samples(samples, samples_gt, color=False):
    m, n = samples_gt.shape[:2]
    out = {'MAE': 0, 'MSE': 0, 'SSIM': 0, 'PSNR': 0}
    for i in range(m):
        for j in range(n):
            error = np.abs(samples_gt[i][j] - samples[i][j])

            out['MAE'] += error.mean()
            out['MSE'] += (error * error).mean()
            if color:
                out['SSIM'] += ssim(samples[i][j], samples_gt[i][j], channel_axis=-1)
            else:
                out['SSIM'] += ssim(samples[i][j], samples_gt[i][j])

            # Can have issue where psnr has division by zero problem
            try:
                val = psnr(samples[i][j], samples_gt[i][j], data_range=255)
            except ZeroDivisionError:
                val = 60.00
            out['PSNR'] += val

    out['MAE'] /= m * n
    out['MSE'] /= m * n
    out['SSIM'] /= m * n
    out['PSNR'] /= m * n

    return out


def validation_dataset_generator(dataset='SIDD'):
    # .mat files
    if dataset in ['SIDD', 'DIV2K_GSN_10', 'DIV2K_SNP_10']:
        return validation_dataset_generator_mat_file(dataset=dataset)
    elif dataset == 'Olivetti':
        return validation_dataset_generator_olivetti()
    elif dataset == 'USPS':
        return validation_dataset_generator_USPS()
    else:
        return False

    return x_noisy, x_gt


def validation_dataset_generator_mat_file(dataset='SIDD'):
    # get current directory
    path = os.getcwd()

    # prints parent directory
    par_path = os.path.abspath(os.path.join(path, os.pardir))

    # .mat files
    if dataset == 'SIDD':
        val_noisy = par_path + '/resources/sidd/validation/val_noisy.mat'  # Noisy
        val_gt = par_path + '/resources/sidd/validation/val_gt.mat'  # GT
        tag_noisy = 'ValidationNoisyBlocksSrgb'
        tag_gt = 'ValidationGtBlocksSrgb'
    elif dataset == 'DIV2K_GSN_10':
        val_noisy = par_path + '/resources/div2k_medium_gsn_10/validation/val_noisy.mat'  # Noisy
        val_gt = par_path + '/resources/div2k_medium_gsn_10/validation/val_gt.mat'  # GT
        tag_noisy = 'val_ng'
        tag_gt = 'val_gt'
    elif dataset == 'DIV2K_SNP_10':
        val_noisy = par_path + '/resources/div2k_medium_snp_10/validation/val_noisy.mat'  # Noisy
        val_gt = par_path + '/resources/div2k_medium_snp_10/validation/val_gt.mat'  # GT
        tag_noisy = 'val_ng'
        tag_gt = 'val_gt'
    else:
        return False

    # Load .mat files
    val_noisy_mat = np.array(scipy.io.loadmat(val_noisy)[tag_noisy])
    val_gt_mat = np.array(scipy.io.loadmat(val_gt)[tag_gt])

    return val_noisy_mat, val_gt_mat


def validation_dataset_generator_olivetti(snr_db=10):
    # Fix random seed and shuffle=False for consistency through the experiments
    np.random.seed(0)
    x = fetch_olivetti_faces(shuffle=False)['images'].reshape(20, 20, 64, 64) * 255
    x = x.astype(np.uint8)

    # Add noise to the images
    x_noisy = add_gaussian_noise(x, snr_db)

    return x_noisy, x


def validation_dataset_generator_USPS(snr_db=10):
    # Fix random seed and shuffle=False for consistency through the experiments
    np.random.seed(0)
    x, _ = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
    x = (x[:9216].reshape(96, 96, 16, 16) * 255).astype(np.uint8)

    # Add noise to the images
    x_noisy = add_gaussian_noise(x, snr_db)

    return x_noisy, x


def ssim_batch(x, y):
    out = 0
    for i in range(x.shape[0]):
        if x.ndim == 4:
            out += ssim(x[i], y[i], channel_axis=-1)
        elif x.ndim == 3:
            out += ssim(x[i], y[i])
        else:
            return -1

    return out / x.shape[0]


def add_gaussian_noise(image, snr):
    """Add Gaussian noise to an image with a given SNR."""
    signal_power = np.mean(image**2)  # Calculate signal power
    noise_power = signal_power / (10**(snr/10))  # Calculate noise power based on SNR
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)  # Generate Gaussian noise
    noisy_image = image + noise  # Add noise to the image
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip values to ensure they are within the valid range

    return noisy_image.astype(np.uint8)
