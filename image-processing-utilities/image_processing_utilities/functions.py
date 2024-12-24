import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
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
            out['PSNR'] += psnr(samples[i][j], samples_gt[i][j])

    out['MAE'] /= m * n
    out['MSE'] /= m * n
    out['SSIM'] /= m * n
    out['PSNR'] /= m * n

    return out


def validation_dataset_generator(dataset='SIDD'):
    # .mat files
    if dataset == 'SIDD':
        return validation_dataset_generator_mat_file()
    elif dataset == 'DIV2K_GSN_10':
        return validation_dataset_generator_mat_file(dataset='DIV2K_GSN_10')
    elif dataset == 'DIV2K_SNP_10':
        return validation_dataset_generator_mat_file(dataset='DIV2K_SNP_10')
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
