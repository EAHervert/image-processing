import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import scipy

def l2_samples(samples, samples_gt):
    val = 0
    for i in range(4):
        for j in range(4):
            val += np.square(samples[i][j] - samples_gt[i][j]).mean()

    return val / 16

def ssim_samples(samples, samples_gt):
    val = 0
    for i in range(4):
        for j in range(4):
            val += ssim(samples[i][j], samples_gt[i][j], channel_axis=-1)

    return val / 16

def psnr_samples(samples, samples_gt):
    val = 0
    for i in range(4):
        for j in range(4):
            val += psnr(samples[i][j], samples_gt[i][j])

    return val / 16

def dataset_generator(dataset='SIDD'):
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
