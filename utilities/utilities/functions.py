import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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
            val += ssim(samples[i][j], samples_gt[i][j], multichannel=True)

    return val / 16

def psnr_samples(samples, samples_gt):
    val = 0
    for i in range(4):
        for j in range(4):
            val += psnr(samples[i][j], samples_gt[i][j])

    return val / 16
