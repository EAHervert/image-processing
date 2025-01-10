import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from bm3d import bm3d, BM3DStages

def nlm_samples(samples, h_val, p_val):
    m, n = samples.shape[:2]
    out = np.zeros_like(samples)
    for i in range(m):
        for j in range(n):

            # Apply non-local means denoising
            sigma_est = estimate_sigma(samples[i, j], h_val, p_val)  # Estimate the sigma value to use
            denoised_img = denoise_nl_means(samples[i, j], h=h_val, sigma=sigma_est, fast_mode=True, patch_size=p_val,
                                            patch_distance=15)
            out[i, j] = np.clip(denoised_img, 0, 255).astype(np.uint8)

def bm3d_samples(samples, sigma):
    m, n = samples.shape[:2]
    out = np.zeros_like(samples)
    for i in range(m):
        for j in range(n):
            # Apply bm3d denoising
            denoised_img = bm3d(samples[i, j], sigma_psd=sigma, stage_arg=BM3DStages.ALL_STAGES)
            out[i, j] = np.clip(denoised_img, 0, 255).astype(np.uint8)
