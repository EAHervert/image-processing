import numpy as np
from skimage.restoration import denoise_tv_chambolle
from image_processing_utilities.functions import img_uint_to_float, img_float_to_uint
def total_variation_samples(samples, lambda_val, color=False):
    m, n = samples.shape[:2]
    out = np.zeros_like(samples)
    for i in range(m):
        for j in range(n):
            sample = img_uint_to_float(samples[i, j])
            if color:
                out[i, j] = img_float_to_uint(denoise_tv_chambolle(sample, weight=1 / lambda_val, channel_axis=-1))
            else:
                out[i, j] = img_float_to_uint(denoise_tv_chambolle(sample, weight=1 / lambda_val))

    return out
