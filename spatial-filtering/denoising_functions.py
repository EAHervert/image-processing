import cv2
import numpy as np

def gaussian_blurr_samples(samples, kernel, sigma):
    m, n = samples.shape[:2]
    out = np.zeros_like(samples)
    for i in range(m):
        for j in range(n):
            out[i, j] = cv2.GaussianBlur(samples[i, j], (kernel, kernel), sigma)

    return out

def median_blurr_samples(samples, kernel):
    m, n = samples.shape[:2]
    out = np.zeros_like(samples)
    for i in range(m):
        for j in range(n):
            out[i, j] = cv2.medianBlur(samples[i, j], kernel)

    return out
