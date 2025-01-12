import numpy as np

def fft_denoising(samples, mask):
    m, n = samples.shape[:2]
    out = np.zeros_like(samples)
    for i in range(m):
        for j in range(n):
            out[i, j] = fft_image(samples[i, j], mask)

    return out

def fft_image(sample, mask):
    if sample.ndim == 3:
        out = np.zeros_like(sample)
        out[:, :, 0], out[:, :, 1], out[:, :, 2] = (denoise_fft(sample[:, :, 0], mask),
                                                    denoise_fft(sample[:, :, 1], mask),
                                                    denoise_fft(sample[:, :, 2], mask))
        return out
    else:
        return denoise_fft(sample, mask)

def denoise_fft(image, mask):
    transform = np.fft.fft2(image)  # Transforms the image to the frequency domain
    shifted_transform = np.fft.fftshift(transform)  # Shifts the image
    mask_transform = shifted_transform * mask  # Applies the mask
    inverse_shifted_mask_transform = np.fft.ifftshift(mask_transform)  # Inverse Shift
    inverse_transform = np.fft.ifft2(inverse_shifted_mask_transform)  # Inverse transform
    real_inverse_transform = np.abs(inverse_transform)  # Return only real values
    mask_image = np.clip(real_inverse_transform, 0, 255).astype(np.uint8)  # Final image

    return mask_image

def mask_a_b(image, a, b, shape='Diamond'):
    if image.ndim == 3:
        mask = np.zeros_like(image[:, :, 0])
    else:
        mask = np.zeros_like(image)
    m, n = mask.shape[:2]
    x, y = np.ogrid[:m, :n]

    # Mask shape
    if shape == 'Diamond':
        mask_area = abs((x - m // 2 - 1) / a) + abs((y - n // 2 - 1) / b) <= 1
    elif shape == 'ellipse':
        mask_area = ((x - m // 2 - 1) / a) ** 2 + ((y - n // 2 - 1) / b) ** 2 <= 1
    elif shape == 'star':
        mask_area = abs((x - m // 2 - 1) / a) ** 0.5 + abs((y - n // 2 - 1) / b) ** 0.5 <= 1
    else:
        return mask

    mask[mask_area] = 1  # Ones inside the mask and zeros outside the mask
    return mask
