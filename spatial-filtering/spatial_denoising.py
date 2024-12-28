from image_processing_utilities.functions import metrics_samples
from image_processing_utilities.functions import validation_dataset_generator
from denoising_functions import gaussian_blurr_samples, median_blurr_samples
import numpy as np
from skimage.metrics import structural_similarity as ssim
import json
import argparse

parser = argparse.ArgumentParser(description='Spatial Denoising')

parser.add_argument('--dataset', type=str, default='SIDD')
parser.add_argument('--color', action='store_true')
parser.add_argument('--method', type=str, default='Gaussian')
args = parser.parse_args()

x_noisy, x_gt = validation_dataset_generator(dataset=args.dataset)
if args.dataset in ['SIDD', 'DIV2K_GSN_10', 'DIV2K_SNP_10']:
    x_noisy, x_gt = validation_dataset_generator(dataset=args.dataset)
    samples = json.load(open('config.json'))[args.dataset]
    x_noisy_samples = np.array([x_noisy[sample[0], sample[1], :, :, :] for sample in samples])
    x_gt_samples = np.array([x_gt[sample[0], sample[1], :, :, :] for sample in samples])
else:
    exit()

# Generate initial metrics:
print('\nMetrics:')
initial_metrics = metrics_samples(x_noisy, x_gt, color=args.color)
print('MAE: ', initial_metrics['MAE'])
print('MSE: ', initial_metrics['MSE'])
print('SSIM: ', initial_metrics['SSIM'])
print('PSNR: ', initial_metrics['PSNR'])

print('\nFinding optimal hyperparameters:')
if args.method == 'Gaussian':
    # Grid search find the best parameters:
    kernels = list(range(1, 51, 2))
    sigmas = list(np.arange(1, 15.25, .25))

    best_loss = 1e4
    best_kernel = None
    best_sigma = None
    best_index = None

    metrics = np.zeros((len(kernels), len(sigmas)))
    for i, kernel in enumerate(kernels):
        for j, sigma in enumerate(sigmas):
            test = gaussian_blurr_samples(x_noisy_samples, kernel, sigma)

            if args.color:
                avg_loss = 1 - ssim(test, x_gt_samples, channel_axis=-1)  # SSIM Loss
            else:
                avg_loss = 1 - ssim(test, x_gt_samples)  # SSIM Loss

            metrics[i, j] = avg_loss

            if avg_loss < best_loss:
                best_index = [i, j]
                best_kernel = kernel
                best_sigma = sigma
                best_loss = avg_loss

            print('Kernel: {kernel}, Sigma: {sigma}, Loss: {loss}'.format(kernel=kernel, sigma=sigma, loss=avg_loss))
            del test

    print('Best Parameters -- Kernel: {kernel}, Sigma: {sigma}'.format(kernel=best_kernel, sigma=best_sigma))
    denoised = gaussian_blurr_samples(x_noisy, best_kernel, best_sigma)

elif args.method == 'Median':
    kernels = list(range(1, 51, 2))
    best_loss = 1e4
    best_kernel = None
    best_sigma = None
    best_index = None

    metrics = np.zeros((len(kernels),))
    for i, kernel in enumerate(kernels):
        test = median_blurr_samples(x_noisy_samples, kernel)

        if args.color:
            avg_loss = 1 - ssim(test, x_gt_samples, channel_axis=-1)  # SSIM Loss
        else:
            avg_loss = 1 - ssim(test, x_gt_samples)  # SSIM Loss

        metrics[i] = avg_loss

        if avg_loss < best_loss:
            best_index = [i]
            best_kernel = kernel
            best_loss = avg_loss

        print('Kernel: {kernel}, Loss: {loss}'.format(kernel=kernel, loss=avg_loss))
        del test

    print('Best Parameters -- Kernel: {kernel}'.format(kernel=best_kernel))
    denoised = median_blurr_samples(x_noisy, best_kernel)

else:
    exit()

# Generate final metrics:
final_metrics = metrics_samples(denoised, x_gt, color=args.color)
del denoised

print('\nDenoised VALUES')
print('MAE: ', final_metrics['MAE'])
print('MSE: ', final_metrics['MSE'])
print('SSIM: ', final_metrics['SSIM'])
print('PSNR: ', final_metrics['PSNR'])

np.savetxt('data/{dataset}_{method}.csv'.format(dataset=args.dataset, method=args.method),
           metrics, delimiter=',')
