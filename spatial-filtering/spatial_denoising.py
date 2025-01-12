from image_processing_utilities.functions import metrics_samples
from image_processing_utilities.functions import validation_dataset_generator
from image_processing_utilities.functions import ssim_batch
from image_processing_utilities.functions import get_samples
from denoising_functions import gaussian_blurr_samples, median_blurr_samples
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='Spatial Denoising')

parser.add_argument('--dataset', type=str, default='SIDD')
parser.add_argument('--method', type=str, default='Gaussian')
args = parser.parse_args()

x_noisy, x_gt = validation_dataset_generator(dataset=args.dataset)
color = False
if args.dataset in ['SIDD', 'DIV2K_GSN_10', 'DIV2K_SNP_10']:
    color = True
    samples = json.load(open('config.json'))[args.dataset]
    x_noisy_samples = get_samples(x_noisy, samples)
    x_gt_samples = get_samples(x_gt, samples)
elif args.dataset in ['Olivetti']:
    x_noisy_samples = x_noisy[:10, :10]
    x_gt_samples = x_gt[:10, :10]
elif args.dataset in ['USPS']:
    x_noisy_samples = x_noisy[:64, :64]
    x_gt_samples = x_gt[:64, :64]
else:
    exit()

# Generate initial metrics:
print('\nMetrics:')
initial_metrics = metrics_samples(x_noisy.astype(np.single), x_gt.astype(np.single), color=color)
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
            avg_loss = 1 - ssim_batch(test.astype(np.single), x_gt_samples.astype(np.single))  # SSIM Loss
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
        avg_loss = 1 - ssim_batch(test, x_gt_samples)  # SSIM Loss
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
final_metrics = metrics_samples(denoised.astype(np.single), x_gt.astype(np.single), color=color)
del denoised

print('\nDenoised VALUES')
print('MAE: ', final_metrics['MAE'])
print('MSE: ', final_metrics['MSE'])
print('SSIM: ', final_metrics['SSIM'])
print('PSNR: ', final_metrics['PSNR'])

np.savetxt('data/{dataset}_{method}.csv'.format(dataset=args.dataset, method=args.method),
           metrics, delimiter=',')
