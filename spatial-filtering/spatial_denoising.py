from image_processing_utilities.functions import metrics_samples
from image_processing_utilities.functions import validation_dataset_generator
from denoising_functions import gaussian_blurr_samples
import numpy as np

# TODO: Make these command line arguments
dataset = 'SIDD'
color = True
method = 'Gaussian'

x_noisy, x_gt = validation_dataset_generator(dataset=dataset)

# Generate metrics:
print('Metrics:\n')
initial_metrics = metrics_samples(x_noisy, x_gt, color=color)
print('MAE: ', initial_metrics['MAE'])
print('MSE: ', initial_metrics['MSE'])
print('SSIM: ', initial_metrics['SSIM'])
print('PSNR: ', initial_metrics['PSNR'])

if method == 'Gaussian':
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
            test = gaussian_blurr_samples(x_noisy, kernel, sigma)

            avg_loss = np.abs(test - x_gt).mean()  # L1 Loss
            metrics[i, j] = avg_loss

            if avg_loss < best_loss:
                best_index = [i, j]
                best_kernel = kernel
                best_sigma = sigma
                best_loss = avg_loss

            print('Kernel: {kernel}, Sigma: {sigma}'.format(kernel=kernel, sigma=sigma))

    print(best_kernel, best_sigma)

    denoised = gaussian_blurr_samples(x_noisy, best_kernel, best_sigma)

else:
    exit()

print('Denoised VALUES\n')
final_metrics = metrics_samples(denoised, x_gt, color=color)
print('MAE: ', final_metrics['MAE'])
print('MSE: ', final_metrics['MSE'])
print('SSIM: ', final_metrics['SSIM'])
print('PSNR: ', final_metrics['PSNR'])

np.savetxt('data/{dataset}_{method}.csv'.format(dataset=dataset, method=method),
           metrics, delimiter=',')

with open('data/{dataset}_{method}.npy'.format(dataset=dataset, method=method), 'wb') as f:
    np.save(f, denoised)
