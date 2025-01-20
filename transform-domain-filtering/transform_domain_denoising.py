from image_processing_utilities.functions import metrics_samples
from image_processing_utilities.functions import validation_dataset_generator
from image_processing_utilities.functions import ssim_batch
from image_processing_utilities.functions import get_samples
from denoising_functions import fft_denoising, mask_a_b
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='Patch Based Denoising')

parser.add_argument('--dataset', type=str, default='SIDD')
parser.add_argument('--method', type=str, default='Fourier')
parser.add_argument('--shape', type=str, default='Diamond')
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
if args.method == 'Fourier':
    # Grid search find the best parameters:
    a_vals = list(range(1, 50, 1))
    b_vals = list(range(1, 50, 1))

    best_loss = 1e4
    best_a_val = None
    best_b_val = None
    best_index = None

    metrics = np.zeros((len(a_vals), len(b_vals)))
    for i, a_val in enumerate(a_vals):
        for j, b_val in enumerate(b_vals):
            mask = mask_a_b(x_noisy_samples[0, 0], a_val, b_val, shape=args.shape)
            # TODO: Fix fft_denoising
            test = fft_denoising(x_noisy_samples, mask)
            avg_loss = 1 - ssim_batch(test, x_gt_samples)  # SSIM Loss
            metrics[i, j] = avg_loss

            if avg_loss < best_loss:
                best_index = [i, j]
                best_a_val = a_val
                best_b_val = b_val
                best_loss = avg_loss

            print('A value: {a_val}, B value: {b_val}, Loss: {loss}'.format(a_val=a_val, b_val=b_val, loss=avg_loss))
            del mask, test

    print('Best Parameters -- A value: {a_val}, B value: {b_val}'.format(a_val=best_a_val, b_val=best_b_val))
    best_mask = mask_a_b(x_noisy_samples[0], best_a_val, best_b_val, shape=args.shape)
    denoised = fft_denoising(x_noisy_samples, best_mask)

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
