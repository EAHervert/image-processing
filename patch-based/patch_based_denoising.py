from image_processing_utilities.functions import metrics_samples
from image_processing_utilities.functions import validation_dataset_generator
from image_processing_utilities.functions import ssim_batch
from denoising_functions import nlm_samples, bm3d_samples
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='Patch Based Denoising')

parser.add_argument('--dataset', type=str, default='SIDD')
parser.add_argument('--method', type=str, default='BM3D')
args = parser.parse_args()

x_noisy, x_gt = validation_dataset_generator(dataset=args.dataset)
color = False
if args.dataset in ['SIDD', 'DIV2K_GSN_10', 'DIV2K_SNP_10']:
    color = True
    samples = json.load(open('config.json'))[args.dataset]
    x_noisy_samples = np.array([x_noisy[sample[0], sample[1], :, :, :] for sample in samples])
    x_gt_samples = np.array([x_gt[sample[0], sample[1], :, :, :] for sample in samples])
elif args.dataset in ['Olivetti']:
    x_noisy_samples = x_noisy[:10, :10]
    x_gt_samples = x_gt[:10, :10]
elif args.dataset in ['USPS']:
    x_noisy_samples = x_noisy[:16, :16]
    x_gt_samples = x_gt[:16, :16]
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
if args.method == 'NLM':
    # Grid search find the best parameters:
    h_vals = (np.arange(10, 200 + 1, 5) / 200)[1:]
    p_vals = np.arange(3, 15 + 1, 2)

    best_loss = 1e4
    best_h_val = None
    best_p_val = None
    best_index = None

    metrics = np.zeros((len(h_vals), len(p_vals)))
    for i, h_val in enumerate(h_vals):
        for j, p_val in enumerate(p_vals):
            test = nlm_samples(x_noisy_samples, h_val, p_val)
            avg_loss = 1 - ssim_batch(test.astype(np.single), x_gt_samples.astype(np.single))  # SSIM Loss
            metrics[i, j] = avg_loss

            if avg_loss < best_loss:
                best_index = [i, j]
                best_h_val = h_val
                best_p_val = p_val
                best_loss = avg_loss

            print('H: {h_val}, Patch Size: {p_val}, Loss: {loss}'.format(h_val=h_val, p_val=p_val, loss=avg_loss))
            del test

    print('Best Parameters -- H: {h_val}, Patch Size: {p_val}'.format(h_val=best_h_val, p_val=best_p_val))
    denoised = nlm_samples(x_noisy, best_h_val, best_p_val)

elif args.method == 'BM3D':
    sigmas = list(range(1, 51, 1))
    best_loss = 1e4
    best_sigma = None
    best_index = None

    metrics = np.zeros((len(sigmas),))
    for i, sigma in enumerate(sigmas):
        test = bm3d_samples(x_noisy_samples, sigma)
        avg_loss = 1 - ssim_batch(test, x_gt_samples)  # SSIM Loss
        metrics[i] = avg_loss

        if avg_loss < best_loss:
            best_index = [i]
            best_sigma = sigma
            best_loss = avg_loss

        print('Sigma: {sigma}, Loss: {loss}'.format(sigma=sigma, loss=avg_loss))
        del test

    print('Best Parameters -- Sigma: {sigma}'.format(sigma=best_sigma))
    denoised = bm3d_samples(x_noisy, best_sigma)

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
