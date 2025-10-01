% TV Denoising Test Script
% Solves: min_z[||x-z||^2/2 + 0.01*TV(z)]
% where x is the noisy image

clear; clc;

% Load and prepare image
img = imread('matlab-python-julia.png');
if size(img, 3) == 3
    img = rgb2gray(img);
end
img = double(img) / 255;  % Normalize to [0,1]

% Add noise
noise_level = 0.1;
rng(42);  % For reproducible results
noisy_img = img + noise_level * randn(size(img));
noisy_img = max(0, min(1, noisy_img));  % Clip to [0,1]
% noisy_img = single(noisy_img);
% noisy_img = gpuArray(noisy_img); % if you test it on noisy_img

% TV denoising parameters
lambda = 1e-1;   % Regularization parameter
p = 1;           % L1 or L2 total variation
niter = 50;      % Number of iterations
norm_weight = 1; % Weight for the dual norm

% Iterative denoising using proximal gradient method
max_iter = 50;
step_size = 1.0;
z = noisy_img;  % Initialize with noisy image

% Initialize TV regularizer
tv_reg = LpTotalVariation(lambda*step_size, p, niter);

fprintf('Starting TV denoising...\n');
tic;
for iter = 1:max_iter
    z_old = z;

    % Gradient step: z = z - step_size * (z - x)
    gradient = z - noisy_img;
    z = z - step_size * gradient;

    % Proximal step: apply TV regularizer
    z = tv_reg.proximal(z);
    % z = helper_fista_TV_inner_gpu(z, single(lambda*step_size), true, false, uint32(niter), single(-100), single(100), single(0), single(0), single(0));

    % Check convergence
    rel_change = norm(z(:) - z_old(:)) / norm(z_old(:));
end
time_period = toc;
fprintf('Mean single iteration time: %f s\n', time_period/max_iter);

% Display results
figure('Position', [100, 100, 1200, 400]);

subplot(1,3,1);
imshow(img);
title('Original Image');

subplot(1,3,2);
imshow(noisy_img);
title(sprintf('Noisy Image (σ=%.2f)', noise_level));

subplot(1,3,3);
imshow(z);
title(sprintf('TV Denoised (λ=%.3f)', lambda));

% Calculate metrics
mse_noisy = mean((img(:) - noisy_img(:)).^2);
mse_denoised = mean((img(:) - z(:)).^2);
psnr_noisy = 10*log10(1/mse_noisy);
psnr_denoised = 10*log10(1/mse_denoised);

fprintf('\nResults:\n');
fprintf('Noisy PSNR: %.2f dB\n', psnr_noisy);
fprintf('Denoised PSNR: %.2f dB\n', psnr_denoised);
fprintf('Improvement: %.2f dB\n', psnr_denoised - psnr_noisy);