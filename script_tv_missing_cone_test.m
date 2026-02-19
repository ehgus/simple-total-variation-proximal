% TV Denoising Test Script - missing cone
% Solves: min_z[||x-z||^2/2 + 0.01*TV(z)]
% where x is the noisy image

clear; clc; close all;

% Load and prepare image
grid_index = -50:50;
k = 1;
x_grid = reshape(grid_index, [], 1);
y_grid = reshape(grid_index, 1, []);
z_grid = reshape(grid_index, 1, 1, []);
img3D = x_grid.^2 + y_grid.^2 + z_grid.^2 < 20^2;
img3D = single(img3D);

% Add missing cone
wedge_mask = ifftshift(abs(z_grid)* k < sqrt(x_grid.^2 + y_grid.^2));
img3D_fourier = fftn(img3D);
img3D_fourier(wedge_mask) = 0;
img3D_missing_cone = ifftn(img3D_fourier,'symmetric');


% TV denoising parameters
lambda = 1e-1;   % Regularization parameter
p = 2;           % L1 or L2 total variation
niter = 100;     % Number of iterations
norm_weight = 1; % Weight for the dual norm
use_gpu = true;  % Set to true to use GPU (if available)

% Iterative denoising using proximal gradient method
max_iter = 200;
step_size = 1e-2;
z = img3D_missing_cone;  % Initialize with noisy image

% Initialize TV regularizer
tv_reg = LpTotalVariation(lambda*step_size, p, niter, norm_weight, use_gpu);

fprintf('Starting TV denoising...\n');
tic;
t_np = 1;
v_np = z;
for iter = 1:max_iter
    t_n = t_np;
    v_n = v_np;
    % Gradient step: z = z - step_size * (z - x)
    z_fourier = fftn(z);
    z_fourier(~wedge_mask) = img3D_fourier(~wedge_mask);
    z = ifftn(z_fourier, 'symmetric');

    % Proximal step: apply TV regularizer
    v_np = tv_reg.proximal(z);
    % FISTA
    t_np = (1+sqrt(1+4*t_n^2))/2;
    z = v_np + ((t_n-1)/(t_np))*(v_np - v_n);
end
time_period = toc;
fprintf('Mean single iteration time: %f s\n', time_period/max_iter);

figure; orthosliceViewer(gather(img3D));
title('Original Image');

figure; orthosliceViewer(gather(img3D_missing_cone));
title(sprintf('Missing cone Image'));

figure; orthosliceViewer(gather(z), 'DisplayRange', [-0.6, 0.3]);
title(sprintf('TV Denoised (λ=%.3f)', lambda));

% Calculate metrics
mse_noisy = mean((img3D(:) - img3D_missing_cone(:)).^2);
mse_denoised = mean((img3D(:) - z(:)).^2);
psnr_noisy = 10*log10(1/mse_noisy);
psnr_denoised = 10*log10(1/mse_denoised);

fprintf('\nResults:\n');
fprintf('Noisy PSNR: %.2f dB\n', psnr_noisy);
fprintf('Denoised PSNR: %.2f dB\n', psnr_denoised);
fprintf('Improvement: %.2f dB\n', psnr_denoised - psnr_noisy);