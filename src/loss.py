import torch

def VAELoss(x, x_hat, mean, log_var, kl_weight=1, reconstruction_weight=1):
    # Reconstruction loss
    pixel_mse = (x - x_hat) ** 2
    pixel_mse = pixel_mse.flatten(1)  # [batch_size, num_channels * height * width]
    reconstruction_loss = pixel_mse.sum(axis=-1).mean()
    # KL loss
    kl = (1 + log_var - mean**2 - torch.exp(log_var))
    kl_per_image = -0.5 * torch.sum(kl, axis=-1)
    kl_loss = torch.mean(kl_per_image)
    return reconstruction_loss * reconstruction_weight + kl_loss * kl_weight
