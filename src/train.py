import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from .dataset import CelebADataset, tensor_transforms
from .model import LinearVariationalAutoEncoder
from .loss import VAELoss
from .config import latent_dim, batch_size, learning_rate, training_iterations, evaluation_iterations, kl_weight, dataset_path

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = CelebADataset(root_dir=dataset_path, transform=tensor_transforms)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model and optimizer
    model = LinearVariationalAutoEncoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss tracking
    train_losses = []
    evaluation_losses = []

    pbar = tqdm(range(training_iterations))
    step_counter = 0
    train = True

    while train:
        for images in train_loader:
            images = images.to(device)
            encoded, mu, logvar, decoded = model(images)
            loss = VAELoss(images, decoded, mu, logvar, kl_weight)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step_counter % evaluation_iterations == 0:
                model.eval()
                eval_losses = []
                for eval_images in test_loader:
                    eval_images = eval_images.to(device)
                    _, mu_eval, logvar_eval, decoded_eval = model(eval_images)
                    eval_loss = VAELoss(eval_images, decoded_eval, mu_eval, logvar_eval, kl_weight)
                    eval_losses.append(eval_loss.item())
                evaluation_losses.append(np.mean(eval_losses))
                model.train()

            step_counter += 1
            pbar.update(1)

            if step_counter >= training_iterations:
                train = False
                break

    print(f"Final training loss: {train_losses[-1] if train_losses else 'N/A'}")
    print(f"Final evaluation loss: {evaluation_losses[-1] if evaluation_losses else 'N/A'}")

    return model, train_losses, evaluation_losses
