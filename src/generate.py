import torch
import matplotlib.pyplot as plt
from .model import LinearVariationalAutoEncoder

def generate_face(model, latent_dim=20):
    z = torch.randn(latent_dim)
    z = z.unsqueeze(0).type(torch.float32)
    with torch.no_grad():
        generated = model.forward_dec(z)
    generated = generated.cpu().reshape(3, 32, 32).permute(1, 2, 0).numpy()  # Reshape and permute for imshow
    plt.imshow(generated)
    plt.show()
    return generated
