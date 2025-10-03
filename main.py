from src.train import train_model
from src.generate import generate_face
from src.model import LinearVariationalAutoEncoder
from src.config import latent_dim

if __name__ == "__main__":
    # Train the model
    model, train_losses, eval_losses = train_model()

    # Generate a face
    generate_face(model, latent_dim)
