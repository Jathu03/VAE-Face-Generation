# VAE Face Generation

This project implements a Variational Autoencoder (VAE) for generating faces using the CelebA dataset.

## Project Structure

- `src/`: Source code
  - `dataset.py`: CelebA dataset loader
  - `model.py`: VAE model definition
  - `loss.py`: VAE loss function
  - `train.py`: Training script
  - `generate.py`: Face generation script
  - `config.py`: Configuration parameters
- `main.py`: Entry point to run training and generation
- `requirements.txt`: Python dependencies
- `data/`: Directory for dataset (not included, download CelebA)

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download CelebA dataset and place images in `data/Images/`
4. Run `python main.py` to train and generate

## Usage

- Training: Modify `src/config.py` for parameters, run `python main.py`
- Generation: Use `generate_face` function in `src/generate.py`

## Note

The dataset path in `config.py` should point to the CelebA images directory.
