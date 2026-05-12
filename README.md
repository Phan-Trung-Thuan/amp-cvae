# AMP-CVAE: Property-Guided Conditional Variational Autoencoder for Antimicrobial Peptides

This repository contains a PyTorch implementation of a Property-Guided Conditional Variational Autoencoder (CVAE) specifically designed for the discovery and generation of novel Antimicrobial Peptides (AMPs).

## Features

The model handles a complex dataset of antimicrobial peptides (`aps_database.csv`) incorporating multiple properties:
- **Amino Acid Sequence:** Processed as tokens.
- **Net Charge:** Modeled as an integer regression (range ~ -12 to 64).
- **Hydrophobic Residue %:** Modeled as an integer regression (range ~ 0 to 100).
- **3D Structure:** Multi-class classification across 7 unique classes.
- **Activities:** Multi-label classification across 27 unique functional classes.

## Architecture

The project serves two main purposes:

### 1. The Encoder (Property Detection & Latent Space Mapping)
The encoder processes peptide sequences and outputs:
- A 768-dimensional latent space (`Z_mean` and `Z_logvar`).
- Predictions for 2 regression targets (Charge and Hydrophobicity).
- Predictions for 7 structure classes.
- Predictions for 27 activity classes.

### 2. The Decoder (Conditional Generation)
The decoder synthesizes entirely novel peptides conditioned on desired properties. 
The input to the decoder is an 804-dimensional vector comprising:
- 768-dim random embedding (sampled from the latent space)
- 2 regression values
- 7 structure class probabilities
- 27 activity class probabilities

## Project Structure

- `src/dataset.py`: Data loading, tokenization, and preprocessing.
- `src/models.py`: CVAE model architecture (Encoder and Decoder).
- `src/loss.py`: Custom Multi-Task loss function (Reconstruction + KL Divergence + Property Predictions).
- `src/train.py`: Training loop.
- `src/generate.py`: Inference script for generating novel peptides based on target properties.
- `kaggle_notebook.ipynb`: A complete Kaggle notebook to clone the repo, install dependencies, train the model, and generate peptides.

## Usage

### Kaggle GPU Compatibility

**⚠️ Note for Kaggle Users:** Please use the **GPU T4x2** accelerator instead of **GPU P100**. The latest versions of PyTorch no longer support the older architecture of the P100 GPU (`sm_60`), which will cause a `CUDA error: no kernel image is available`. The T4 GPUs are fully supported.

### Training Results

The `kaggle_notebook.ipynb` will train all baseline sequence encoders (`rnn`, `lstm`, `gru`, `transformer`) for 100 epochs, display metrics with `tqdm` progress bars, and plot loss component charts (Reconstruction, KL, Charge, Hydrophobicity, Structure, and Activity) so that their final performances can be compared. Once trained, a comparison table will be printed showing the final validation metrics of each model.

### Training

To train the model on your local machine:
```bash
python src/train.py
```

### Generation

To generate new peptides:
```bash
python src/generate.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.