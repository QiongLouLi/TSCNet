# TSCNet

This repository provides the anonymous implementation of **TSCNet** for reproducibility during the double-blind review process.

## Main Contents

This repository contains the model code, experimental configurations, and checkpoints. The main structure is as follows:

- `models/`: model implementation;
- `layers/`: model components;
- `data_provider/`: data loading and processing;
- `exp/`: experiment running code;
- `utils/`: utility functions;
- `Configs/`: experimental configuration files;
- `checkpoints/`: model checkpoints.

## Configurations

The `Configs/` directory provides the experimental configurations for different loss functions:

- `Configs/HL_loss.png`: configurations for the HL loss;
- `Configs/MSE_loss.png/`: configurations for the MSE loss.

## Checkpoints

The `checkpoints/` directory provides model checkpoints trained with different loss functions:

- `checkpoints/HL_loss/`
- `checkpoints/L1_loss/`
- `checkpoints/MSE_loss/`

These checkpoints can be used for direct evaluation or for verifying the experimental results reported in the manuscript.

## Anonymous Review Statement

This repository is prepared for double-blind review. It does not contain author names, affiliations, email addresses, personal webpages, or other identifying information.
