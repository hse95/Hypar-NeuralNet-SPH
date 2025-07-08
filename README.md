# Hypar_NeuralNetwork_SPH
A repo for the study on Neural network model for predicting wave-induced pressure distribution on hyperbolic paraboloid free-surface breakwaters using SPH simulation data.

## Repository Structure
- `src/` contains Python modules for data loading, model definitions, training, and evaluation.
- `analysis/` holds the original data and notebooks used during development.
- `requirements.txt` lists Python dependencies.

## Usage
Install dependencies (preferably in a virtual environment):
```bash
pip install -r requirements.txt
```
Train a model:
```bash
python -m src.train --model cnn --epochs 100 --device cuda:0
```
Evaluate trained models:
```bash
python -m src.evaluate --device cuda:0
```
