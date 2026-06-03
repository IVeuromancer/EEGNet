# EEGproj — BCI Motor Imagery Project

End-to-end brain-computer interface pipeline for decoding left vs. right arm motor imagery using an OpenBCI Cyton (8-channel, 250Hz). Collects EEG data, trains a CNN classifier, and uses predictions to control a pygame game in real time.

## Hardware

- **Board**: OpenBCI Cyton, 8 channels, 250Hz sampling rate
- **Connection**: Serial port (e.g. COM3 on Windows)
- **BrainFlow board ID**: `BoardIds.CYTON_BOARD`

## Environment

```bash
conda create -n EEGproj python=3.11
conda activate EEGproj
pip install brainflow mne scipy pygame torch mlflow scikit-learn numpy matplotlib
```

## Project structure

```
src/
  collect_data.py   — BrainFlow session recorder with pygame arrow cues
  preprocess.py     — 8–30Hz bandpass, baseline correction, artifact rejection, z-score
  dataset.py        — PyTorch Dataset + leave-one-session-out (LOSO) split helper
  model.py          — EEGNet (primary) and 1D U-Net (upgrade for 500+ trials)
  train.py          — LOSO cross-validation with MLflow experiment tracking
  realtime.py       — 4-thread pipeline: acquisition → preprocess → inference → decision
  game.py           — pygame dodge game controlled by EEG predictions

data/raw/           — raw .npz files from BrainFlow (gitignored)
data/processed/     — preprocessed dataset.npz + standardizer.npz (gitignored)
models/             — saved .pt checkpoints (gitignored)
results/            — plots, confusion matrices
```

## Data collection protocol

- 40 trials/session (20 left, 20 right, randomized)
- Trial: 2s fixation → 0.5s arrow cue → 4s motor imagery → 1.5s ITI
- Target: 5 sessions (~200 trials total)
- Saved as `data/raw/session_YYYY-MM-DD_HH-MM.npz` with keys `eeg` (n_trials, 8, 1000) and `labels`

## Preprocessing

- Bandpass 8–30Hz (mu + beta bands), 4th-order Butterworth
- Epoch: 0.5–4.5s post-cue → 1000 samples
- Baseline correct (subtract mean of first 125 samples)
- Artifact rejection: drop trials with any channel > ±150µV
- Z-score standardization fit on training set only

## Model

- **EEGNet**: default, best for <500 trials. Input `(B, 8, 1000)`.
- **UNet1D**: upgrade path for 500+ trials. Same input shape.
- Switch via `--arch eegnet` or `--arch unet1d` in `train.py`

## How to run

```bash
# Test data collection with synthetic board (no hardware needed)
python src/collect_data.py --board synthetic

# Preprocess all sessions
python src/preprocess.py

# Train with MLflow tracking
python src/train.py --arch eegnet

# Test game with keyboard (no model needed)
python src/game.py --demo

# Full live BCI pipeline
python src/game.py --model models/best_eegnet.pt --board cyton --port COM3
```

## MLflow experiment tracking

```bash
mlflow server  # runs on http://127.0.0.1:5000
```

Each training run is logged under experiment `EEGMotorImagery`, tracking hyperparameters, train/val loss and accuracy per epoch, and the best model artifact.
