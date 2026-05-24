"""
Data collection script for EEG motor imagery (left vs right arm).
Uses BrainFlow for Cyton streaming and pygame for cue display.

Usage:
    python src/collect_data.py --board cyton --port COM3
    python src/collect_data.py --board synthetic   # for testing
"""

import argparse
import time
import random
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter

# ── constants ────────────────────────────────────────────────────────────────
SRATE = 250          # Cyton sampling rate (Hz)
N_TRIALS = 6         # trials per session (20 left, 20 right)
FIXATION_S = 2.0     # fixation cross duration (s)
CUE_S = 0.5          # cue display duration (s)
IMAGERY_S = 4.0      # motor imagery window (s)
ITI_S = 1.5          # inter-trial interval (s)

EPOCH_START_S = 0.5  # seconds after cue onset to start epoch
EPOCH_END_S = 4.5    # seconds after cue onset to end epoch
EPOCH_SAMPLES = int((EPOCH_END_S - EPOCH_START_S) * SRATE)  # 1000 samples

LABEL_LEFT = 0
LABEL_RIGHT = 1

COLORS = {
    "bg": (30, 30, 30),
    "white": (240, 240, 240),
    "gray": (120, 120, 120),
    "left": (70, 130, 220),
    "right": (220, 90, 70),
}


# ── board setup ───────────────────────────────────────────────────────────────
def make_board(board_type: str, port: str) -> tuple[BoardShim, int, list[int]]:
    params = BrainFlowInputParams()
    if board_type == "cyton":
        board_id = BoardIds.CYTON_BOARD
        params.serial_port = port
    else:
        board_id = BoardIds.SYNTHETIC_BOARD

    board = BoardShim(board_id, params)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    return board, board_id, eeg_channels


# ── pygame cue display ────────────────────────────────────────────────────────
def pygame_sleep(seconds: float):
    """Sleep while keeping the pygame event loop alive."""
    end = time.time() + seconds
    while time.time() < end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                raise SystemExit("Session aborted by user.")
        pygame.time.wait(16)  # ~60fps tick


def draw_fixation(screen, font):
    screen.fill(COLORS["bg"])
    text = font.render("+", True, COLORS["white"])
    rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text, rect)
    pygame.display.flip()


def draw_cue(screen, font, label: int):
    screen.fill(COLORS["bg"])
    arrow = "←" if label == LABEL_LEFT else "→"
    color = COLORS["left"] if label == LABEL_LEFT else COLORS["right"]
    text = font.render(arrow, True, color)
    rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text, rect)
    pygame.display.flip()


def draw_rest(screen, font, trial_num: int, n_trials: int):
    screen.fill(COLORS["bg"])
    msg = f"Rest  ({trial_num}/{n_trials})"
    text = font.render(msg, True, COLORS["gray"])
    rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text, rect)
    pygame.display.flip()


def draw_done(screen, font):
    screen.fill(COLORS["bg"])
    text = font.render("Session complete!", True, COLORS["white"])
    rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text, rect)
    pygame.display.flip()


# ── data collection ───────────────────────────────────────────────────────────
# Sample indices from the per-trial flush point (constants)
_EPOCH_START_SAMP = int((FIXATION_S + EPOCH_START_S) * SRATE)   # 625
_EPOCH_END_SAMP   = _EPOCH_START_SAMP + EPOCH_SAMPLES            # 1625
_GRACE_S          = 0.15   # extra wait if timing jitter leaves us short


def collect_session(board: BoardShim, eeg_channels: list[int], labels: list[int],
                    screen, font) -> tuple[np.ndarray, np.ndarray]:
    epochs = []
    valid_labels = []

    board.start_stream()
    pygame_sleep(2)  # let buffer fill

    for i, label in enumerate(labels):
        # fixation
        draw_fixation(screen, font)
        board.get_board_data()  # flush buffer before trial
        pygame_sleep(FIXATION_S)

        # cue
        draw_cue(screen, font, label)
        pygame_sleep(CUE_S)

        # imagery window
        pygame_sleep(IMAGERY_S)

        # grab data; pygame_sleep jitter can leave us a few samples short,
        # so retry once with a short grace period before giving up
        raw = board.get_board_data()
        eeg = raw[eeg_channels, :]

        if eeg.shape[1] < _EPOCH_END_SAMP:
            pygame_sleep(_GRACE_S)
            extra = board.get_board_data()
            eeg = np.concatenate([eeg, extra[eeg_channels, :]], axis=1)

        if eeg.shape[1] >= _EPOCH_END_SAMP:
            epoch = eeg[:, _EPOCH_START_SAMP:_EPOCH_END_SAMP]   # (8, 1000)
            epochs.append(epoch)
            valid_labels.append(label)
        else:
            print(f"  Trial {i+1}: only {eeg.shape[1]}/{_EPOCH_END_SAMP} samples — skipping.")

        # inter-trial interval
        draw_rest(screen, font, i + 1, len(labels))
        pygame_sleep(ITI_S)

    board.stop_stream()

    X = np.stack(epochs, axis=0)   # (n_trials, 8, 1000)
    y = np.array(valid_labels)
    return X, y


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="synthetic", choices=["cyton", "synthetic"])
    parser.add_argument("--port", default="COM3", help="Serial port for Cyton (e.g. COM3)")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    # randomize trial order (balanced)
    half = args.trials // 2
    labels = [LABEL_LEFT] * half + [LABEL_RIGHT] * half
    random.shuffle(labels)

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("EEG Motor Imagery — Data Collection")
    font = pygame.font.SysFont("Arial", 120, bold=True)
    small_font = pygame.font.SysFont("Arial", 36)

    # show start screen
    screen.fill(COLORS["bg"])
    lines = [
        "EEG Motor Imagery Collection",
        "",
        "← = imagine moving LEFT arm",
        "→ = imagine moving RIGHT arm",
        "",
        "Press SPACE to begin",
    ]
    for j, line in enumerate(lines):
        surf = small_font.render(line, True, COLORS["white"])
        screen.blit(surf, (80, 100 + j * 60))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

    board, board_id, eeg_channels = make_board(args.board, args.port)
    board.prepare_session()

    try:
        X, y = collect_session(board, eeg_channels, labels, screen, font)
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to close...")
        return
    finally:
        board.release_session()

    draw_done(screen, font)
    pygame_sleep(2)
    pygame.quit()

    # save
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = out_dir / f"session_{timestamp}.npz"
    np.savez(out_path, eeg=X, labels=y)
    print(f"Saved {X.shape[0]} trials to {out_path}")
    print(f"  Shape: {X.shape}  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")


if __name__ == "__main__":
    main()
