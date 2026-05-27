"""
Real-time EEG inference pipeline.

Architecture (four threads connected by queues):
    BrainFlow board → acquisition → preprocessing → inference → decision → game queue

Usage:
    python src/realtime.py --model models/best_eegnet.pt --board synthetic
    python src/realtime.py --model models/best_eegnet.pt --board cyton --port COM3
"""

import argparse
import queue
import threading
import time
import collections
from pathlib import Path

import numpy as np
import torch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

from src.model import build_model, EEGNet
from src.preprocess import bandpass_filter, baseline_correct, ChannelStandardizer

# ── constants ─────────────────────────────────────────────────────────────────
SRATE = 250
WINDOW_SAMPLES = 1000     # 4s window
STRIDE_SAMPLES = 125      # 0.5s stride → new prediction every 500ms
SMOOTH_N = 5              # majority vote over last N predictions
CONFIDENCE_THRESH = 0.55  # only act if max softmax > threshold

LABEL_LEFT = 0
LABEL_RIGHT = 1
LABEL_NONE = -1

DEVICE = torch.device("cpu")   # keep inference on CPU for real-time latency

MODEL_CHANNELS = [2, 3]   # CH3 (C4) and CH4 (C3) — indices into eeg_channels


# ── acquisition thread ────────────────────────────────────────────────────────
def acquisition_thread(board: BoardShim, eeg_channels: list[int],
                       raw_queue: queue.Queue, stop_event: threading.Event):
    """Continuously reads from the board and pushes sliding windows to raw_queue."""
    ring = collections.deque(maxlen=WINDOW_SAMPLES)
    samples_since_last = 0

    board.start_stream()
    print("[Acquisition] streaming started")

    while not stop_event.is_set():
        data = board.get_board_data()   # (n_channels, n_new_samples)
        if data.shape[1] == 0:
            time.sleep(0.02)
            continue

        eeg = data[eeg_channels, :]                  # (8, n_new)
        eeg = eeg[MODEL_CHANNELS, :]                 # (2, n_new) — CH3+CH4 only
        for i in range(eeg.shape[1]):
            ring.append(eeg[:, i])      # append column vectors
        samples_since_last += eeg.shape[1]

        if samples_since_last >= STRIDE_SAMPLES and len(ring) == WINDOW_SAMPLES:
            window = np.stack(list(ring), axis=1).astype(np.float32)  # (2, 1000)
            try:
                raw_queue.put_nowait(window)
            except queue.Full:
                pass   # drop oldest if consumer is slow
            samples_since_last = 0

        time.sleep(0.01)

    board.stop_stream()
    print("[Acquisition] stopped")


# ── preprocessing thread ──────────────────────────────────────────────────────
def preprocess_thread(raw_queue: queue.Queue, proc_queue: queue.Queue,
                      standardizer: ChannelStandardizer, stop_event: threading.Event):
    print("[Preprocess] ready")
    while not stop_event.is_set():
        try:
            window = raw_queue.get(timeout=0.1)   # (8, 1000)
        except queue.Empty:
            continue

        window = window * 1e-3   # BrainFlow nV → µV
        window = bandpass_filter(window, 8.0, 30.0, SRATE)
        window = baseline_correct(window)
        window = standardizer.transform(window[np.newaxis])[0]   # z-score (1, 2, 1000)

        try:
            proc_queue.put_nowait(window)
        except queue.Full:
            pass

    print("[Preprocess] stopped")


# ── inference thread ──────────────────────────────────────────────────────────
def inference_thread(model: torch.nn.Module, proc_queue: queue.Queue,
                     pred_queue: queue.Queue, stop_event: threading.Event):
    model.eval()
    print("[Inference] ready")

    with torch.no_grad():
        while not stop_event.is_set():
            try:
                window = proc_queue.get(timeout=0.1)   # (8, 1000)
            except queue.Empty:
                continue

            x = torch.from_numpy(window).unsqueeze(0).to(DEVICE, dtype=torch.float32)   # (1, 2, 1000)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()   # (2,)

            try:
                pred_queue.put_nowait(probs)
            except queue.Full:
                pass

    print("[Inference] stopped")


# ── decision thread ───────────────────────────────────────────────────────────
def decision_thread(pred_queue: queue.Queue, action_queue: queue.Queue,
                    stop_event: threading.Event):
    history = collections.deque(maxlen=SMOOTH_N)
    print("[Decision] ready")

    while not stop_event.is_set():
        try:
            probs = pred_queue.get(timeout=0.1)   # (2,)
        except queue.Empty:
            continue

        confidence = probs.max()
        pred = int(probs.argmax())
        history.append(pred)

        if confidence < CONFIDENCE_THRESH or len(history) < SMOOTH_N:
            action = LABEL_NONE
        else:
            # majority vote
            counts = [history.count(0), history.count(1)]
            action = int(np.argmax(counts))

        label = {LABEL_LEFT: "LEFT", LABEL_RIGHT: "RIGHT", LABEL_NONE: "---"}[action]
        print(f"[Decision] {label}  conf={confidence:.2f}  hist={list(history)}")

        try:
            action_queue.put_nowait(action)
        except queue.Full:
            pass

    print("[Decision] stopped")


# ── main ──────────────────────────────────────────────────────────────────────
def build_pipeline(model_path: str, standardizer_path: str,
                   board_type: str, port: str):
    # load model
    model = build_model("eegnet", n_channels=len(MODEL_CHANNELS), n_samples=1000, n_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).float()

    # load standardizer
    standardizer = ChannelStandardizer.load(standardizer_path)

    # board
    params = BrainFlowInputParams()
    if board_type == "cyton":
        board_id = BoardIds.CYTON_BOARD
        params.serial_port = port
    else:
        board_id = BoardIds.SYNTHETIC_BOARD

    board = BoardShim(board_id, params)
    board.prepare_session()
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    return model, standardizer, board, eeg_channels


def run_pipeline(model, standardizer, board, eeg_channels) -> queue.Queue:
    """Start all threads; return the action_queue for the game to consume."""
    raw_queue = queue.Queue(maxsize=10)
    proc_queue = queue.Queue(maxsize=10)
    pred_queue = queue.Queue(maxsize=10)
    action_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()

    threads = [
        threading.Thread(target=acquisition_thread,
                         args=(board, eeg_channels, raw_queue, stop_event), daemon=True),
        threading.Thread(target=preprocess_thread,
                         args=(raw_queue, proc_queue, standardizer, stop_event), daemon=True),
        threading.Thread(target=inference_thread,
                         args=(model, proc_queue, pred_queue, stop_event), daemon=True),
        threading.Thread(target=decision_thread,
                         args=(pred_queue, action_queue, stop_event), daemon=True),
    ]

    for t in threads:
        t.start()

    return action_queue, stop_event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_eegnet.pt")
    parser.add_argument("--stats", default="data/processed/standardizer_realtime.npz")
    parser.add_argument("--board", default="synthetic", choices=["cyton", "synthetic"])
    parser.add_argument("--port", default="COM3")
    args = parser.parse_args()

    model, standardizer, board, eeg_channels = build_pipeline(
        args.model, args.stats, args.board, args.port
    )

    action_queue, stop_event = run_pipeline(model, standardizer, board, eeg_channels)

    print("Pipeline running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        board.release_session()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()
