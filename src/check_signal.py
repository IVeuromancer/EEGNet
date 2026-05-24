"""
Continuous signal checker — streams 10s windows from all 8 EEG channels.
Use this to verify you're getting real signal before running a full session.

Usage:
    python src/check_signal.py --board synthetic
    python src/check_signal.py --board cyton --port COM7
    python src/check_signal.py --board cyton --port COM7 --raw   # skip notch/bandpass
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, iirnotch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

SRATE = 250
WINDOW_S = 10
N_SAMPLES = SRATE * WINDOW_S

CHANNEL_NAMES = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]


def make_board(board_type: str, port: str):
    params = BrainFlowInputParams()
    if board_type == "cyton":
        board_id = BoardIds.CYTON_BOARD
        params.serial_port = port
    else:
        board_id = BoardIds.SYNTHETIC_BOARD
    board = BoardShim(board_id, params)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    return board, eeg_channels


def apply_filters(eeg, raw=False):
    if raw:
        return eeg
    filtered = eeg.copy()
    # 60Hz notch (US power line)
    b_notch, a_notch = iirnotch(60.0, Q=30, fs=SRATE)
    # 1–50Hz bandpass to match OpenBCI GUI default view
    b_bp, a_bp = butter(4, [1.0, 50.0], btype="band", fs=SRATE)
    for i in range(filtered.shape[0]):
        filtered[i] = filtfilt(b_notch, a_notch, filtered[i])
        filtered[i] = filtfilt(b_bp, a_bp, filtered[i])
    return filtered


def collect_window(board, eeg_channels, raw=False):
    board.get_board_data()  # flush stale samples
    print(f"  Collecting {WINDOW_S}s ...", end=" ", flush=True)
    time.sleep(WINDOW_S)
    data = board.get_board_data()
    eeg = data[eeg_channels, :]        # (8, n_samples)
    # trim to exactly N_SAMPLES if we got more
    if eeg.shape[1] > N_SAMPLES:
        eeg = eeg[:, -N_SAMPLES:]
    print(f"got {eeg.shape[1]} samples across {len(eeg_channels)} channels")
    return eeg, apply_filters(eeg, raw=raw)


def print_stats(eeg_raw, eeg_filt):
    # stats on raw to catch railing; rms on filtered to match OpenBCI GUI display
    print(f"\n  {'Channel':<8} {'Raw Std':>9} {'Filt RMS':>9} {'Status'}")
    print("  " + "-" * 48)
    for i, name in enumerate(CHANNEL_NAMES):
        sd_raw = eeg_raw[i].std()
        rms_filt = np.sqrt(np.mean(eeg_filt[i] ** 2))
        rail_pct = np.mean(np.abs(eeg_raw[i]) > 180) * 100  # Cyton rails ~±187µV at 24x gain
        if sd_raw < 0.5:
            status = "FLAT — electrode disconnected?"
        elif rail_pct > 50:
            status = f"RAILED {rail_pct:.0f}% — poor contact or floating"
        elif rail_pct > 5:
            status = f"Partial rail {rail_pct:.0f}% — check gel/contact"
        elif rms_filt < 2:
            status = "very low — check contact"
        else:
            status = "OK"
        print(f"  {name:<8} {sd_raw:>8.1f}  {rms_filt:>8.1f}   {status}")
    print()


def plot_window(eeg_raw, eeg_filt, window_num):
    t = np.linspace(0, WINDOW_S, eeg_filt.shape[1])
    n_ch = eeg_filt.shape[0]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"EEG Signal Check — Window {window_num}  (filtered: 1–50 Hz + 60Hz notch)",
        fontsize=12,
    )
    gs = gridspec.GridSpec(n_ch, 1, hspace=0.08)

    for i in range(n_ch):
        ax = fig.add_subplot(gs[i])
        rail_pct = np.mean(np.abs(eeg_raw[i]) > 180) * 100
        color = "#e84c4c" if rail_pct > 20 else "#4c9be8"
        ax.plot(t, eeg_filt[i], lw=0.7, color=color)
        label = f"{CHANNEL_NAMES[i]}"
        if rail_pct > 5:
            label += f"\n{rail_pct:.0f}% railed"
        ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=38, va="center")
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, WINDOW_S)
        half = max(3 * eeg_filt[i].std(), 25)
        ax.set_ylim(-half, half)
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        if i < n_ch - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", choices=["cyton", "synthetic"], default="synthetic")
    parser.add_argument("--port", default="COM3", help="Serial port for Cyton (e.g. COM7)")
    parser.add_argument("--raw", action="store_true", help="Plot raw unfiltered signal")
    args = parser.parse_args()

    BoardShim.disable_board_logger()

    print(f"\nConnecting to {args.board} board...")
    board, eeg_channels = make_board(args.board, args.port)
    board.prepare_session()
    board.start_stream()
    print("Streaming started. Press Enter to collect a 10s window, or type 'q' + Enter to quit.\n")

    filter_label = "RAW (no filter)" if args.raw else "filtered 1–50Hz + 60Hz notch"
    print(f"Signal display: {filter_label}")
    print("Channels railed in RED. Fix BIAS/SRB2 reference first if multiple channels look bad.\n")

    window_num = 0
    try:
        while True:
            cmd = input("  [Enter = collect | q = quit] > ").strip().lower()
            if cmd == "q":
                break
            window_num += 1
            eeg_raw, eeg_filt = collect_window(board, eeg_channels, raw=args.raw)
            print_stats(eeg_raw, eeg_filt)
            plot_window(eeg_raw, eeg_filt, window_num)
    finally:
        board.stop_stream()
        board.release_session()
        print("Board released. Close any open plot windows.")
        plt.show()  # keep final plots open


if __name__ == "__main__":
    main()
