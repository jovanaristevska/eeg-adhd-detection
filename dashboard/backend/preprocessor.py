# OVA E PRVATA VERZIJA NA PREPROCESSOR (TOA SO GO KORISTIME 01.06)
# import logging
# import numpy as np
# import pandas as pd
# import mne
# from scipy.signal import welch

# logger = logging.getLogger(__name__)

# CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
#             'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'FZ', 'CZ', 'PZ']

# SFREQ_INPUT = 128.0
# SFREQ_OUTPUT = 256.0
# WINDOW_SAMPLES = 1024  # 4 sec at 256 Hz
# VIZ_SAMPLES = 2560     # 10 sec at 256 Hz
# VIZ_CHANNELS = ['FP1', 'FP2', 'O1', 'T7']


# def _band_power(freqs, psd, f_low, f_high):
#     mask = (freqs >= f_low) & (freqs < f_high)
#     return float(np.trapz(psd[mask], freqs[mask]))


# def _band_status(band, power):
#     if band == 'delta':
#         return 'High' if power >= 50 else 'Normal'
#     elif band == 'theta':
#         return 'High' if power >= 60 else 'Normal'
#     elif band == 'alpha':
#         return 'High' if power >= 40 else 'Normal'
#     elif band == 'beta':
#         if power >= 70:
#             return 'High'
#         elif power >= 45:
#             return 'Slightly High'
#         return 'Normal'
#     elif band == 'gamma':
#         return 'High' if power >= 30 else 'Normal'
#     return 'Normal'


# class PreprocessorPipeline:
#     def preprocess(self, csv_file_path: str) -> dict:
#         df = pd.read_csv(csv_file_path)

#         # Case-insensitive column mapping
#         col_map = {c.upper(): c for c in df.columns}
#         missing = [ch for ch in CHANNELS if ch not in col_map]
#         if missing:
#             raise ValueError(f"Missing channels: {missing}")

#         # Extract only the 19 EEG channels in canonical order
#         eeg_df = df[[col_map[ch] for ch in CHANNELS]]
#         data = eeg_df.to_numpy(dtype=np.float64).T  # (19, n_timepoints)
#         # data *= 1e-6  # µV → V NO conversion to v - model trained on µV values

#         # Build MNE RawArray
#         info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ_INPUT,
#                                ch_types='eeg', verbose=False)
#         raw = mne.io.RawArray(data, info, verbose=False)

#         raw.notch_filter(freqs=50.0, verbose=False)
#         # raw.filter(l_freq=0.1, h_freq=60.0, verbose=False)  # max < Nyquist (64Hz)
#         raw.filter(l_freq=0.1, h_freq=None, verbose=False)
#         raw.resample(SFREQ_OUTPUT, verbose=False)

#         data_resampled = raw.get_data()  # (19, n_timepoints_resampled)

#         # Slice into non-overlapping windows
#         n_total = data_resampled.shape[1]
#         n_windows = n_total // WINDOW_SAMPLES
#         trimmed = data_resampled[:, :n_windows * WINDOW_SAMPLES]
#         windows = trimmed.reshape(19, n_windows, WINDOW_SAMPLES)
#         windows_tensor = windows.transpose(1, 0, 2)  # (N, 19, 1024)
#         # MAX_WINDOWS = 20
#         # if windows_tensor.shape[0] > MAX_WINDOWS:
#         #     windows_tensor = windows_tensor[:MAX_WINDOWS]
#         #     n_windows = MAX_WINDOWS

#         # EEG signal visualization (first 10 sec, selected channels)
#         viz_data = data_resampled[:, :VIZ_SAMPLES]
#         t = np.arange(viz_data.shape[1]) / SFREQ_OUTPUT
#         eeg_signal = {'time': t.tolist()}
#         for ch in VIZ_CHANNELS:
#             idx = CHANNELS.index(ch)
#             eeg_signal[ch] = (viz_data[idx]).tolist() 

#         # Band powers on mean of all channels
#         mean_signal = data_resampled.mean(axis=0)
#         freqs, psd = welch(mean_signal, fs=SFREQ_OUTPUT, nperseg=512)

#         # Scale to a displayable range (µV²/Hz)
#         psd_uv = psd

#         bands = {
#             'delta': (0.5, 4),
#             'theta': (4, 8),
#             'alpha': (8, 12),
#             'beta': (12, 30),
#             'gamma': (30, 45),
#         }
#         band_powers = {}
#         for band, (f_lo, f_hi) in bands.items():
#             power = _band_power(freqs, psd_uv, f_lo, f_hi)
#             band_powers[band] = {'power': round(power, 2),
#                                  'status': _band_status(band, power)}

#         return {
#             'windows_tensor': windows_tensor,
#             'eeg_signal': eeg_signal,
#             'band_powers': band_powers,
#             'n_windows': n_windows,
#         }


#OVA E VTORATA VERZIJA 
# """
# dashboard/backend/preprocessor.py — Training-aligned preprocessing.

# This version replicates EEG-FM-Bench training preprocessing EXACTLY:
#   - µV → V conversion (×1e-6) before filtering
#   - set_montage with standard_1020
#   - Filter ORDER: highpass → notch → resample (NOT notch first!)
#   - get_data(units='uV') after processing (proper unit handling)

# Replace dashboard/backend/preprocessor.py with this file, restart uvicorn,
# then test patient 151 on the dashboard again.
# """
# import logging
# import warnings

# import numpy as np
# import pandas as pd
# import mne
# from scipy.signal import welch

# logger = logging.getLogger(__name__)

# CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
#             'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
#             'FZ', 'CZ', 'PZ']

# SFREQ_INPUT    = 128.0
# SFREQ_OUTPUT   = 256.0
# WINDOW_SAMPLES = 1024  # 4 sec at 256 Hz
# VIZ_SAMPLES    = 2560  # 10 sec for visualization
# VIZ_CHANNELS   = ['FP1', 'FP2', 'O1', 'T7']


# def _band_power(freqs, psd, f_low, f_high):
#     mask = (freqs >= f_low) & (freqs < f_high)
#     return float(np.trapz(psd[mask], freqs[mask]))


# def _band_status(band, power):
#     if band == 'delta':
#         return 'High' if power >= 50 else 'Normal'
#     elif band == 'theta':
#         return 'High' if power >= 60 else 'Normal'
#     elif band == 'alpha':
#         return 'High' if power >= 40 else 'Normal'
#     elif band == 'beta':
#         if power >= 70:
#             return 'High'
#         elif power >= 45:
#             return 'Slightly High'
#         return 'Normal'
#     elif band == 'gamma':
#         return 'High' if power >= 30 else 'Normal'
#     return 'Normal'


# class PreprocessorPipeline:
#     """
#     Preprocessing pipeline aligned with EEG-FM-Bench training.

#     Pipeline steps (matching adhd.py._read_raw_data + builder.py._resample_and_filter):
#       1. Read CSV → select 19 channels (case-insensitive)
#       2. Transpose to (channels, time)
#       3. Multiply by 1e-6 (µV → V)            ← training does this
#       4. Create MNE RawArray with sfreq=128
#       5. Set standard_1020 montage              ← training does this
#       6. Apply highpass filter (l_freq=0.1, h_freq=None)
#          (h_freq=None because input fs=128 < 100*2=200)
#       7. Apply notch filter at 50 Hz            ← AFTER highpass, not before
#       8. Resample to 256 Hz
#       9. Extract data with get_data(units='uV') ← back to µV scale
#       10. Window into 4-sec non-overlapping segments → (N, 19, 1024)
#     """
#     def preprocess(self, csv_file_path: str) -> dict:
#         df = pd.read_csv(csv_file_path)

#         # ── 1. Channel selection (case-insensitive) ──────────────────────────
#         col_map = {c.upper(): c for c in df.columns}
#         missing = [ch for ch in CHANNELS if ch not in col_map]
#         if missing:
#             raise ValueError(f"Missing channels: {missing}")

#         eeg_df = df[[col_map[ch] for ch in CHANNELS]]
#         data = eeg_df.to_numpy(dtype=np.float64).T  # (19, n_timepoints)

#         # ── 2. µV → V conversion (training does this!) ───────────────────────
#         data = data * 1e-6

#         # ── 3. Build MNE Raw with sfreq=128, ch_types='eeg' ──────────────────
#         info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ_INPUT,
#                                ch_types='eeg', verbose=False)
#         raw = mne.io.RawArray(data, info, verbose=False)

#         # ── 4. Set standard 10-20 montage (training does this!) ──────────────
#         try:
#             dig_montage = mne.channels.make_standard_montage('standard_1020')
#             raw.set_montage(dig_montage, match_case=False,
#                             on_missing='ignore', verbose=False)
#         except Exception as exc:
#             logger.warning(f"set_montage failed (non-critical): {exc}")

#         # ── 5. Highpass filter FIRST (training order) ────────────────────────
#         # h_freq=None because input fs (128) < filter_high (100) * 2
#         # so MNE doesn't apply lowpass at 100 Hz — equivalent to "no lowpass"
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             raw = raw.filter(l_freq=0.1, h_freq=None, verbose=False)

#         # ── 6. Notch filter at 50 Hz SECOND (after highpass) ─────────────────
#         raw = raw.notch_filter(freqs=[50.0], verbose=False)

#         # ── 7. Resample to 256 Hz ────────────────────────────────────────────
#         raw = raw.resample(sfreq=SFREQ_OUTPUT, verbose=False)

#         # ── 8. Extract data in µV (training calls get_data(units='uV')!) ─────
#         data_resampled = raw.get_data(units='uV').astype(np.float64)

#         # ── 9. Window into 4-second non-overlapping segments ─────────────────
#         n_total   = data_resampled.shape[1]
#         n_windows = n_total // WINDOW_SAMPLES
#         if n_windows == 0:
#             raise ValueError(
#                 f"Recording too short for one 4-second window "
#                 f"({n_total} samples after resampling, need {WINDOW_SAMPLES})"
#             )

#         trimmed = data_resampled[:, :n_windows * WINDOW_SAMPLES]
#         windows = trimmed.reshape(19, n_windows, WINDOW_SAMPLES)
#         windows_tensor = windows.transpose(1, 0, 2)  # (N, 19, 1024)

#         # ── 10. Build visualization signal (first 10 sec) ────────────────────
#         viz_data = data_resampled[:, :VIZ_SAMPLES]
#         t = np.arange(viz_data.shape[1]) / SFREQ_OUTPUT
#         eeg_signal = {'time': t.tolist()}
#         for ch in VIZ_CHANNELS:
#             idx = CHANNELS.index(ch)
#             eeg_signal[ch] = viz_data[idx].tolist()

#         # ── 11. Compute band powers on mean of all channels ──────────────────
#         mean_signal = data_resampled.mean(axis=0)
#         freqs, psd = welch(mean_signal, fs=SFREQ_OUTPUT, nperseg=512)

#         bands = {
#             'delta': (0.5, 4),
#             'theta': (4, 8),
#             'alpha': (8, 12),
#             'beta':  (12, 30),
#             'gamma': (30, 45),
#         }
#         band_powers = {}
#         for band, (f_lo, f_hi) in bands.items():
#             power = _band_power(freqs, psd, f_lo, f_hi)
#             band_powers[band] = {
#                 'power':  round(power, 2),
#                 'status': _band_status(band, power),
#             }

#         logger.info(f"Preprocessed: {n_windows} windows, "
#                     f"data range: [{data_resampled.min():.2f}, {data_resampled.max():.2f}] µV")

#         return {
#             'windows_tensor': windows_tensor,
#             'eeg_signal':     eeg_signal,
#             'band_powers':    band_powers,
#             'n_windows':      n_windows,
#         }


























"""
dashboard/backend/preprocessor.py — Auto-detect format and preprocess.

Supports two EEG formats:
  - adhd_19: Clinical EEG with 19 channels (10-20 system)
  - crown_8: Neurosity Crown with 8 channels (10-10 system)

Pipeline matches EEG-FM-Bench training preprocessing:
  - µV → V conversion (×1e-6)
  - set_montage with standard_1005 (covers both 10-20 and 10-10)
  - Filter: highpass 0.1 Hz → notch 50 Hz → resample to 256 Hz
  - get_data(units='uV') to return values in µV
  - Window into 4-second non-overlapping segments (1024 samples @ 256 Hz)
"""
import logging
import warnings

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

logger = logging.getLogger(__name__)

# ─── CHANNEL CONFIGURATIONS ──────────────────────────────────────────────────
ADHD_19CH = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
             'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
             'FZ', 'CZ', 'PZ']

CROWN_8CH = ['F5', 'F6', 'C3', 'C4', 'CP3', 'CP4', 'PO3', 'PO4']

# Visualization channels for the EEG signal chart (one set per format)
VIZ_CHANNELS = {
    'adhd_19': ['FP1', 'FP2', 'O1', 'T7'],
    'crown_8': ['F5', 'F6', 'PO3', 'PO4'],
}

# ─── PIPELINE PARAMETERS ─────────────────────────────────────────────────────
SFREQ_INPUT    = 128.0
SFREQ_OUTPUT   = 256.0
WINDOW_SAMPLES = 1024   # 4 sec at 256 Hz
VIZ_SAMPLES    = 2560   # 10 sec for visualization


# ─── BAND POWER HELPERS ──────────────────────────────────────────────────────
def _band_power(freqs, psd, f_low, f_high):
    mask = (freqs >= f_low) & (freqs < f_high)
    return float(np.trapz(psd[mask], freqs[mask]))


def _band_status(band, power):
    if band == 'delta':
        return 'High' if power >= 50 else 'Normal'
    elif band == 'theta':
        return 'High' if power >= 60 else 'Normal'
    elif band == 'alpha':
        return 'High' if power >= 40 else 'Normal'
    elif band == 'beta':
        if power >= 70:
            return 'High'
        elif power >= 45:
            return 'Slightly High'
        return 'Normal'
    elif band == 'gamma':
        return 'High' if power >= 30 else 'Normal'
    return 'Normal'


# ─── FORMAT DETECTION ────────────────────────────────────────────────────────
def detect_format(df_columns: list):
    """
    Detect which EEG format is in the CSV by checking channel presence.

    Returns:
        (format_name, channel_list) where format_name is 'adhd_19' or 'crown_8'
        and channel_list is the canonical channel order for that format.

    Raises:
        ValueError if neither format is fully present.
    """
    upper_cols = {c.upper() for c in df_columns}

    has_all_19 = all(ch in upper_cols for ch in ADHD_19CH)
    has_all_8  = all(ch in upper_cols for ch in CROWN_8CH)

    if has_all_19:
        return ('adhd_19', ADHD_19CH)

    if has_all_8:
        return ('crown_8', CROWN_8CH)

    # Neither format detected — build a helpful error message
    missing_19 = [ch for ch in ADHD_19CH if ch not in upper_cols]
    missing_8  = [ch for ch in CROWN_8CH  if ch not in upper_cols]

    raise ValueError(
        f"CSV does not match any supported EEG format.\n"
        f"\n"
        f"Found channels: {sorted(upper_cols)}\n"
        f"\n"
        f"Required for Clinical EEG (19-channel): {ADHD_19CH}\n"
        f"  Missing: {missing_19}\n"
        f"\n"
        f"Required for Neurosity Crown (8-channel): {CROWN_8CH}\n"
        f"  Missing: {missing_8}"
    )


# ─── PIPELINE ────────────────────────────────────────────────────────────────
class PreprocessorPipeline:
    """
    Auto-detecting EEG preprocessing pipeline.

    For both 19-ch and 8-ch formats:
      1. Read CSV — detect format from channel names
      2. Extract EEG data in canonical channel order
      3. µV — V — MNE filtering — resampling — µV
      4. Window into 4-second non-overlapping segments
      5. Compute band powers (Delta/Theta/Alpha/Beta/Gamma)
      6. Build visualization signal (first 10 sec, format-specific channels)
    """
    def preprocess(self, csv_file_path: str) -> dict:
        df = pd.read_csv(csv_file_path)

        # 1. Detect format
        format_name, channels = detect_format(df.columns.tolist())
        n_chans = len(channels)
        logger.info(f"Detected format: {format_name} ({n_chans} channels)")

        # 2. Extract EEG data in canonical order (case-insensitive)
        col_map = {c.upper(): c for c in df.columns}
        eeg_cols = [col_map[ch] for ch in channels]
        data = df[eeg_cols].to_numpy(dtype=np.float64).T  # (n_chans, n_timepoints)

        # 3. uV - V conversion (training does this)
        data = data * 1e-6

        # 4. Build MNE Raw
        info = mne.create_info(ch_names=channels, sfreq=SFREQ_INPUT,
                               ch_types='eeg', verbose=False)
        raw = mne.io.RawArray(data, info, verbose=False)

        # 5. Set montage
        # standard_1005 contains BOTH 10-20 and 10-10 positions (covers both formats)
        try:
            dig_montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(dig_montage, match_case=False,
                            on_missing='ignore', verbose=False)
        except Exception as exc:
            logger.warning(f"set_montage failed (non-critical): {exc}")

        # 6. Filtering (training order: highpass - notch - resample)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = raw.filter(l_freq=0.1, h_freq=None, verbose=False)

        raw = raw.notch_filter(freqs=[50.0], verbose=False)
        raw = raw.resample(sfreq=SFREQ_OUTPUT, verbose=False)

        # 7. Extract data in uV
        data_resampled = raw.get_data(units='uV').astype(np.float64)

        # 8. Window into 4-sec non-overlapping segments
        n_total = data_resampled.shape[1]
        n_windows = n_total // WINDOW_SAMPLES
        if n_windows == 0:
            raise ValueError(
                f"Recording too short for one 4-second window "
                f"({n_total} samples after resampling, need {WINDOW_SAMPLES})"
            )

        trimmed = data_resampled[:, :n_windows * WINDOW_SAMPLES]
        windows = trimmed.reshape(n_chans, n_windows, WINDOW_SAMPLES)
        windows_tensor = windows.transpose(1, 0, 2)  # (N, n_chans, 1024)

        # 9. Build visualization signal (first 10 sec, format-specific)
        viz_chs = VIZ_CHANNELS[format_name]
        viz_data = data_resampled[:, :VIZ_SAMPLES]
        t = np.arange(viz_data.shape[1]) / SFREQ_OUTPUT
        eeg_signal = {'time': t.tolist()}
        for ch in viz_chs:
            if ch in channels:
                idx = channels.index(ch)
                eeg_signal[ch] = viz_data[idx].tolist()

        # 10. Compute band powers on mean of all channels
        mean_signal = data_resampled.mean(axis=0)
        freqs, psd = welch(mean_signal, fs=SFREQ_OUTPUT, nperseg=512)

        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta':  (12, 30),
            'gamma': (30, 45),
        }
        band_powers = {}
        for band, (f_lo, f_hi) in bands.items():
            power = _band_power(freqs, psd, f_lo, f_hi)
            band_powers[band] = {
                'power':  round(power, 2),
                'status': _band_status(band, power),
            }

        logger.info(f"Preprocessed [{format_name}]: {n_windows} windows, "
                    f"range [{data_resampled.min():.2f}, {data_resampled.max():.2f}] uV")

        return {
            'windows_tensor': windows_tensor,
            'eeg_signal':     eeg_signal,
            'band_powers':    band_powers,
            'n_windows':      n_windows,
            'format':         format_name,    # NEW: 'adhd_19' or 'crown_8'
            'n_channels':     n_chans,        # NEW: 19 or 8
            'channels':       channels,       # NEW: list of channel names
            'viz_channels':   viz_chs,        # NEW: which channels are in eeg_signal
        }


