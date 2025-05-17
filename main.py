# #audio quality only 


# import sys
# import time
# import numpy as np
# import sounddevice as sd
# import librosa
# from pesq import pesq
# import soundfile as sf

# # 1. Select audio device (same as before)
# if sys.platform == 'win32':
#     devices = sd.query_devices()
#     device_idx = next(
#         i for i, dev in enumerate(devices)
#         if dev['max_input_channels'] > 0 and 'loopback' in dev['name'].lower()
#     )
# else:
#     device_idx = sd.default.device[0]

# # 2. Stream parameters
# SAMPLERATE = 48000
# BLOCKSIZE = 2048

# # 3. Throttle settings
# PRINT_INTERVAL = 1.0  # seconds between prints
# last_print_time = 0

# def categorize(value, good_threshold, moderate_threshold, invert=False):
#     if invert:
#         if value <= good_threshold:      return "Good"
#         if value <= moderate_threshold:  return "Moderate"
#     else:
#         if value >= good_threshold:      return "Good"
#         if value >= moderate_threshold:  return "Moderate"
#     return "Bad"

# def audio_callback(indata, frames, time_info, status):
#     global last_print_time

#     # compute metrics
#     data = indata[:, 0]
#     rms      = np.sqrt(np.mean(data**2))
#     peak     = np.max(np.abs(data))
#     clipping = int(np.sum(np.abs(data) >= 0.99))
#     S        = np.abs(librosa.stft(data, n_fft=2048))
#     flatness = float(librosa.feature.spectral_flatness(S=S).mean())
#     zcr      = float(librosa.feature.zero_crossing_rate(data).mean())

#     # categorize
#     rms_cat   = categorize(rms,      0.05, 0.02)
#     peak_cat  = categorize(peak,     0.5,  0.2)
#     clip_cat  = categorize(clipping, 0,    5,    invert=True)
#     flat_cat  = categorize(flatness, 0.2,  0.4,  invert=True)
#     zcr_cat   = categorize(zcr,      0.02, 0.05, invert=True)

#     # overall
#     cats = [rms_cat, peak_cat, clip_cat, flat_cat, zcr_cat]
#     overall = "Bad" if "Bad" in cats else ("Moderate" if "Moderate" in cats else "Good")

#     # only print if enough time has passed
#     now = time.time()
#     if now - last_print_time >= PRINT_INTERVAL:
#         last_print_time = now
#         print("\n🔊 Audio Quality:", overall)
#         print(f"  • RMS Level:         {rms:.4f} ({rms_cat})")
#         print(f"  • Peak Level:        {peak:.4f} ({peak_cat})")
#         print(f"  • Clipping Count:    {clipping} ({clip_cat})")
#         print(f"  • Spectral Flatness: {flatness:.4f} ({flat_cat})")
#         print(f"  • Zero-Cross Rate:   {zcr:.4f} ({zcr_cat})")

# def compute_pesq(reference_path, captured_path):
#     ref, sr = sf.read(reference_path)
#     deg, _  = sf.read(captured_path)
#     return pesq(ref, deg, sr)

# if __name__ == "__main__":
#     print(f"Using device index: {device_idx} ({sd.query_devices(device_idx)['name']})")
#     stream = sd.InputStream(
#         device=device_idx,
#         channels=1,
#         samplerate=SAMPLERATE,
#         blocksize=BLOCKSIZE,
#         callback=audio_callback
#     )
#     stream.start()
#     print(f"Monitoring audio quality... output every {PRINT_INTERVAL} second(s). Press Ctrl+C to stop.")
#     try:
#         while True:
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         print("\nStopped by user.")
#     finally:
#         stream.stop()
#         stream.close()
























# import sys
# import time
# import numpy as np
# import sounddevice as sd
# import librosa
# import webrtcvad       # uv add webrtcvad-wheels
# from pesq import pesq  # uv add install pesq
# import soundfile as sf

# # --- Configuration ---
# SAMPLE_RATE_CAPTURE = 48000    # capture at 48 kHz
# BLOCKSIZE = 2048               # frames per block from sounddevice
# VAD_RATE = 16000               # WebRTC VAD works at 8/16/32/48 kHz; we'll resample
# FRAME_MS = 30                  # ms per VAD frame
# PRINT_INTERVAL = 1.0           # seconds between printed summaries
# VAD_AGGRESSIVENESS = 2         # 0–3, higher = more aggressive speech detection

# # --- Select Loopback Device for System Audio ---
# if sys.platform == 'win32':
#     devices = sd.query_devices()
#     device_idx = next(
#         i for i, d in enumerate(devices)
#         if d['max_input_channels'] > 0 and 'loopback' in d['name'].lower()
#     )
# else:
#     # On macOS/Linux, set up a virtual audio device (e.g. BlackHole) as default input
#     device_idx = sd.default.device[0]

# # --- Voice Activity Detector ---
# vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
# FRAME_SIZE = int(VAD_RATE * FRAME_MS / 1000)  # samples per VAD frame

# # --- Helper: Categorize Metric Values ---
# def categorize(val, good, moderate, invert=False):
#     """
#     Classify val into 'Good', 'Moderate', or 'Bad'.
#     If invert=True, lower values are better.
#     """
#     if invert:
#         if val <= good:      return "Good"
#         if val <= moderate:  return "Moderate"
#     else:
#         if val >= good:      return "Good"
#         if val >= moderate:  return "Moderate"
#     return "Bad"

# last_print = 0.0

# # --- Audio Callback ---
# def callback(indata, frames, time_info, status):
#     global last_print

#     # Convert to mono float32
#     mono = indata.mean(axis=1)

#     # Resample to VAD rate
#     if SAMPLE_RATE_CAPTURE != VAD_RATE:
#         mono_vad = librosa.resample(mono, SAMPLE_RATE_CAPTURE, VAD_RATE)
#     else:
#         mono_vad = mono

#     # Split into VAD frames and detect speech
#     speech_flags = []
#     for start in range(0, len(mono_vad), FRAME_SIZE):
#         frame = mono_vad[start:start+FRAME_SIZE]
#         if len(frame) < FRAME_SIZE:
#             break
#         pcm = (frame * 32767).astype(np.int16).tobytes()
#         speech_flags.append(vad.is_speech(pcm, VAD_RATE))

#     # If no speech detected, skip processing
#     if not any(speech_flags):
#         return

#     # Compute quality metrics on full block (original sample rate)
#     data = indata[:, 0]
#     rms      = np.sqrt(np.mean(data**2))
#     peak     = np.max(np.abs(data))
#     clipping = int(np.sum(np.abs(data) >= 0.99))
#     S        = np.abs(librosa.stft(data, n_fft=2048))
#     flatness = float(librosa.feature.spectral_flatness(S=S).mean())
#     zcr      = float(librosa.feature.zero_crossing_rate(data).mean())

#     # Categorize metrics
#     rms_cat   = categorize(rms,      good=0.05, moderate=0.02)
#     peak_cat  = categorize(peak,     good=0.5,  moderate=0.2)
#     clip_cat  = categorize(clipping, good=0,    moderate=5, invert=True)
#     flat_cat  = categorize(flatness, good=0.2,  moderate=0.4, invert=True)
#     zcr_cat   = categorize(zcr,      good=0.02, moderate=0.05, invert=True)

#     # Overall quality = worst of all categories
#     cats = [rms_cat, peak_cat, clip_cat, flat_cat, zcr_cat]
#     if "Bad" in cats:
#         overall = "Bad"
#     elif "Moderate" in cats:
#         overall = "Moderate"
#     else:
#         overall = "Good"

#     # Throttle printing
#     now = time.time()
#     if now - last_print < PRINT_INTERVAL:
#         return
#     last_print = now

#     # Print user-friendly summary
#     print(f"\n🔊 Detected speech — Quality: {overall}")
#     print(f"  • RMS Level:         {rms:.4f} ({rms_cat})")
#     print(f"  • Peak Level:        {peak:.4f} ({peak_cat})")
#     print(f"  • Clipping Count:    {clipping} ({clip_cat})")
#     print(f"  • Spectral Flatness: {flatness:.4f} ({flat_cat})")
#     print(f"  • Zero-Cross Rate:   {zcr:.4f} ({zcr_cat})")

# # --- Optional: PESQ Scoring Function ---
# def compute_pesq(ref_path, cap_path):
#     ref, sr = sf.read(ref_path)
#     deg, _  = sf.read(cap_path)
#     return pesq(ref, deg, sr)

# # --- Main Execution ---
# if __name__ == "__main__":
#     dev_name = sd.query_devices(device_idx)['name']
#     print(f"Using device #{device_idx}: {dev_name}")
#     print("Capturing system audio; press Ctrl+C to stop.")
#     stream = sd.InputStream(
#         device=device_idx,
#         channels=1,
#         samplerate=SAMPLE_RATE_CAPTURE,
#         blocksize=BLOCKSIZE,
#         callback=callback
#     )
#     stream.start()
#     try:
#         while True:
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         print("\nStopped by user.")
#     finally:
#         stream.stop()
#         stream.close()










# normal speech quality 



import sys, time, threading, queue
import numpy as np
import sounddevice as sd
import librosa
import webrtcvad
from pesq import pesq      # optional
import soundfile as sf

# --- Configuration ---
CAPTURE_RATE = 16000        # capture directly at VAD rate
BLOCKSIZE    = 480          # ~30 ms blocks at 16 kHz
PRINT_INT    = 1.0          # print once per second
VAD_AGG      = 2            # aggressiveness level (0–3)
N_FFT        = 256          # <= BLOCKSIZE to avoid warnings

# --- Setup VAD and queue ---
vad = webrtcvad.Vad(VAD_AGG)
audio_queue = queue.Queue()
last_print = 0.0

# --- Callback: only enqueue data ---
def callback(indata, frames, time_info, status):
    try:
        mono = indata.mean(axis=1)
        audio_queue.put_nowait(mono.copy())
    except queue.Full:
        pass
    except Exception:
        raise sd.CallbackStop

# --- Worker thread: process blocks ---
def worker():
    global last_print
    while True:
        data = audio_queue.get()
        if data is None:
            break

        # VAD decision on this frame
        pcm = (data * 32767).astype(np.int16).tobytes()
        is_speech = vad.is_speech(pcm, CAPTURE_RATE)
        if not is_speech:
            continue

        # Compute metrics
        rms       = np.sqrt(np.mean(data**2))
        peak      = np.max(np.abs(data))
        clips     = int(np.sum(np.abs(data) >= 0.99))
        S         = np.abs(librosa.stft(data, n_fft=N_FFT))
        flatness  = float(librosa.feature.spectral_flatness(S=S).mean())
        zcr       = float(librosa.feature.zero_crossing_rate(data).mean())

        # Categorize
        def cat(v, g, m, inv=False):
            if inv:
                return "Good" if v <= g else ("Moderate" if v <= m else "Bad")
            return "Good" if v >= g else ("Moderate" if v >= m else "Bad")

        cats = {
            "RMS":      cat(rms,      0.05, 0.02),
            "Peak":     cat(peak,     0.5,  0.2),
            "Clips":    cat(clips,    0,    5, True),
            "Flatness": cat(flatness, 0.2,  0.4, True),
            "ZCR":      cat(zcr,      0.02, 0.05, True)
        }
        overall = ("Bad" if "Bad" in cats.values()
                   else "Moderate" if "Moderate" in cats.values()
                   else "Good")

        # Throttle printing
        now = time.time()
        if now - last_print < PRINT_INT:
            continue
        last_print = now

        print(f"\n🔊 Speech Quality: {overall}")
        for name, rating in cats.items():
            val = locals()[name.lower()] if name != "Clips" else clips
            print(f"  • {name}: {val:.4f} ({rating})")

# --- Main: start stream and worker ---
if __name__ == "__main__":
    # Select loopback device on Windows, default elsewhere
    if sys.platform == 'win32':
        devs = sd.query_devices()
        idx  = next(i for i,d in enumerate(devs)
                    if d['max_input_channels']>0 and 'loopback' in d['name'].lower())
    else:
        idx = sd.default.device[0]

    name = sd.query_devices(idx)['name']
    print(f"Using device #{idx}: {name}")
    print("Capturing system audio at 16 kHz; press Ctrl+C to stop.")

    # Start worker
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    # Open stream
    stream = sd.InputStream(
        device=idx,
        channels=1,
        samplerate=CAPTURE_RATE,
        blocksize=BLOCKSIZE,
        callback=callback
    )
    stream.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        audio_queue.put(None)
        th.join()
        print("Stopped.")
















# import sys
# import time
# import numpy as np
# import sounddevice as sd
# import torch
# import librosa

# from torchmetrics.audio import (
#     DeepNoiseSuppressionMeanOpinionScore,
#     NonIntrusiveSpeechQualityAssessment
# )

# # 1. Select audio device
# if sys.platform == 'win32':
#     devices = sd.query_devices()
#     device_idx = next(
#         i for i, dev in enumerate(devices)
#         if dev['max_input_channels'] > 0 and 'loopback' in dev['name'].lower()
#     )
# else:
#     device_idx = sd.default.device[0]

# print(f"Using device index: {device_idx} ({sd.query_devices(device_idx)['name']})")

# # 2. Stream parameters
# SAMPLERATE     = 48000
# BLOCKSIZE      = 2048
# PRINT_INTERVAL = 1.0  # seconds between outputs

# # 3. Speech-quality thresholds
# GOOD_SPEECH_THRESH    = 4.0
# MOD_SPEECH_THRESH     = 3.0

# # 4. Helper for categorization
# def categorize(value, good_threshold, moderate_threshold, invert=False):
#     if invert:
#         if value <= good_threshold:      return "Good"
#         if value <= moderate_threshold:  return "Moderate"
#     else:
#         if value >= good_threshold:      return "Good"
#         if value >= moderate_threshold:  return "Moderate"
#     return "Bad"

# # 5. Initialize TorchMetrics on-device models
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # DNSMOS: gives [p808_mos, mos_sig, mos_bak, mos_ovr]
# dnsmos = DeepNoiseSuppressionMeanOpinionScore(
#     fs=SAMPLERATE, personalized=False
# ).to(device)

# # NISQA: gives [MOS, noisiness, discontinuity, coloration, loudness]
# nisqa = NonIntrusiveSpeechQualityAssessment(
#     fs=SAMPLERATE
# ).to(device)

# last_print_time = 0

# def audio_callback(indata, frames, time_info, status):
#     global last_print_time

#     # 1) Raw audio array
#     data = indata[:, 0].astype(np.float32)

#     # 2) Compute speech MOS scores
#     audio_tensor = torch.from_numpy(data).to(device)
#     with torch.no_grad():
#         dnsmos_scores = dnsmos(audio_tensor.unsqueeze(0))
#         nisqa_scores  = nisqa(audio_tensor.unsqueeze(0))

#     dnsmos_ovr = dnsmos_scores[0, -1].item()
#     nisqa_mos  = nisqa_scores[0, 0].item()

#     # 3) Overall speech MOS as average
#     avg_speech_mos = (dnsmos_ovr + nisqa_mos) / 2.0
#     speech_cat     = categorize(avg_speech_mos, GOOD_SPEECH_THRESH, MOD_SPEECH_THRESH)

#     # 4) Compute audio-signal quality features
#     rms      = np.sqrt(np.mean(data**2))
#     peak     = np.max(np.abs(data))
#     clipping = int(np.sum(np.abs(data) >= 0.99))
#     S        = np.abs(librosa.stft(data, n_fft=2048))
#     flatness = float(librosa.feature.spectral_flatness(S=S).mean())
#     zcr      = float(librosa.feature.zero_crossing_rate(data).mean())

#     rms_cat   = categorize(rms,      0.05, 0.02)
#     peak_cat  = categorize(peak,     0.5,  0.2)
#     clip_cat  = categorize(clipping, 0,    5,    invert=True)
#     flat_cat  = categorize(flatness, 0.2,  0.4,  invert=True)
#     zcr_cat   = categorize(zcr,      0.02, 0.05, invert=True)

#     # 5) Print throttled output
#     now = time.time()
#     if now - last_print_time >= PRINT_INTERVAL:
#         last_print_time = now

#         # Speech-quality section
#         print("\n🔈 Speech Quality Assessment:")
#         print(f"  • DNSMOS Overall MOS:   {dnsmos_ovr:.2f}")
#         print(f"  • NISQA Overall MOS:    {nisqa_mos:.2f}")
#         print(f"  • Avg Speech MOS:       {avg_speech_mos:.2f} ({speech_cat})")

#         # Audio-signal quality section
#         print("\n🎚️ Audio Signal Quality:")
#         print(f"  • RMS Level:           {rms:.4f} ({rms_cat})")
#         print(f"  • Peak Level:          {peak:.4f} ({peak_cat})")
#         print(f"  • Clipping Count:      {clipping}   ({clip_cat})")
#         print(f"  • Spectral Flatness:   {flatness:.4f} ({flat_cat})")
#         print(f"  • Zero-Cross Rate:     {zcr:.4f} ({zcr_cat})")

# def main():
#     stream = sd.InputStream(
#         device=device_idx,
#         channels=1,
#         samplerate=SAMPLERATE,
#         blocksize=BLOCKSIZE,
#         callback=audio_callback
#     )
#     stream.start()
#     print(f"Monitoring live call quality… updates every {PRINT_INTERVAL}s. Press Ctrl+C to stop.")
#     try:
#         while True:
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         print("\nStopped by user.")
#     finally:
#         stream.stop()
#         stream.close()

# if __name__ == "__main__":
#     main()




#for both audio and speech quality check on any device that is connected 

import sys
import time
import numpy as np
import sounddevice as sd
import torch
import librosa
from collections import deque
from torchmetrics.audio import (
    DeepNoiseSuppressionMeanOpinionScore,
    NonIntrusiveSpeechQualityAssessment
)

# 1. List all devices
print("Available audio devices:")
devices = sd.query_devices()
for idx, dev in enumerate(devices):
    io = []
    if dev['max_input_channels'] > 0:
        io.append("input")
    if dev['max_output_channels'] > 0:
        io.append("output")
    io_str = "/".join(io) or "—"
    print(f"  [{idx:2d}] {dev['name']} ({io_str})")

# 2. Prompt user for device index
while True:
    choice = input("\nEnter the index of the device you’d like to use for recording: ")
    if not choice.isdigit():
        print("  Please enter a number.")
        continue
    idx = int(choice)
    if idx < 0 or idx >= len(devices):
        print(f"  Invalid index: must be between 0 and {len(devices)-1}.")
        continue
    if devices[idx]['max_input_channels'] < 1:
        print("  That device has no input channels—choose another.")
        continue
    device_idx = idx
    break

print(f"\nUsing device #{device_idx}: {devices[device_idx]['name']}")

# # 1. Audio device selection
# if sys.platform == 'win32':
#     devices = sd.query_devices()
#     device_idx = next(
#         i for i, d in enumerate(devices)
#         if d['max_input_channels'] > 0 and 'loopback' in d['name'].lower()
#     )
# else:
#     device_idx = sd.default.device[0]
# print(f"Using device: {sd.query_devices(device_idx)['name']}")

# 2. Stream & buffer params
SR             = 48000              # sampling rate
BLOCK          = 2048               # ~0.043 s blocks
PRINT_INTERVAL = 1.0                # seconds between updates
BUF_SECS       = 3.0                # NISQA needs ≥3s at once
BUF_SIZE       = int(SR * BUF_SECS)

# 3. Ring buffer for last 3s of audio
ring_buffer = deque(maxlen=BUF_SIZE)

# 4. Speech-quality thresholds
GOOD_THRESH = 4.0
MOD_THRESH  = 3.0

def categorize(val, good, mod, invert=False):
    if invert:
        if val <= good:  return "Good"
        if val <= mod:   return "Moderate"
    else:
        if val >= good:  return "Good"
        if val >= mod:   return "Moderate"
    return "Bad"

# 5. Initialize TorchMetrics (requires onnxruntime, librosa, requests)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dnsmos = DeepNoiseSuppressionMeanOpinionScore(
    fs=SR,
    personalized=False
).to(device)
nisqa  = NonIntrusiveSpeechQualityAssessment(fs=SR).to(device)

# 6. Warm-up to download/cache ONNX weights
print("Warming up DNSMOS and NISQA models…")
dummy = torch.zeros(1, BUF_SIZE, device=device)
with torch.no_grad():
    _ = dnsmos(dummy)
    _ = nisqa(dummy)
print("Warm-up complete.")

last_print = 0.0

def audio_callback(indata, frames, time_info, status):
    global last_print
    data = indata[:, 0].astype(np.float32)
    ring_buffer.extend(data.tolist())

    # --- Audio-signal quality (always)
    rms      = np.sqrt((data**2).mean())
    peak     = np.max(np.abs(data))
    clip_cnt = int((np.abs(data) >= 0.99).sum())
    S_spec   = np.abs(librosa.stft(data, n_fft=512, hop_length=256))
    flat     = float(librosa.feature.spectral_flatness(S=S_spec).mean())
    zcr      = float(librosa.feature.zero_crossing_rate(data).mean())

    rms_cat  = categorize(rms, 0.05, 0.02)
    peak_cat = categorize(peak, 0.5,  0.2)
    clip_cat = categorize(clip_cnt, 0,  5, invert=True)
    flat_cat = categorize(flat,    0.2, 0.4, invert=True)
    zcr_cat  = categorize(zcr,     0.02,0.05, invert=True)

    # Default speech-quality state
    dnsmos_ovr = nisqa_mos = avg_mos = None
    speech_cat = "Buffering…"

    # --- Speech-quality metrics (only if buffer ≥3s)
    if len(ring_buffer) == BUF_SIZE:
        buf    = np.array(ring_buffer, dtype=np.float32)
        tensor = torch.from_numpy(buf).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                d_scores = dnsmos(tensor)  # → Tensor shape (4,)
                n_scores = nisqa(tensor)   # → Tensor shape (5,)

            # Squeeze to 1D and index correctly
            d_scores = d_scores.squeeze()  # shape (4,)
            n_scores = n_scores.squeeze()  # shape (5,)

            dnsmos_ovr = d_scores[-1].item()  # overall MOS
            nisqa_mos  = n_scores[0].item()   # overall MOS
            avg_mos    = (dnsmos_ovr + nisqa_mos) / 2.0
            speech_cat = categorize(avg_mos, GOOD_THRESH, MOD_THRESH)
        except Exception as e:
            print(f"[Speech QC error] {e}")
            speech_cat = "Error"

    # --- Throttled console output
    now = time.time()
    if now - last_print >= PRINT_INTERVAL:
        last_print = now

        # Speech-quality block
        print("\n🔈 Speech Quality:")
        if avg_mos is not None:
            print(f"  • DNSMOS Overall MOS:  {dnsmos_ovr:.2f}")
            print(f"  • NISQA Overall MOS:   {nisqa_mos:.2f}")
            print(f"  • Avg Speech MOS:      {avg_mos:.2f} ({speech_cat})")
        else:
            print(f"  • {speech_cat}")

        # Audio-signal quality block
        print("\n🎚️ Audio Signal Quality:")
        print(f"  • RMS Level:           {rms:.4f} ({rms_cat})")
        print(f"  • Peak Level:          {peak:.4f} ({peak_cat})")
        print(f"  • Clipping Count:      {clip_cnt}   ({clip_cat})")
        print(f"  • Spectral Flatness:   {flat:.4f} ({flat_cat})")
        print(f"  • Zero-Cross Rate:     {zcr:.4f} ({zcr_cat})")

def main():
    stream = sd.InputStream(
        device=device_idx, channels=1,
        samplerate=SR, blocksize=BLOCK,
        callback=audio_callback
    )
    stream.start()
    print(f"\nMonitoring live speech & audio quality… updates every {PRINT_INTERVAL}s. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    main()











# import sounddevice as sd
# for idx, dev in enumerate(sd.query_devices()):
#     print(idx, dev['name'], dev['max_input_channels'])
