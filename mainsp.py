# #!/usr/bin/env python3
# """
# mainsp.py

# Live audio capture → VAD segmentation → Whisper large-v3-turbo transcription (English only).
# """

# import logging
# import queue
# import sys
# import time

# import numpy as np
# import sounddevice as sd
# import webrtcvad
# import librosa

# import torch
# from transformers import (
#     AutoProcessor,
#     AutoModelForSpeechSeq2Seq,
#     pipeline,
#     logging as hf_logging,
# )

# # ──────────────────────────────────────────────────────────────────────────────
# # 0. Suppress unwanted warnings
# # ──────────────────────────────────────────────────────────────────────────────
# # Python logging: silence deprecation/future warnings from Transformers
# hf_logging.set_verbosity_error()
# logging.getLogger("transformers.generation_whisper").setLevel(logging.ERROR)

# # ──────────────────────────────────────────────────────────────────────────────
# # 1. Configuration
# # ──────────────────────────────────────────────────────────────────────────────
# SAMPLE_RATE     = 16000     # both VAD and Whisper expect 16 kHz
# FRAME_DURATION  = 30        # each VAD frame is 30 ms
# FRAME_SIZE      = int(SAMPLE_RATE * FRAME_DURATION / 1000)
# VAD_MODE        = 2         # aggressiveness (0–3)
# SILENCE_TIMEOUT = 0.5       # seconds of silence before flushing an utterance

# MODEL_ID        = "openai/whisper-large-v3-turbo"
# DEVICE          = "cpu"     # force CPU inference

# # ──────────────────────────────────────────────────────────────────────────────
# # 2. Initialize VAD + buffers
# # ──────────────────────────────────────────────────────────────────────────────
# vad = webrtcvad.Vad(VAD_MODE)
# audio_queue = queue.Queue()
# utterance_frames = []
# last_speech_time = None

# # ──────────────────────────────────────────────────────────────────────────────
# # 3. Load Whisper pipeline with fixes
# # ──────────────────────────────────────────────────────────────────────────────
# processor = AutoProcessor.from_pretrained(MODEL_ID)
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     MODEL_ID,
#     attn_implementation="eager"      # use eager attention to silence warnings
# ).to(DEVICE)

# asr = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     device=-1,                         # CPU
#     chunk_length_s=30,
#     return_timestamps="word",
#     generate_kwargs={                 # force English transcription
#         "language": "en",
#         "task": "transcribe"
#     }
# )

# # ──────────────────────────────────────────────────────────────────────────────
# # 4. Audio callback — enqueue mono float32 frames
# # ──────────────────────────────────────────────────────────────────────────────
# def audio_callback(indata, frames, time_info, status):
#     if status:
#         print(f"[Stream status] {status}", file=sys.stderr)
#     audio_queue.put(indata[:, 0].copy())

# # ──────────────────────────────────────────────────────────────────────────────
# # 5. Helper — convert float32 array to 16-bit PCM bytes
# # ──────────────────────────────────────────────────────────────────────────────
# def to_pcm16(frame: np.ndarray) -> bytes:
#     pcm = (frame * 32767).astype(np.int16)
#     return pcm.tobytes()

# # ──────────────────────────────────────────────────────────────────────────────
# # 6. Main processing loop
# # ──────────────────────────────────────────────────────────────────────────────
# def main():
#     global last_speech_time

#     # Select default input device
#     dev_idx = sd.default.device[0]
#     dev_name = sd.query_devices(dev_idx)["name"]
#     print(f"Using audio device #{dev_idx}: {dev_name}")
#     print("Listening (Ctrl+C to stop)…\n")

#     # Start audio stream
#     stream = sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         blocksize=FRAME_SIZE,
#         channels=1,
#         dtype="float32",
#         callback=audio_callback
#     )
#     stream.start()

#     try:
#         while True:
#             frame = audio_queue.get()
#             if frame is None:
#                 break

#             is_speech = vad.is_speech(to_pcm16(frame), SAMPLE_RATE)
#             now = time.time()

#             if is_speech:
#                 # append speech frame
#                 utterance_frames.append(frame)
#                 last_speech_time = now
#             else:
#                 # if silence surpassed timeout, flush utterance
#                 if last_speech_time and (now - last_speech_time) > SILENCE_TIMEOUT:
#                     audio_data = np.concatenate(utterance_frames)
#                     utterance_frames.clear()
#                     last_speech_time = None

#                     # Resample if necessary
#                     target_sr = processor.feature_extractor.sampling_rate
#                     if SAMPLE_RATE != target_sr:
#                         audio_data = librosa.resample(
#                             audio_data,
#                             orig_sr=SAMPLE_RATE,
#                             target_sr=target_sr
#                         )

#                     # Transcribe with Whisper
#                     result = asr(audio_data, batch_size=1)
#                     text = result.get("text", "").strip()
#                     print(f"\n📝 Transcript: {text}\n")

#             # slight sleep to avoid busy-spin
#             time.sleep(0.001)

#     except KeyboardInterrupt:
#         print("\nInterrupted, shutting down.")
#     finally:
#         stream.stop()
#         stream.close()

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# real_time_transcribe.py

# Live audio capture → WebRTC VAD segmentation → quantized Whisper-Tiny transcription via faster-whisper on CPU.
# """

# import queue
# import sys
# import time

# import numpy as np
# import sounddevice as sd
# import webrtcvad
# from faster_whisper import WhisperModel

# # ──────────────────────────────────────────────────────────────────────────────
# # Configuration
# # ──────────────────────────────────────────────────────────────────────────────
# SAMPLE_RATE     = 16000   # 16 kHz for both VAD and model
# FRAME_DURATION  = 30      # ms per VAD frame
# FRAME_SIZE      = int(SAMPLE_RATE * FRAME_DURATION / 1000)
# VAD_MODE        = 3       # aggressiveness 0–3 (higher = more aggressive filtering)
# SILENCE_TIMEOUT = 0.5     # seconds of silence to flush utterance

# # ──────────────────────────────────────────────────────────────────────────────
# # Initialize VAD & Audio Queue
# # ──────────────────────────────────────────────────────────────────────────────
# vad = webrtcvad.Vad(VAD_MODE)
# audio_queue = queue.Queue()

# # ──────────────────────────────────────────────────────────────────────────────
# # Load Quantized Whisper-Tiny Model
# # ──────────────────────────────────────────────────────────────────────────────
# # explicit keyword style
# # Load the CTranslate2-converted Tiny English model
# model = WhisperModel(
#     "tiny.en",         # Systran/faster-whisper-tiny.en under the hood
#     device="cpu",
#     compute_type="int8"
# )

# segments, info = model.transcribe("path/to/audio.wav")
# for seg in segments:
#     print(f"[{seg.start:.2f}s → {seg.end:.2f}s] {seg.text}")

# # ──────────────────────────────────────────────────────────────────────────────
# # Audio callback: enqueue mono float32 frames
# # ──────────────────────────────────────────────────────────────────────────────
# def audio_callback(indata, frames, time_info, status):
#     if status:
#         print(f"[Stream status] {status}", file=sys.stderr)
#     audio_queue.put(indata[:, 0].copy())

# # ──────────────────────────────────────────────────────────────────────────────
# # Helper: convert float32 samples to 16-bit PCM bytes for VAD
# # ──────────────────────────────────────────────────────────────────────────────
# def to_pcm16(frame: np.ndarray) -> bytes:
#     pcm = (frame * 32767).astype(np.int16)
#     return pcm.tobytes()

# # ──────────────────────────────────────────────────────────────────────────────
# # Main loop: segment via VAD, transcribe each utterance
# # ──────────────────────────────────────────────────────────────────────────────
# def main():
#     # Select default input device
#     dev_idx = sd.default.device[0]
#     dev_name = sd.query_devices(dev_idx)["name"]
#     print(f"Using audio device #{dev_idx}: {dev_name}")
#     print("Listening (Ctrl+C to stop)...\n")

#     stream = sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         blocksize=FRAME_SIZE,
#         channels=1,
#         dtype="float32",
#         callback=audio_callback
#     )
#     stream.start()

#     utterance = []
#     last_speech_time = None

#     try:
#         while True:
#             frame = audio_queue.get()

#             # VAD decision on this 30 ms frame
#             is_speech = vad.is_speech(to_pcm16(frame), SAMPLE_RATE)

#             now = time.time()
#             if is_speech:
#                 utterance.append(frame)
#                 last_speech_time = now
#             else:
#                 # If we've had speech and now >timeout silence, process utterance
#                 if last_speech_time and (now - last_speech_time) > SILENCE_TIMEOUT:
#                     audio_data = np.concatenate(utterance)
#                     utterance.clear()
#                     last_speech_time = None

#                     # Transcribe via faster-whisper
#                     segments, info = model.transcribe(
#                         audio_data,
#                         beam_size=5,
#                         language="en",
#                         word_timestamps=False
#                     )
#                     # Print each segment
#                     for seg in segments:
#                         start = seg.start
#                         end   = seg.end
#                         text  = seg.text.strip()
#                         print(f"{start:.2f}s → {end:.2f}s: {text}")

#             # Avoid busy-spin
#             time.sleep(0.001)

#     except KeyboardInterrupt:
#         print("\nInterrupted by user; shutting down.")
#     finally:
#         stream.stop()
#         stream.close()

# if __name__ == "__main__":
#     main()

##########################################################################################################
















# this is the first working model but there is a time and accuracy issue 


#!/usr/bin/env python3
# """
# mainsp.py

# Live audio capture → WebRTC VAD segmentation → quantized Whisper-Tiny transcription via faster-whisper (tiny.en) on CPU.
# """

# import queue
# import sys
# import time

# import numpy as np
# import sounddevice as sd
# import webrtcvad
# from faster_whisper import WhisperModel

# # ──────────────────────────────────────────────────────────────────────────────
# # Configuration
# # ──────────────────────────────────────────────────────────────────────────────
# SAMPLE_RATE     = 16000   # 16 kHz for both VAD and model
# FRAME_DURATION  = 30      # ms per VAD frame
# FRAME_SIZE      = int(SAMPLE_RATE * FRAME_DURATION / 1000)
# VAD_MODE        = 3       # aggressiveness 0–3
# SILENCE_TIMEOUT = 0.5     # seconds of silence to flush utterance

# # ──────────────────────────────────────────────────────────────────────────────
# # Initialize VAD & Audio Queue
# # ──────────────────────────────────────────────────────────────────────────────
# vad = webrtcvad.Vad(VAD_MODE)
# audio_queue = queue.Queue()

# # ──────────────────────────────────────────────────────────────────────────────
# # Load the CTranslate2-converted Tiny English Whisper model
# # ──────────────────────────────────────────────────────────────────────────────
# model = WhisperModel(
#     "tiny.en",    # uses Systran/faster-whisper-tiny.en under the hood
#     device="cpu",
#     compute_type="int8"
# )

# # ──────────────────────────────────────────────────────────────────────────────
# # Audio callback: enqueue mono float32 frames
# # ──────────────────────────────────────────────────────────────────────────────
# def audio_callback(indata, frames, time_info, status):
#     if status:
#         print(f"[Stream status] {status}", file=sys.stderr)
#     # indata is shape (frames, channels); take channel 0
#     audio_queue.put(indata[:, 0].copy())

# # ──────────────────────────────────────────────────────────────────────────────
# # Helper: convert float32 array to 16-bit PCM bytes for VAD
# # ──────────────────────────────────────────────────────────────────────────────
# def to_pcm16(frame: np.ndarray) -> bytes:
#     pcm = (frame * 32767).astype(np.int16)
#     return pcm.tobytes()

# # ──────────────────────────────────────────────────────────────────────────────
# # Main loop: segment via VAD, transcribe each utterance
# # ──────────────────────────────────────────────────────────────────────────────
# def main():
#     # Pick default input device
#     dev_idx = sd.default.device[0]
#     dev_name = sd.query_devices(dev_idx)["name"]
#     print(f"Using audio device #{dev_idx}: {dev_name}")
#     print("Listening (Ctrl+C to stop)…\n")

#     stream = sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         blocksize=FRAME_SIZE,
#         channels=1,
#         dtype="float32",
#         callback=audio_callback
#     )
#     stream.start()

#     utterance_frames = []
#     last_speech_time = None

#     try:
#         while True:
#             frame = audio_queue.get()

#             # VAD decision on this frame
#             is_speech = vad.is_speech(to_pcm16(frame), SAMPLE_RATE)
#             now = time.time()

#             if is_speech:
#                 # accumulate speech frames
#                 utterance_frames.append(frame)
#                 last_speech_time = now
#             else:
#                 # if we had speech but now silence > timeout, flush utterance
#                 if last_speech_time and (now - last_speech_time) > SILENCE_TIMEOUT:
#                     audio_data = np.concatenate(utterance_frames)
#                     utterance_frames.clear()
#                     last_speech_time = None

#                     # Transcribe directly from NumPy array
#                     segments, _ = model.transcribe(
#                         audio_data,
#                         beam_size=5,
#                         language="en"
#                     )

#                     # Print each segment with timestamps
#                     for seg in segments:
#                         start, end, text = seg.start, seg.end, seg.text.strip()
#                         print(f"[{start:.2f}s → {end:.2f}s] {text}")

#             # tiny sleep to avoid busy‐spin
#             time.sleep(0.001)

#     except KeyboardInterrupt:
#         print("\nInterrupted, shutting down.")
#     finally:
#         stream.stop()
#         stream.close()

# if __name__ == "__main__":
#     main()


















# #######################################################################

# this working little fine 
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# [0.00s→5.20s] And she had some sort of top clearance and
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# [0.00s→1.80s] like why did you why did you look this up?
# [1.80s→3.00s] Like what is this all about?
# [3.00s→5.64s] And I think she wound up either getting fired
# [5.64s→7.60s] or transferred to some of the position
# [7.60s→10.32s] or lost to whatever clearance that she had.
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected
# ❌ Speech not detected

# #!/usr/bin/env python3
# """
# mainsp.py

# Only transcribe when YOU speak; otherwise report "Speech not detected".
# """

# import queue
# import sys
# import time

# import numpy as np
# import sounddevice as sd
# import webrtcvad
# import torch

# from speechbrain.inference import EncoderClassifier
# from faster_whisper import WhisperModel

# # ──────────────────────────────────────────────────────────────────────────────
# # Config
# # ──────────────────────────────────────────────────────────────────────────────
# SR               = 16000    # sampling rate
# FRAME_MS         = 30       # ms per VAD frame
# FRAME_SIZE       = int(SR * FRAME_MS / 1000)
# VAD_MODE         = 3        # 0–3 aggressiveness
# SIM_THRESHOLD    = 0.7      # speaker-verification threshold
# SILENCE_TIMEOUT  = 0.2      # seconds to end an utterance
# NOT_SPEECH_PRINT = 1.0      # throttle for "Speech not detected"

# # ──────────────────────────────────────────────────────────────────────────────
# # Init VAD and queues
# # ──────────────────────────────────────────────────────────────────────────────
# vad = webrtcvad.Vad(VAD_MODE)
# audio_q = queue.Queue()

# # ──────────────────────────────────────────────────────────────────────────────
# # Speaker-verification (ECAPA-TDNN)
# # ──────────────────────────────────────────────────────────────────────────────
# spk_model = EncoderClassifier.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb",
#     run_opts={"device":"cpu"}
# )

# print(">>> Press Enter and speak 8s to enroll your voice.")
# input()
# enroll_buf = []

# def enroll_cb(indata, frames, ti, status):
#     enroll_buf.append(indata[:,0].copy())

# with sd.InputStream(samplerate=SR, blocksize=FRAME_SIZE,
#                     channels=1, dtype="float32",
#                     callback=enroll_cb):
#     sd.sleep(8000)

# enroll_audio = np.concatenate(enroll_buf)
# enroll_emb = spk_model.encode_batch(
#     torch.from_numpy(enroll_audio).unsqueeze(0)
# ).squeeze().detach().cpu().flatten()
# print("✅ Enrollment done.\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # ASR model (Whisper-Tiny INT8)
# # ──────────────────────────────────────────────────────────────────────────────
# asr = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# # ──────────────────────────────────────────────────────────────────────────────
# # Audio callback: enqueue frames
# # ──────────────────────────────────────────────────────────────────────────────
# def audio_cb(indata, frames, ti, status):
#     if status:
#         print(f"[Stream status] {status}", file=sys.stderr)
#     audio_q.put(indata[:,0].copy())

# stream = sd.InputStream(samplerate=SR, blocksize=FRAME_SIZE,
#                        channels=1, dtype="float32",
#                        callback=audio_cb)
# stream.start()
# print("Listening… (Ctrl+C to exit)\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # Main loop
# # ──────────────────────────────────────────────────────────────────────────────
# utter_buf = []
# last_sp_time = None
# last_not_speech = 0.0

# def to_pcm16(x):
#     return (x * 32767).astype(np.int16).tobytes()

# try:
#     while True:
#         frame = audio_q.get()
#         now = time.time()
#         is_speech = vad.is_speech(to_pcm16(frame), SR)

#         if is_speech:
#             # accumulate
#             utter_buf.append(frame)
#             last_sp_time = now

#         else:
#             # if we were speaking and now silent long enough → flush
#             if last_sp_time and (now - last_sp_time) > SILENCE_TIMEOUT:
#                 audio_data = np.concatenate(utter_buf)
#                 utter_buf.clear()
#                 last_sp_time = None

#                 # speaker verification
#                 seg_emb = spk_model.encode_batch(
#                     torch.from_numpy(audio_data).unsqueeze(0)
#                 ).squeeze().detach().cpu().flatten()
#                 sim = (   torch.dot(enroll_emb, seg_emb)
#                         / (enroll_emb.norm()*seg_emb.norm())
#                       ).item()

#                 if sim >= SIM_THRESHOLD:
#                     # your voice → transcribe
#                     segments, _ = asr.transcribe(audio_data,
#                                                  language="en",
#                                                  beam_size=5)
#                     for s in segments:
#                         print(f"[{s.start:.2f}s→{s.end:.2f}s] {s.text.strip()}")
#                 else:
#                     # other voice/music → skip
#                     pass

#             # if truly no speech, print notification (throttled)
#             if not last_sp_time and now - last_not_speech > NOT_SPEECH_PRINT:
#                 print("❌ Speech not detected")
#                 last_not_speech = now

#         time.sleep(0.001)

# except KeyboardInterrupt:
#     print("\nExiting…")
# finally:
#     stream.stop()
#     stream.close()




##################################################################################################################################
#####
# This is the first working code for speaking for 60s and then getting the transcribe 

# Please speak for 60 seconds, then remain silent.
# ⏹ Stop speaking.

# 📝 Transcript:
# he was doing was inappropriate for her job like maybe what she was doing was 
# demonstrating that she couldn't be trusted because she's doing something that's 
# not there was no request to look up little green man yeah so like she she did it on her own
# like it's a hundred percent factual hard to tell you know because it was a girl I dated I don't 
# I don't really know oh you dated yeah I dated her yeah this is yeah like like I said early 90s 
# I was living in New York I'd actually dated her in Boston then we met up again in New York a couple years 
# later so she was telling me about her job and then she was telling me when I checked this out I shouldn't be 
# telling you this but part back then I was all in on UFOs I was like wow you have always real but as an older man 
# I look back and go well if you have some kid you know she was basically my age she's probably 24 as well and you're having this kid



#!/usr/bin/env python3
# """
# record_and_transcribe.py

# – Prompt for 60 s of speech
# – Record 60 s
# – Check for speech; if none, report
# – Otherwise, signal stop and transcribe whole clip
# """

# import numpy as np
# import sounddevice as sd
# import webrtcvad
# from faster_whisper import WhisperModel

# # ─── Config ──────────────────────────────────────────────────────────────────
# DURATION       = 60       # seconds to record
# SAMPLE_RATE    = 16000    # 16 kHz
# VAD_MODE       = 3        # 0–3 aggressiveness
# FRAME_DURATION = 30       # ms per VAD frame
# FRAME_SIZE     = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# # ─── 1. Record 60 s ────────────────────────────────────────────────────────────
# print(f"Please speak for {DURATION} seconds, then remain silent.")
# audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
#                channels=1, dtype="float32")  #  [oai_citation:0‡python-sounddevice.readthedocs.io](https://python-sounddevice.readthedocs.io/en/0.4.1/usage.html?utm_source=chatgpt.com)
# sd.wait()  # block until recording is done

# # ─── 2. Check for any speech via WebRTC VAD ───────────────────────────────────
# vad = webrtcvad.Vad(VAD_MODE)                      #  [oai_citation:1‡GitHub](https://github.com/wiseman/py-webrtcvad?utm_source=chatgpt.com)
# pcm_data = (audio.flatten() * 32767).astype("int16").tobytes()
# has_speech = False
# for i in range(0, len(pcm_data), FRAME_SIZE * 2):  # 2 bytes/sample
#     frame = pcm_data[i : i + FRAME_SIZE * 2]
#     if len(frame) < FRAME_SIZE * 2:
#         break
#     if vad.is_speech(frame, SAMPLE_RATE):
#         has_speech = True
#         break

# # ─── 3. Respond or transcribe ─────────────────────────────────────────────────
# if not has_speech:
#     print("❌ Speech not detected")
# else:
#     print("⏹ Stop speaking.")  # user has finished the minute
#     # Load quantized Whisper-Tiny (INT8) via Faster-Whisper ─────────────────
#     model = WhisperModel("tiny.en", device="cpu", compute_type="int8")  #  [oai_citation:2‡GitHub](https://github.com/SYSTRAN/faster-whisper?utm_source=chatgpt.com)
#     audio_np = audio.flatten()                                        # shape (N,)
#     segments, _ = model.transcribe(
#         audio_np, language="en", beam_size=5
#     )
#     # Concatenate and print
#     transcript = " ".join(seg.text.strip() for seg in segments)
#     print("\n📝 Transcript:")
#     print(transcript)



