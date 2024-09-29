import numpy as np
import librosa


def process_audio(audio, sr=16000, silence_thresh=-60, min_silence_len=250):
    """
    Splits an audio signal into segments using a fixed frame size and hop size.

    Parameters:
    - audio (np.ndarray): The audio signal to split.
    - sr (int): The sample rate of the input audio (default is 16000).
    - silence_thresh (int): Silence threshold (default =-60dB)
    - min_silence_len (int): Minimum silence duration (default 250ms).

    Returns:
    - list of np.ndarray: A list of audio segments.
    - np.ndarray: The intervals where the audio was split.
    """
    frame_length = int(min_silence_len / 1000 * sr)
    hop_length = frame_length // 2
    intervals = librosa.effects.split(
        audio, top_db=-silence_thresh, frame_length=frame_length, hop_length=hop_length
    )
    audio_segments = [audio[start:end] for start, end in intervals]

    return audio_segments, intervals


def merge_audio(audio_segments, intervals, sr_orig, sr_new):
    """
    Merges audio segments back into a single audio signal, filling gaps with silence.

    Parameters:
    - audio_segments (list of np.ndarray): The non-silent audio segments.
    - intervals (np.ndarray): The intervals used for splitting the original audio.
    - sr_orig (int): The sample rate of the original audio
    - sr_new (int): The sample rate of the model

    Returns:
    - np.ndarray: The merged audio signal with silent gaps restored.
    """
    sr_ratio = sr_new / sr_orig if sr_new > sr_orig else 1.0

    merged_audio = audio_segments[0]

    for i in range(1, len(intervals)):
        silence_duration = int((intervals[i][0] - intervals[i - 1][1]) * sr_ratio)
        silence = np.zeros(silence_duration, dtype=audio_segments[0].dtype)
        merged_audio = np.concatenate((merged_audio, silence, audio_segments[i]))

    return merged_audio
