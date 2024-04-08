import numpy as np
import librosa
import soundfile as sf

y, sr = librosa.load("data/audio/Punk Jazz Revisited.mp3", duration=10)
S = np.abs(librosa.stft(y))
y_inv = librosa.griffinlim(S)
sf.write("data/audio_output/Punk Jazz Revisited.wav", y_inv, sr)