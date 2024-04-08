import librosa
import librosa.display
import numpy as np
import os
from PIL import Image


def mp3_to_signal(audio_file_path:str, display:bool=False) -> tuple[np.ndarray, int]:
    """
    Load audio file and return audio signal and sampling rate
    """
    
    if display:
        print("mp3_to_signal()")
        print("  - Checking input parameters...")

    assert audio_file_path.endswith(".mp3"), "Audio file must be mp3 format"
    
    # Load audio file
    if display: print("  - Loading audio file...")
    y, sr = librosa.load(audio_file_path, sr=None)
    
    if display: print("## end mp3_to_signal()")
    return y, sr


def signal_batch_maker(audio_signal:np.ndarray, batch_size:int, display:bool=False) -> np.ndarray:
    """
    Split audio signal into batches
    """
    
    if display:
        print("signal_batch_maker()")
        print("  - Checking input parameters...")

    assert type(audio_signal) == np.ndarray, "Audio signal must be numpy array"
    assert type(batch_size) == int, "Batch size must be integer"
    
    # Split audio signal into batches
    if display: print("  - Splitting audio signal into batches...")
    # all batches have the same size except the last one so we ignore it
    audio_signal_batches = np.array([audio_signal[i:i+batch_size] for i in range(0, audio_signal.shape[0], batch_size) if i+batch_size < audio_signal.shape[0]])
    
    if display: print("## end signal_batch_maker()")
    return audio_signal_batches


def signal_to_spectro(audio_signal:np.ndarray, n_fft:int, hop_length:int, display:bool=False) -> np.ndarray:
    """
    Convert audio signal to spectrogram
    """
    
    if display:
        print("signal_to_spectro()")
        print("  - Checking input parameters...")

    assert type(audio_signal) == np.ndarray, "Audio signal must be numpy array"
    
    # Convert audio to spectrogram
    if display: print("  - Converting audio to spectrogram...")
    
    D = np.abs(librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length))
    
    # Convert spectrogram to decibels
    if display: print("  - Converting spectrogram to decibels...")
    D = librosa.amplitude_to_db(D)
    
    if display: print("## end signal_to_spectro()")
    return D


def spectro_to_image(spectro_db:np.ndarray, spectro_file_path:str, display:bool=False) -> None:
    """
    Convert spectrogram to image and save it to spectro_dir_path
    """
    
    if display: print("spectro_to_image()")
    
    # rescacle to 0-255 and convert to uint8
    if display: print("  - Rescaling spectrogram (0 - 255)...")
    spectro_db = 255 * (spectro_db - spectro_db.min()) / (spectro_db.max() - spectro_db.min())
    spectro_db = spectro_db.astype(np.uint8)

    # crop image to square
    if display: print("  - Cropping spectrogram...")
    image = Image.fromarray(spectro_db)
    
    if display: print("  - Saving spectrogram...")
    image.save(spectro_file_path, format="png")
    
    if display: print("## end spectro_to_image()")






####################################################################################################################################
####################################################################################################################################






def image_to_spectro(spectro_file_path:str, min_spectro_db:float, max_spectro_db:float, display:bool=False) -> np.ndarray:
    """
    Load spectrogram image and convert it to spectrogram
    """
    
    if display:
        print("image_to_spectro()")
        print("  - Checking input parameters...")

    assert spectro_file_path.endswith(".png"), "Spectrogram file must be png format"
    
    if display: print("  - Loading spectrogram image...")
    # Load spectrogram image
    image = Image.open(spectro_file_path)
    
    if display:
        print("  - Spectrogram image shape:", image.size)
        print("  - Converting spectrogram image to numpy array...")
    # Convert image to numpy array
    spectro_db = np.array(image)
    spectro_db = spectro_db.astype(np.float32)
    
    # rescale to min_spectro_db - max_spectro_db
    spectro_db = spectro_db / 255
    spectro_db = spectro_db * (max_spectro_db - min_spectro_db) + min_spectro_db
    
    if display:
        print("    - Spectrogram shape:", spectro_db.shape)
        print("## end image_to_spectro()")
    return spectro_db


def spectro_to_signal(spectro_db:np.ndarray, n_fft:int, hop_length:int, display:bool=False) -> np.ndarray:
    """
    Convert spectrogram to audio signal
    """
    
    if display:
        print("spectro_to_signal()")
        print("  - Checking input parameters...")

    assert type(spectro_db) == np.ndarray, "Spectrogram must be numpy array"
    assert spectro_db.ndim == 2, "Spectrogram must be 2D array"
    
    # Convert decibels to spectrogram
    spectro = librosa.db_to_amplitude(spectro_db)
    
    # Convert spectrogram to audio
    if display: print("  - Converting spectrogram to audio...")
    audio_signal = librosa.griffinlim(spectro, hop_length=hop_length, n_fft=n_fft)
    
    # rescale to -1 - 1
    if display: print("  - Rescaling audio signal (-1, 1)...")
    # audio_signal = audio_signal.astype(np.float32)
    # audio_signal = audio_signal / audio_signal.max()
    
    if display: print("## end spectro_to_signal()")
    return audio_signal


def signal_batch_joiner(audio_signal_batches:np.ndarray, display:bool=False) -> np.ndarray:
    """
    Join audio signal batches
    """
    
    if display:
        print("signal_batch_joiner()")
        print("  - Checking input parameters...")

    assert type(audio_signal_batches) == np.ndarray, "Audio signal batches must be numpy array"
    assert audio_signal_batches.ndim == 2, "Audio signal batches must be 2D array"
    
    # Join audio signal batches
    if display: print("  - Joining audio signal batches...")
    audio_signal = np.concatenate(audio_signal_batches)
    
    if display: print("## end signal_batch_joiner()")
    return audio_signal


def signal_to_mp3(audio_signal:np.ndarray, sr:int, audio_file_path:str, display:bool=False) -> None:
    """
    Save audio signal to mp3 file
    """
    import soundfile as sf
    
    if display:
        print("signal_to_mp3()")
        print("  - Checking input parameters...")

    assert type(audio_signal) == np.ndarray, "Audio signal must be numpy array"
    assert type(sr) == int, "Sampling rate must be integer"
    assert audio_file_path.endswith(".mp3"), "Audio file must be mp3 format"
    
    # Save audio signal to mp3 file
    if display: 
        print("  - Saving audio signal to mp3 file...")
        print("    - Audio signal shape:", audio_signal.shape)
        print("    - Sampling rate:", sr)
        print("  - Saving audio signal to mp3 file...")
        
    # format audio signal to 16-bit int
    
    print(audio_signal.shape)
    print(audio_signal.dtype)
    print(audio_signal.max())
    print(audio_signal.min())
    print(audio_signal[-1000])
        
    print(audio_signal.shape)
    print(audio_signal.dtype)
    

    sf.write(audio_file_path, audio_signal, sr)
    
    if display: print("## end signal_to_mp3()")





####################################################################################################################################
####################################################################################################################################



def main_2():
    music_name = "a-night-in-tunisia"
    
    signal, sr = mp3_to_signal("data/audio/" + music_name + ".mp3", display=True)
    S = signal_to_spectro(signal, 2048, 512, display=True)
    signal_recovered = spectro_to_signal(S, 2048, 512, display=True)
    signal_to_mp3(signal_recovered, sr, "data/audio_output/" + music_name + "_recovered.mp3", display=True)


def main_3():
    music_name = "a-night-in-tunisia"
    
    signal, sr = mp3_to_signal("data/audio/" + music_name + ".mp3", display=True)
    S = signal_to_spectro(signal, 2048, 512, display=True)
    spectro_to_image(S, "data/audio_output/" + music_name + ".png", display=True)
    S_recovered = image_to_spectro("data/audio_output/" + music_name + ".png", S.min(), S.max(), display=True)
    signal_recovered = spectro_to_signal(S_recovered, 2048, 512, display=True)
    signal_to_mp3(signal_recovered, sr, "data/audio_output/" + music_name + "_recovered.mp3", display=True)


def main_4():
    music_name = "a-night-in-tunisia"
    
    signal, sr = mp3_to_signal("data/audio/" + music_name + ".mp3", display=True)
    batch_duration = 60 # seconds
    batch_size = sr * batch_duration
    signals = signal_batch_maker(signal, batch_size, display=True)
    
    spectrograms = []
    for i, signal in enumerate(signals):
        S = signal_to_spectro(signal, 2048, 512, display=True)
        spectrograms.append(S)
        spectro_to_image(S, "data/audio_output/" + music_name + "_" + str(i) + ".png", display=True)
    
    signals_recovered = []
    for i in range(len(signals)):
        S = image_to_spectro("data/audio_output/" + music_name + "_" + str(i) + ".png", spectrograms[i].min(), spectrograms[i].max(), display=True)
        signal = spectro_to_signal(S, 2048, 512, display=True)
        signals_recovered.append(signal)
        signal_to_mp3(signal, sr, "data/audio_output/" + music_name + "_" + str(i) + "_recovered.mp3", display=True)
    
    signal_recovered = signal_batch_joiner(np.array(signals_recovered), display=True)
    signal_to_mp3(signal_recovered, sr, "data/audio_output/" + music_name + "_recovered.mp3", display=True)

if __name__ == "__main__":
    main_4()
    