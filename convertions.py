import librosa
import librosa.display
import numpy as np
from PIL import Image


def mp3_to_signal(audio_file_path:str, display:bool=False) -> [np.ndarray, int]:
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



def signal_to_spectro(audio_signal:np.ndarray, display:bool=False) -> np.ndarray:
    """
    Convert audio signal to spectrogram
    """
    
    if display:
        print("signal_to_spectro()")
        print("  - Checking input parameters...")

    assert type(audio_signal) == np.ndarray, "Audio signal must be numpy array"
    
    # Convert audio to spectrogram
    if display: print("  - Converting audio to spectrogram...")
    D = np.abs(librosa.stft(audio_signal))
    
    if display: print("## end signal_to_spectro()")
    return D


def spectro_to_image(spectro:np.ndarray, spectro_file_path:str, display:bool=False, image_size:int=1024) -> None:
    """
    Convert spectrogram to image and save it to spectro_dir_path
    """
    
    if display: print("spectro_to_image()")
    
    # Convert spectrogram to decibels
    if display: print("  - Converting spectrogram to decibels...")
    spectro_db = librosa.amplitude_to_db(spectro)

    # rescacle to 0-255 and convert to uint8
    if display: print("  - Rescaling spectrogram (0 - 255)...")
    spectro_db = 255 * (spectro_db - spectro_db.min()) / (spectro_db.max() - spectro_db.min())
    spectro_db = spectro_db.astype(np.uint8)
    
    # crop image to square
    if display: print("  - Cropping spectrogram...")
    image = Image.fromarray(spectro_db)
    image = image.crop((0, 0, image_size, image_size))
    
    if display: print("  - Saving spectrogram...")
    image.save(spectro_file_path, format="png")
    
    if display: print("## end spectro_to_image()")

    

def mp3_to_image(audio_file_path:str, spectro_dir_path:str=None, display:bool=False) -> None:
    """
    Convert audio file to spectrogram and save it to spectro_file_path
    """
    
    if display: print("mp3_to_image()")
    
    # Load audio file
    y, sr = mp3_to_signal(audio_file_path, display=display)
    if display:
        print("  - Audio signal shape:", y.shape)
        print("  - Sampling rate:", sr)
    
    # Convert audio to spectrogram
    D = signal_to_spectro(y, display=display)
    if display: print("  - Spectrogram shape:", D.shape)
    
    # Convert spectrogram to image
    spectro_file_path = spectro_dir_path + audio_file_path.split("/")[-1].split(".")[0] + ".png"
    spectro_to_image(D, spectro_file_path, display=display)
    
    if display: print("## end mp3_to_image()")    



if __name__ == "__main__":
    
    music_name = "Punk Jazz Revisited"
    
    mp3_to_image("data/audio/" + music_name + ".mp3", "data/spectro/", display=True)
    