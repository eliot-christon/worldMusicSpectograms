import librosa
import librosa.display
import numpy as np
import os
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


def spectro_to_image(spectro_db:np.ndarray, spectro_file_path:str, display:bool=False, image_size:int=1025) -> None:
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
    image = image.crop((0, 0, image_size, image_size))
    
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
    audio_signal = librosa.istft(spectro, hop_length=hop_length, n_fft=n_fft)
    
    # rescale to -1 - 1
    if display: print("  - Rescaling audio signal (-1, 1)...")
    audio_signal = audio_signal.astype(np.float32)
    audio_signal = audio_signal / audio_signal.max()
    
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
    
    audio_signal = audio_signal.astype(np.int16)
    
    print(audio_signal.shape)
    print(audio_signal.dtype)
    

    sf.write(audio_file_path, audio_signal, sr)
    
    if display: print("## end signal_to_mp3()")









####################################################################################################################################
####################################################################################################################################










def mp3_to_image(audio_file_path:str, spectro_dir_path:str=None, batch_duration:int=120, image_size:int=1025, display:bool=False) -> dict:
    """
    Convert audio file to spectrogram and save it to spectro_file_path
    
    returns params
    """
    
    params = {"sampling_rate": None, "n_fft": None, "hop_length": None, "min_spectro_db": [], "max_spectro_db": []}
    
    if display: 
        print("mp3_to_image()")
        print("  - Loading audio file...")
    
    # Load audio file
    audio_signal, sr = mp3_to_signal(audio_file_path, display=display)        
    
    # Split audio signal into batches of duration
    if display: print("  - Splitting audio signal into batches...")
    batch_size = int(sr * batch_duration)
    audio_signal_batches = signal_batch_maker(audio_signal, batch_size, display=display)
    
    # create folder for spectrograms
    if display: print("  - Creating folder for spectrograms...")
    spectro_dir_path += audio_file_path.split("/")[-1].split(".")[0] + "/"
    if not os.path.exists(spectro_dir_path):
        os.makedirs(spectro_dir_path)
    
    # get the n_fft and hop_length parameters
    # we want the spectrogram to be square, dim = (image_size, image_size)
    # We can know in advance the size of the spectrogram with n_fft and hop_length
    # the spectrogram size is ( n_fft/2 + 1, audio_signal.shape[0] / hop_length + 1 )
    n_fft = int(image_size * 2 - 2)
    hop_length = int((batch_size / image_size) + 1)
    
    if display:
        print(" - Parameters:")
        print("      Audio signal shape:", audio_signal.shape)
        print("      Sampling rate     :", sr)
        print("      Batch size        :", batch_size)
        print("      n_fft             :", n_fft)
        print("      hop_length        :", hop_length)
    
    for i, audio_signal_batch in enumerate(audio_signal_batches):
        if display: print("  - Batch", i, "shape:", audio_signal_batch.shape)
        
        # Convert audio to spectrogram
        
        D = signal_to_spectro(audio_signal_batch, n_fft, hop_length, display=display)
        
        assert D.shape[0] == image_size and D.shape[1] == image_size, "Spectrogram shape must be (image_size, image_size) = ({}, {}) but is {}".format(image_size, image_size, D.shape)
        if display: print("  - Spectrogram shape:", D.shape)
        
        params["min_spectro_db"].append(D.min())
        params["max_spectro_db"].append(D.max())
        
        # Convert spectrogram to image
        spectro_file_path = spectro_dir_path + str(i) + ".png"
        
        spectro_to_image(D, spectro_file_path, display=display)
    
    params["sampling_rate"] = sr
    params["n_fft"] = n_fft
    params["hop_length"] = hop_length
    
    if display: print("## end mp3_to_image()")    
    return params



def image_to_mp3(spectro_dir_path:str, params:dict, audio_file_path:str=None, batch_duration:int=120, image_size:int=1025, display:bool=False) -> None:
    """
    Convert spectrogram to audio file and save it to audio_file_path
    """
    
    if display: print("image_to_mp3()")
    
    # get parameters
    sr = params["sampling_rate"]
    n_fft = params["n_fft"]
    hop_length = params["hop_length"]
    
    # get each audio signal from spectrograms
    audio_signal_batches = []
    
    batch_size = int(sr * batch_duration)
    
    if display:
        print(" - Parameters:")
        print("      Sampling rate     :", sr)
        print("      Batch size        :", batch_size)
        print("      n_fft             :", n_fft)
        print("      hop_length        :", hop_length)
    
    
    for i, spectro_file_name in enumerate(os.listdir(spectro_dir_path)):
        if display: print("  - Spectrogram", i, "file name:", spectro_file_name)
        
        # Load spectrogram
        spectro_file_path = spectro_dir_path + spectro_file_name
        
        min_spectro_db = params["min_spectro_db"][i]
        max_spectro_db = params["max_spectro_db"][i]
        
        D = image_to_spectro(spectro_file_path, min_spectro_db=min_spectro_db, max_spectro_db=max_spectro_db, display=display)
        
        assert D.shape[0] == image_size and D.shape[1] == image_size, "Spectrogram shape must be (image_size, image_size) = ({}, {}) but is {}".format(image_size, image_size, D.shape)
        
        if display: print("    - Spectrogram shape:", D.shape)
        
        # Convert spectrogram to audio
        audio_signal_batch = spectro_to_signal(D, n_fft, hop_length, display=display)
        
        # resize audio signal batch
        audio_signal_batch = np.resize(audio_signal_batch, batch_size)
        
        assert audio_signal_batch.shape[0] == batch_size, "Audio signal batch shape must be ({},) but is {}".format(batch_size, audio_signal_batch.shape)
        
        if display: print("    - Audio signal shape:", audio_signal_batch.shape)
        
        audio_signal_batches.append(audio_signal_batch)
    
    # Join audio signal batches
    audio_signal_batches = np.array(audio_signal_batches)
    audio_signal = signal_batch_joiner(audio_signal_batches, display=display)
    if display: print("  - Audio signal shape:", audio_signal.shape)
    
    # Save audio signal to mp3 file
    signal_to_mp3(audio_signal, sr, audio_file_path, display=display)
    



if __name__ == "__main__":
    
    music_name = "Punk Jazz Revisited"
    
    sr = mp3_to_image("data/audio/" + music_name + ".mp3", "data/spectro/", display=True)
    
    
    # now convert spectrograms back to audio
    
    print("\n\n\n")
    
    image_to_mp3("data/spectro/" + music_name + "/", sr, "data/audio_output/" + music_name + "_recovered.mp3", display=True)
    