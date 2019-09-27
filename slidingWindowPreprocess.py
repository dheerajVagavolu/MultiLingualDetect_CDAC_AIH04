import librosa
import numpy as np


def dataprepocess(imgSrc):
        
    dur = 4
    sampling = 16000
    samples = sampling*dur
    audio,sampling = librosa.load(imgSrc,sr=sampling)
    noFrames = 0
    #print(len(audio)%samples)
    if len(audio)<samples:
        while len(audio)<samples:
            audio = np.concatenate((audio,audio), axis=0)
        audio = audio[:samples]
        noFrames = 1
        
    else:
        noSec= np.floor(len(audio)/sampling)
        noFrames=noSec-dur+1   

    frames = []
    for i in range(int(noFrames)):    
        melspec=librosa.feature.melspectrogram(
                audio[i*sampling:(i+dur-1)*sampling], sr = samples, n_mels = 129, 
                fmax = 5000, n_fft = 1600, hop_length = 128)
        melspec=librosa.power_to_db(melspec,ref=np.max)
        melspec-=np.min(melspec)
        melspec=melspec/np.max(melspec)
        frames.append(melspec)
        
    
    framesArray = np.array(frames)
    framesArray = framesArray.reshape(int(noFrames),1,129,501)
    return framesArray
