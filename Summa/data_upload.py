'''
this image will be saved in the current working directory which is to 
be fed to our network
'''
import librosa
import numpy as np


def dataprepocess(imgSrc):
        
    expDur = 6
    sampling = 16000
    samples = sampling*expDur
    audio,sampling = librosa.load(imgSrc,sr=sampling)
    noFrames = 0
    #print(len(audio)%samples)
    if len(audio)<samples:
        while len(audio)<samples:
            audio = np.concatenate((audio,audio), axis=0)
        audio = audio[:samples]
        noFrames = 1
        
    elif (len(audio) % samples > samples*0.5):
        noFrames = int(np.ceil(len(audio) / samples))
        audio = np.concatenate((audio,audio), axis=0)
        audio = audio[:noFrames*samples]
        
    elif (len(audio) % samples < samples*0.5):
        noFrames = int(np.floor(len(audio) / samples))
        audio = audio[:noFrames*samples]

    frames = []
    for i in range(int(noFrames)):    
        temp = librosa.feature.melspectrogram(
                audio[i*samples:(i+1)*samples], sr = samples, n_mels = 129, 
                fmax = 5000, n_fft = 1600, hop_length = 192)
        temp = librosa.power_to_db(temp, ref=np.max)
        temp -= np.min(temp)
        temp = temp*255/np.max(temp)
        frames.append(temp)

    framesArray = np.array(frames)
    framesArray = framesArray.reshape(int(noFrames),1,129,501)
    return framesArray
