
import tkinter as tk
import threading
import pyaudio
import time 
import wave
from data_upload import dataprepocess
from tkinter import *
import CRNN
import torch
model = CRNN.CRNN()
model.load_state_dict(torch.load("weights/TrialRunWeights.pth"))

lang_list = ["TAMIL", "GUJARATI", "MARATHI", "HINDI", "TELUGU"]

def Nmaxelements(list1, N): 
    final_list = [] 
  
    for i in range(0, N):  
        max1 = 0
          
        for j in range(len(list1)):      
            if list1[j] > max1: 
                max1 = list1[j]; 
                  
        list1.remove(max1); 
        final_list.append(j)
    return final_list 

def main():
    m=tk.Tk()
    w = Label(m, text="Yo", width = 25)

    def record():
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        fs = 44100  # Record at 44100 samples per second
        seconds = 3
        filename = "output.wav"

        #p = pyaudio.PyAudio()  # Create an interface to PortAudio

        print('Recording')

        
        for k in range(5):
            p = pyaudio.PyAudio()
            stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)
            frames = []  # Initialize array to store frames
            filename = str(k)+"output.wav" 
            # Store data in chunks for 3 seconds
            for i in range(0, int(fs / chunk * seconds)):
                data = stream.read(chunk)
                frames.append(data)

            print('Finished recording')

            # Save the recorded data as a WAV file
            filename = "recordings/"+filename
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            imgs = dataprepocess(filename)
            noFrames = imgs.shape[0]
            imgs = torch.from_numpy(imgs)
            prob = model(imgs)
            prob = prob.tolist()
            if noFrames == 1:
                ans = prob
            else:
                ans = np.array(prob[0])
                for i in range(1,noFrames):
                    ans = np.multiply(ans,np.array(prob[i]))
                ans = list(ans)
            probabilites = [float(i)/sum(ans) for i in ans]
            print(probabilites)
            top_lang = Nmaxelements(probabilites, 3)

            top_lang_str = ""

            top_lang_str += lang_list[top_lang[0]]
            # print(imgs.shape)
            
            # Stop and close the stream 
            stream.stop_stream()
            stream.close()
            # Terminate the PortAudio interface
            p.terminate()
            w.config(text=top_lang_str+str(k))
            w.pack()
             
            m.update()
        
            
            

    m.title('Counting Seconds')
    m.geometry("500x500")

    button = tk.Button(m, text='record', width=20, command=record)
    button.pack()
    
    m.lift()
    m.attributes("-topmost", True)
    m.mainloop()
