# in order to make this program to work i need Pyaudio Library 

import speech_recognition as sr
import time

def callback(r,audio):
    print(r.recognize_google(audio))
    return r.recognize_google(audio)

r = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    print("say something")
    r.adjust_for_ambient_noise(source)
    

    
stopper = r.listen_in_background(mic,callback)

time.sleep(1)
print(stopper())




    
