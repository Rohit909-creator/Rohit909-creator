import speech_recognition as sr
import time



for i in range(5):
    

    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("speak")
        audio = r.listen(source)
        #print(audio)
        #time.sleep(0.1)

        text = r.recognize_google(audio)
        print(text)

    print("Start Again")
    print('/n')
