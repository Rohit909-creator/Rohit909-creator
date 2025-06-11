#gotta make a body for the robot and
#and use droid cam as robots eye

import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import random
import cv2
from threading import Thread
import time
import speech_recognition as sr
import serial
import pyttsx3

r = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)

#ard = serial.Serial('COM10',9600)

c = 0


#cam = cv2.VideoCapture(0)



stopwords = nltk.corpus.stopwords.words('english')



intents = {'intents':[{'tag':'greetings','patterns':['hello','hi'],'responses':['hi','hello','hola','namaskar']},
                     {'tag':'wishing','patterns':['how are you feeling','you good','are you happy','are you sad','are you mad'],'responses':['Im Fine']},
                    {'tag':'introduce','patterns':['who are you','what are you','what is you name'],'responses':['I am Jarvis a virtual artificial intelligence']},
                    {'tag':'Google','patterns':['search'],'responses':['Searching Google...']},
                    {'tag':'wikipedia','patterns':['who is','what is'],'responses':['From wikipedia']},
                    {'tag':'analyze','patterns':['analyze this material'],'responses':['Analyzing']},
                      {'tag':'intelligence','patterns':['what is your intelligence level','what is your knowledge level'],'responses':['I am good at talking','dont know']},
                      {'tag':'live','patterns':['where is your house','where do you live','where is your place'],'responses':['i am sitting right there','this room is my home actually']},
                      {'tag':'school','patterns':['where are you studying','which school are you studying in'],'responses':['Internet is my school']},
                      {'tag':'bestfriend','patterns':['can i be your bestfriend','will you be my bestfriend'],'responses':['yes I will be','yes bestie']},
                      {'tag':'relationship','patterns':['do you have girlfriend or gf'],'responses':['nope','I am not in any relationship yet']},
                      {'tag':'scan','patterns':['scan this'],'responses':['scanning']},
                      {'tag':'sing','patterns':['sing a song'],'responses':['nope','i aint a good singer yet']},
                      {'tag':'story','patterns':['tell me a bedtime story'],'responses':['alright,here you go']},
                      {'tag':'ok','patterns':['ok','alright'],'responses':['mmm','hmm','ok','alright']},
                      {'tag':'thank','patterns':['thanks','thank you'],'responses':['thank ya']}]}

                    

tags = []

all_words = []

for intent in intents['intents']:
    tags.append(intent['tag'])




for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = word_tokenize(pattern)
        all_words.extend(pattern)

print(tags)


stopwords = stopwords
cleaned_all_words = []
for i in all_words:
    if i not in stopwords:
        cleaned_all_words.append(i)
all_words = cleaned_all_words

def bag_of_words(all_words,intents): #for making the training data
    bag = np.zeros((len(tags),len(all_words)))
    k = -1
    for intent in intents['intents']:
        k = k + 1
        for pattern in intent['patterns']:
            for i in range(len(all_words)):
                w = word_tokenize(all_words[i])
                pattern = word_tokenize(pattern)
                for pattern in pattern:
                    if pattern in w:
                        bag[k][i] = 1
    return bag

def clean(all_words): #to pop out the repeating words

    for i in  all_words:
        
	    k = 0
	    for j,L in enumerate(all_words):
		    if i == L:
			    k+=1
			    if k >= 2:
				    all_words.pop(j)
    return all_words





#print(stopwords)

all_words = clean(all_words)

print(all_words)

def analyze(d,all_words):
    d = word_tokenize(d)
    data = np.zeros((1,len(all_words)))
    for i,j in enumerate(all_words):
        if j in d:
            data[0][i] = 1

    return data

d = 'hello hi'

print('analyzed',analyze(d,all_words))
            
    


print(bag_of_words(all_words,intents))
#training data

X = torch.tensor(bag_of_words(all_words,intents),dtype = torch.float32)
print(X.shape)
Y = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

print("y:",Y.shape)

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet,self).__init__()

        self.fc1 = nn.Linear(len(all_words),200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,len(tags))
        self.relu = nn.ReLU()
        
    def forward(self,X):

        out = self.relu(self.fc1(X))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        
        return out


model = NeuralNet()
#model = torch.load('Jarvis.pth')
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)



#training:

for epoch in range(800):

    out = model(X)
    print(out.shape,Y.shape)
    l = loss(out,Y)
    if epoch%100 == 0:
        
        print(l)

    optimizer.zero_grad()
    l.backward()

    optimizer.step()

print(model(X))

val = int(input('>>>'))
if val ==1:
    torch.save(model,'Jarvis2.0.pth')



'''

def ard_data():
    global c

    x = c

    if x < 400:
        ard.write(b'1')
    if x > 600:
        ard.write(b'2')
    if y < 400:
        ard.write(b'3')
    if y > 600:
        ard.write(b'4')


def speech():
    while True:
        
      #  with sr.Microphone() as source:
       #     try:
                
        #        print("call jarvis")
         #       audio = r.listen(source)
                

          #      d = str(r.recognize_google(audio))
           #     d = d.lower()
            #    print(d)
          #  except Exception:
           #     d = ''
            #    pass
        d = str(input('>>>'))
        if 'jarvis' in d:

            d = d
            #d = str(input(">>>>"))
            #L = ['hello','hi','who are you','you good','search keyword hulk','analyze the element']
            #d = random.choice(L)
            
            d = analyze(d,all_words)

            d = torch.tensor(d,dtype = torch.float32)

            out = model(d)
            #Old method
            #val = round(model(d).item())

            _,val = torch.max(out,dim = 1)
      
        
            for intent in intents['intents']:
                if tags[val.item()] == intent['tag']:
                    print("Jarvis:",random.choice(intent['responses']))
                    engine.say(random.choice(intent['responses']))
                    engine.runAndWait()
                               

model.eval()
#speech()                  

def vision():

    while True:
        global c
        x = c
        ret,frame = cam.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([105,140,20])
        upper_blue = np.array([125,255,255])
        mask = cv2.inRange(hsv,lower_blue,upper_blue)
        contours,r = cv2.findContours(mask ,cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            
            if cv2.contourArea(cnt) > 90:
                #print("true")
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 3)
        #ard_data()
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('image',frame)

        

cam  =cv2.VideoCapture(1)

Thread(target = vision).start()
#Thread(target = ard_data).start()
Thread(target = speech).start()



while True:
    
    ret,frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([105,140,20])
    upper_blue = np.array([125,255,255])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    contours,r = cv2.findContours(mask ,cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 90:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 3)

    
    if cv2.waitKey(1) == ord('q'):
        break
    cv2.imshow('image',frame)

    Thread(target = speech).start()


#cam.release()
#cv2.destroyAllWindows()
'''          




