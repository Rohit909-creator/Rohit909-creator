import torch
import torch.nn as nn
import numpy as np
from nltk.tokenize import word_tokenize



all_char = ['what is',
            'is going',
            'going on',
            'how are',
            'are you']

targets = ['is going',
           'going on',
           'how are',
           'are you',
           'feeling']



def char_to_tensor(string):
    word = [0]*len(all_char)
    for i,j in enumerate(all_char):        
        if string == j:
            word[i] = 1
    return word

#print(char_to_tensor('is going'))


words = []
for k,l in enumerate(all_char):
    words.append(char_to_tensor(l))
    
train_data = torch.tensor(words)
words.pop(0)
words.insert(len(all_char),[0,0,0,0,0])

target = words

#print(train_data)    
#print(train_data.shape)

target_data = torch.tensor([0,1,2,3,4])
print('targets:',target_data)
print(len(targets))
print(target_data.shape)

    

class RNN(nn.Module):

    def __init__(self):
        super(RNN,self).__init__()

        self.embed = nn.Embedding(len(all_char),10)
        self.lstm = nn.LSTM(10,10,1,batch_first = True)
        self.fc = nn.Linear(50,len(targets))

    def forward(self,X):

        h = torch.zeros(1, 5, 10)
        c = torch.zeros(1, 5, 10)

        out = self.embed(X)
        out,(h,c) = self.lstm(out,(h,c))
        #print(out.shape)
        out = out.reshape(-1,50)
        #print(out.shape)
        out = self.fc(out)
        #print(out.shape)

        return out


model = RNN()
loss =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.03)


#training:

for i in range(20):

    out = model(train_data)
    l = loss(out,target_data)

    if i%10 == 0:
        print(l)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

print(model(train_data))

data = []
while True:
    d = str(input('>>'))

    for k in all_char:
        data.append(char_to_tensor(d))

    d = torch.tensor(data)

    out = model(d)

    val = torch.argmax(out).item()
    print(val)

    print(targets[val])

    data.clear()
