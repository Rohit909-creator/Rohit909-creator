import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random

#simply just
X = torch.tensor([[[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1],
                  [0,0,0,0]]],dtype = torch.float32)



print(X.shape)
print(X[0][0])
X.reshape(5,1,4)
#print(X[0][0][0][0].shape)
#print(X[0][0][0][0])

Y = torch.tensor([0,1,2,3,4])
print(Y.shape)
class RNN(nn.Module):

    def __init__(self):
        super(RNN,self).__init__()

        self.rnn = nn.GRU(4,45,1,batch_first = True)
        self.linear = nn.Linear(45,20)
        self.linear2 = nn.Linear(20,5)
        self.relu = nn.ReLU()

    def forward(self,X):

        hi = torch.zeros(1,1,45)
        out,hidden  = self.rnn(X,hi)
        #print(out.shape)
        out = out[-1,:,:]
        #print(out.shape)
        out = self.relu(self.linear(out))
        out = self.linear2(out)
        #out = out.reshape(-1,1)

        return out,hidden


model = RNN()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

k = 0
#training loop
for i in range(1500):

    out,hidden = model(X)
    
    l = loss(out,Y)
    if i%100 == 0:
        print(l)
    optimizer.zero_grad()
    l.backward()
    
    optimizer.step()

print(hidden)

print(model(X))

print(model(torch.tensor([[[1,0,0,0]]],dtype = torch.float32)))
      

