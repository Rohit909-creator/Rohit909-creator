#neural network for the natural language process data I prepared
#I think using feedforward nn is right for this data

import torch
import torch.nn as nn

X = torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0], #1
                  [0,1,1,1,1,0,0,0,0,0,0,0], #2
                  [1,0,1,0,0,0,0,0,0,0,0,0], #3
                  [0,0,1,0,0,0,0,0,0,0,0,0], #4
                  [0,0,1,1,1,0,0,0,0,0,0,0], #5
                  [0,0,0,0,0,1,1,1,0,0,0,0], #6
                  [0,0,0,0,0,0,0,0,1,1,1,1], #7
                  [0,0,0,0,0,0,0,0,1,1,1,0]  #8
                  ],dtype = torch.float32)

Y =  torch.tensor([[1],[9],[9],[9],[9],[2],[3],[3]],dtype = torch.float32)

print(X.shape)
print(Y.shape)

#w = torch.tensor([[0],[0],[0]],dtype = torch.float32,requires_grad = True)

n_sample, n_features = X.shape

Input = n_sample
Output = n_features

model = nn.Linear(12,1)

lr = 0.01

loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr = lr)

for i in range(4889):
    

    y_cap = model(X)

    l = loss(Y,y_cap)


    l.backward()

    optimizer.step()

    optimizer.zero_grad
   

    
print(model(X))

print(model(torch.tensor([[0,0,0,0,0,0,0,0,1,1,1,1]],dtype = torch.float32)))

