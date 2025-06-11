import torch
import torch.nn as nn

X = torch.tensor([[1,0,0],
                  [0,1,1],
                  [1,0,1],
                  [0,0,1]
                  ],dtype = torch.float32)

Y =  torch.tensor([[1],[0],[1],[0]],dtype = torch.float32)

print(X.shape)
print(Y.shape)

#w = torch.tensor([[0],[0],[0]],dtype = torch.float32,requires_grad = True)

n_sample, n_features = X.shape

Input = n_sample
Output = n_features

model = nn.Linear(3,1)

lr = 0.01

loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr = lr)

for i in range(90):
    

    y_cap = model(X) 

    l = loss(Y,y_cap)
    print(l)


    l.backward()

    optimizer.step()

    optimizer.zero_grad
   

    
print(torch.sigmoid(model(X)))

print(torch.sigmoid(model(torch.tensor([[1,1,1]],dtype = torch.float32))))
print("&&&&&&&&&")
print(torch.sigmoid(model(torch.tensor([[0,1,1]],dtype = torch.float32))))
print(torch.sigmoid(model(torch.tensor([[0,0,1]],dtype = torch.float32))))
        
print(torch.sigmoid(model(torch.tensor([[1,1,1]],dtype = torch.float32))))






                  
