import torch
import torch.nn as nn

X = torch.tensor([[13,53],
                  [13,65],
                  [13,58],
                  [13,50]]
                  ,dtype = torch.float32)

Y =  torch.tensor([[23],[11],[18],[26]],dtype = torch.float32)

print(X.shape)
print(Y.shape)

#w = torch.tensor([[0],[0],[0]],dtype = torch.float32,requires_grad = True)

n_sample, n_features = X.shape

Input = n_sample
Output = n_features

model = nn.Linear(2,1)

lr = 0.01

loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr = lr)

for i in range(1000):
    

    y_cap = model(X) 

    l = loss(Y,y_cap)


    l.backward()

    optimizer.step()

    optimizer.zero_grad
   

    
print(torch.sigmoid(model(X)))
'''
print(torch.sigmoid(model(torch.tensor([[1,1,1]],dtype = torch.float32))))
print("&&&&&&&&&")
print(torch.sigmoid(model(torch.tensor([[0,1,1]],dtype = torch.float32))))
print(torch.sigmoid(model(torch.tensor([[0,0,1]],dtype = torch.float32))))
        
print(torch.sigmoid(model(torch.tensor([[0,0,0]],dtype = torch.float32))))

'''




                  
