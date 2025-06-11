import torch
import torch.nn as nn

x = torch.tensor([[5],[2],[3],[4]],dtype = torch.float32)
y = torch.tensor([[10],[4],[6],[8]],dtype = torch.float32)

w = torch.tensor(0.0,dtype = torch.float32,requires_grad = True)

n_sample, n_features = x.shape

Input = n_features
Output = n_features

model = nn.Linear(Input,Output)





learning_rate = 0.01

loss = nn.MSELoss()


optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

          
          
for i in range(95):
    
    #Forward pass
    y_cap = model(x)


    l = loss(y,y_cap)
          
    #backward pass
    l.backward()
    optimizer.step()

    optimizer.zero_grad
    
print(model(torch.tensor([5],dtype = torch.float32)))
      
