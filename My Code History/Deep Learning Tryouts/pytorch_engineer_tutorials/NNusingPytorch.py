import torch
import torch.nn as nn
import numpy as np

x_train = torch.tensor([[0,0,1],
                        [0,1,0]
                        ],dtype = torch.float32)

x_label = torch.tensor([[1],[2]],dtype = torch.float32)


class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.linear = nn.Linear(3,6)
        self.linear2 = nn.Linear(6,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        y_pred = self.linear2(self.relu(self.linear(x)))
        return y_pred

model = Model()


loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#TRAINING


for i in range(70):

    y_pred = model(x_train)

    loss = loss_func(y_pred,x_label)
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print("resulr is ",model(x_train))
'''
while True:

    n1 = int(input(">>>"))
    

    tst = torch.Tensor([[n1]])
    print(model(tst).data[0][0].item())

'''
