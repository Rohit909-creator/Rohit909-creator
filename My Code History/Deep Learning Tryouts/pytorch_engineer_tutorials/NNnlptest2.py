import torch
import torch.nn as nn

input_size = 10
hidden_size = 50
num_classes = 1
lr = 0.01

class NeuralNet(nn.Module):

    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size,bias = True)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes ,bias = True)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size,hidden_size,num_classes)


X = torch.tensor([[1,0,0,0,0,0,0,0,0,0], #1
                  [0,1,1,1,1,0,0,0,0,0], #2
                  [1,0,1,0,0,0,0,0,0,0], #3
                  [0,0,1,0,0,0,0,0,0,0], #4
                  [0,0,1,1,0,0,1,0,0,0], #5
                  [0,0,0,0,1,1,1,0,0,0], #6
                  [0,0,0,0,0,0,1,1,1,1], #7
                  [0,0,0,0,0,0,0,0,0,1]  #8
                  ],dtype = torch.float32)

Y =  torch.tensor([[2],[2],[2],[2],[2],[3],[4],[4]],dtype = torch.float32)




loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr = lr)



for i in range(600):
    y_pred = model(X)

    l = loss(Y,y_pred)

    

    l.backward()
    

    optimizer.step()

    optimizer.zero_grad

print(model(X))

    
    

        
        
    
