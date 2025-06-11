#NeuralNetwork Pathway for Integrated Information proccessing for an Advanced AI
#upgraded version will be available on colab soon
import torch
import torch.nn as nn
import copy

class NN_Pathway(nn.Module):

    def __init__(self,input_size,hidden_size,sfc_size,output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size,hidden_size,bias = True)
        self.sfc1 = nn.Linear(hidden_size,sfc_size,bias = True)
        self.sfc2 = nn.Linear(hidden_size,sfc_size,bias = True)
        self.out = nn.Linear(sfc_size,output_size,bias  = True)
        self.relu = nn.ReLU()

    def forward(self,X):

        out = self.relu(self.fc1(X))
        out1 = self.relu(self.sfc1(out))
        out2 = self.relu(self.sfc2(out))
        val1 = torch.sum(out1)
        val2 = torch.sum(out2)

        out = self.out(out1)
        #if val1 > val2:
        #    out = self.out(out1)
        #elif val2 < val1:
        #    out = self.out(out2)

        return out,val1,val2,out1,out2

class NN_Pathway2(nn.Module):

    def __init__(self,input_size,hidden_size,sfc_size,output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size,hidden_size,bias = True)
        self.sfc1 = nn.Linear(hidden_size,sfc_size,bias = True)
        self.sfc2 = nn.Linear(hidden_size,sfc_size,bias = True)
        self.out = nn.Linear(sfc_size,output_size,bias  = True)
        self.relu = nn.ReLU()

    def forward(self,X):

        out = self.relu(self.fc1(X))
        out1 = self.relu(self.sfc1(out))
        out2 = self.relu(self.sfc2(out))
        val1 = torch.sum(out1)
        val2 = torch.sum(out2)

        out = self.out(out2)
        #if val1 > val2:
        #    out = self.out(out1)
        #elif val2 < val1:
        #    out = self.out(out2)

        return out,val1,val2,out1,out2

X = torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype = torch.float32)

Y = torch.tensor([[0],[0],[0]],dtype = torch.float32)

X_new = torch.tensor([[1,1,0],[0,1,1],[1,0,1]],dtype = torch.float32)
Y_new = torch.tensor([[1],[1],[1]],dtype = torch.float32)

model = NN_Pathway(3,100,256,1)
model2 = NN_Pathway2(3,100,256,1)
model.fc1.requires_grad_ = False

#training on X:
model.sfc2.requires_grad_ = False

optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)
loss = nn.MSELoss()

for epoch in range(10):
    out,val1,val2,out1,out2 = model(X)
    l = loss(out,Y)
    if epoch%5 == 0:
        print(f'Epoch{epoch} Loss:{l}')
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

out,val1,val2,out1,out2 = model(X)
print(f'Output from 1st dataset:{out}')

model2 = copy.deepcopy(model.state_dict())
model2.fc1.requires_grad_ = True
model2.sfc1.requires_grad_ = False
model2.sfc2.requires_grad_ = True
optimizer = torch.optim.SGD(model2.parameters(),lr =0.1)

#training on X_new:

for i in range(10):
    out,val1,val2,out1,out2 = model2(X_new)
    l = loss(out,Y_new)
    if epoch%5 == 0:
        print(f'Epoch{epoch} Loss:{l}')
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

out,val1,val2,out1,out2 = model2(X_new)

print(f'Output from 2nd dataset:{out}')

model2.sfc1.requires_grad_ = True
model1.eval()
model2.eval()


