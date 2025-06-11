import torch
x = torch.tensor([2,3,4,5],dtype = torch.float32)
y = torch.tensor([4,6,8,10],dtype = torch.float32)


w = torch.tensor(0.0,dtype = torch.float32,requires_grad = True)

learning_rate = 0.01



for i in range(1000):
    
    #Forward pass
    y_cap = w*x


    loss = ((y_cap - y)**2).mean()

    #backward pass
    loss.backward()

    
    with torch.no_grad():
        w -= learning_rate*w.grad.item()
        #print(w)

    w.grad.zero_()

print(x*w)





