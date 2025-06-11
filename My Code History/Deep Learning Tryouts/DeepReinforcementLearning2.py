#still have to work on the code
#experimenting with the hyperparameters in this AI's NeuralNet and also the training pipeline

import torch
import torch.nn as nn
from collections import namedtuple
import random
import turtle
import numpy as np






    
p = turtle.Turtle(shape = 'circle')
p.color('blue')
p.penup()
r = turtle.Turtle(shape = 'circle')
p.color('red')
r.penup()
r.goto(20,20)

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.00009
gamma = 0.1
d = 0
states = []
actions = []
rewards = []
next_states = []
reward_data = []
training_data = []


def state_function():
    L1 = [0]*7
    if int(p.xcor()) <= int(r.xcor()+20) and int(p.xcor()) >= int(r.xcor() - 20):
        L1[0] = 1
        
    if int(p.ycor()) <= int(r.ycor()+20) and int(p.ycor()) >= int(r.ycor() - 20):
        L1[1] = 1

    if int(p.ycor()) < int(r.ycor()):
        L1[2] = 1

    if int(p.ycor()) > int(r.ycor()):
        L1[3] = 1

    if int(p.xcor()) < int(r.xcor()):
        L1[4] = 1

    if int(p.xcor()) > int(r.xcor()):
        L1[5] = 1

    return L1

'''

def state_function():
    
    
    L1 = [0]*4
    if int(p.xcor()) <= int(r.xcor()+18) and int(p.xcor()) >= int(r.xcor() - 18) or int(p.ycor()) < int(r.ycor()):
        L1[0] = 1

    if int(p.xcor()) <= int(r.xcor()+18) and int(p.xcor()) >= int(r.xcor() - 18) or int(p.ycor()) > int(r.ycor()):
        L1[1] = 1
        
    if int(p.ycor()) <= int(r.ycor()+18) and int(p.ycor()) >= int(r.ycor() - 18) or int(p.xcor()) > int(r.ycor()):
        L1[2] = 1

    if int(p.ycor()) <= int(r.ycor()+18) and int(p.ycor()) >= int(r.ycor() - 18) or int(p.xcor()) < int(r.xcor()):
        L1[3] = 1

    return L1

'''

def do_action(inp):

    if inp == 0:
        p.setx(p.xcor() + 1)
    elif inp == 1:
        p.setx(p.xcor() - 1)
    elif inp == 2:
        p.sety(p.ycor() - 1)
    elif inp == 3:
        p.sety(p.ycor() + 1)

    


class DQN(nn.Module):

    def __init__(self,input_size,output_size,hidden_size):
        super(DQN,self).__init__()

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)
        self.tanh = nn.Tanh()

    
    def forward(self,X):

        out = self.tanh(self.fc1(X))
        out = self.tanh(self.fc2(out))
        out = self.fc3(out)
        return out



class replay_memory():

    def __init__(self,capacity):

        self.experience = namedtuple('memory',('action','state','next_state','reward'))
        self.replay_memory = []
        self.capacity = capacity
        train_data = []

    def get_memory(self,state,action,next_state,reward):
        
        e = self.experience(action,state,next_state,reward)
        if len(self.replay_memory) < self.capacity:
            self.replay_memory.append(e)

        else:
            self.replay_memory.pop(-1)
            self.replay_memory.insert(0,e)

    def sample_memory(self,batch_size):
        return random.sample(self.replay_memory,batch_size)
    def memory_length(self):
        return len(self.replay_memory)
    def print_memory(self):
        return self.replay_memory


train_data = []
next_state_data = []
at_data = []
state_data = []
action_data = []
next_data = []
memory = replay_memory(10)
model = DQN(7,4,100)
#model = torch.load("Self_learned_AI_5.0.pth")
loss = nn.MSELoss()
#or use
#loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)
done = False
#model.eval()


#training loop
for epochs in range(100):
    print('epoch:',epochs)
    r.goto(random.randint(40,80),random.randint(40,80))
    X1= np.arange(35,int(r.xcor()) - 20,1)
    X2 = np.arange(int(r.xcor()) + 20,86,1)
    Y1 = np.arange(35,int(r.ycor()) - 20,1)
    Y2 = np.arange(int(r.ycor()) + 20,86,1)
    

    X1 = X1.tolist()
    X2 = X2.tolist()
    Y1 = Y1.tolist()
    Y2 = Y2.tolist()
    X1.extend(X2)
    Y1.extend(Y2)
    
    
    p.goto(random.choice(X1),random.choice(Y1))

    
    
    if done == True:
        d += 1
        print(d)
        memory.get_memory(state,action,new_state,reward)

    states.clear()
    actions.clear()
    next_states.clear()

    
    for steps in range(500):

        value = np.sqrt(((p.xcor() - r.xcor())**2 + (p.ycor() - r.ycor())**2))
        
        #print(int(value))
        
        #if int(p.xcor()) == int(r.xcor()) and int(p.ycor()) == int(r.ycor()):
         #   reward = 10
          #  done = True
        
        
        #print(reward)
        L2 = [0]*4
        
        state = state_function()
        #print(state)
        state_data = torch.tensor([state],dtype = torch.float32)
        exploration_threshold = random.uniform(0,1)
        
        if exploration_threshold > exploration_rate:
            
            print('exploit')
            action = torch.argmax(model(state_data)).item()
        else:
            #print('er')
            action = random.choice([0,1,2,3])

        if epochs < 10:
            action = int(input(">>>>"))

        #perform action
        do_action(action)
        
        L2[action] = 1
        action_data = torch.tensor(L2)
        new_state = state_function()
        states.append(state)
        actions.append(action)
        
        next_states.append(new_state)
        
        if int(value) < 20:
            reward = 0.2
            done = True

        else:
            reward = 0
            done = False
        
        
        if steps%100 == 0:

            if int(memory.memory_length()) > 1:
                reward = 0.2
                print(memory.memory_length())
                print("reward is:",reward)
            
                if int(memory.memory_length()) < 5:
                    mem = memory.sample_memory(memory.memory_length())
                else:
                    mem = memory.sample_memory(5)
                            
                for j in range(len(mem)):
                    training_data.append(mem[j].state)
                    next_state_data.append(mem[j].next_state)
                    at_data.append(mem[j].action)
                #print(at_data)

                train_data = torch.tensor(training_data,dtype = torch.float32)
                n_state = torch.tensor(next_state_data,dtype = torch.float32)
                pred = model(train_data)
                x,y = pred.shape
                target = torch.zeros(x,y)
                action_index = torch.tensor(at_data,dtype = torch.float32)
                #target = action_index.int()
                Q_value_new = reward +(gamma*torch.max(model(n_state)))
                for i in range(len(target)):
                        
                    target[i][int(action_index[i].item())] = Q_value_new
                    #target[int(action_index[i].item()) - 1] = Q_value_new
                #model.train()
                
                l = loss(pred,target)
                print(l)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                print(at_data)    
                training_data.clear()
                next_state_data.clear()
                at_data.clear()
                rewards.clear()

        
        rewards.append(reward)
        
        if done == True:
            break
    if epochs > 20:
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*(epochs - 20))
    

n1 =int(input('Training Command y/n:'))


model.eval()        

print("Trained A.I Model Testing")

#testing the trained A.I:
#i will have to train him for a couple more hours with google's gpu or tpu
for i in range(15):
    p.goto(random.randint(10,100),random.randint(10,100))
    r.goto(random.randint(40,80),random.randint(40,80))
    
    for j in range(500):
        done = False
        
        value = np.sqrt(((p.xcor() - r.xcor())**2 + (p.ycor() - r.ycor())**2))
        
        #print(int(value))
        if int(value) < 20:
            done = True
            break
        

        reward = 10
        
        
        state = state_function()

        test_data = torch.tensor([state],dtype = torch.float32)

        pred = torch.argmax(model(test_data)).item()
        print(pred)
        do_action(pred)

        

         
        

        if done == True:
            break



n1 = int(input('>>>>'))
        
if n1 == 1:
        torch.save(model,'Self_learned_AI_6_0.pth')
        print('Model saved as Self_learned_AI_6_0.pth ')
else:
    print('Model not saved')


    

    
        
