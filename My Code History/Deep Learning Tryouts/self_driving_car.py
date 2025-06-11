import turtle
import numpy as np
import random
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import namedtuple



class Env(turtle.Turtle):

    def __init__(self,color = 'red'):
        super().__init__()

        self.p = turtle.Turtle()
        self.p.color('red')

    def reset():
        p.color('blue')
        p.penup()
        p.goto(0,0)
        p.pendown()
        p.goto(0,200)
        p.penup()
        p.goto(20,200)
        p.pendown()
        p.goto(20,0)
        p.goto(10,0)
        p.color('red')

    def action_space:
        return 'Env object'
    
    
    def observation_space:
        return [0,0]

    
    def step(action):
        if action == 0:
            self.p.setx(p.xcor()+1)
        if action == 0:
            self.p.setx(p.xcor()-1)    


    def state():
        d1 = math.sqrt((0 - p.xcor())**2 + (p.ycor() - p.ycor())**2)
        d2 = math.sqrt((20 - p.xcor())**2 + (p.ycor() - p.ycor())**2)
        
        return torch.tensor([d1,d2],dtype = torch.float32)


env = Env(color = 'red')





    
done = False
num_epochs = 300

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.003
gamma = 0.1



#time.sleep(0.1)

#Neural Network Class

class DQN(nn.Module):

    def __init__(self,input_size,output_size,hidden_size):
        super(DQN,self).__init__()

        self.fc1 = nn.Linear(input_size,hidden_size,bias = True)
        self.fc2 = nn.Linear(hidden_size,hidden_size,bias = True)
        self.fc3 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()

    
    def forward(self,X):

        out = self.relu(self.fc1(X))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

#Replay Memory Class
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


model = DQN(len(env.reset()),env.action_space.n,256)

optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
loss = nn.MSELoss()

memory = replay_memory(300)


for epochs in range(num_epochs):
    try:
            
        state = env.reset()
            
        for ts in range(200):

            exploration_threshold = random.uniform(0,1)

            if exploration_threshold > exploration_rate:
                #print('exploit')
                state_data = torch.tensor(state,dtype = torch.float32)
                action = torch.argmax(model(state_data)).item()
            else:
                #print('er')
                action = random.choice([0,1])

        
            # time.sleep(0.1)
            n_state,reward,done,info = env.step(action)
            print(reward)
            r_s += reward
            # Training on every 100 epochs
            if ts%100 == 0:
                if memory.memory_length() > 2:
                    if memory.memory_length() < 10:
                        data = memory.sample_memory(memory.memory_length())
                    else:                
                        data = memory.sample_memory(10)
                    rewards = []
                    actions = []
                    states = []
                    n_states = []
                    
                    #unpacking:

                    for i in data:
                        rewards.append(i.reward)
                        actions.append(i.action)
                        states.append(i.state)
                        n_states.append(i.next_state)
                
                    
                    
                    n_states = torch.tensor(n_states,dtype = torch.float32)
                    t_states = torch.tensor(states,dtype = torch.float32)
                    rewards = torch.tensor(rewards,dtype = torch.float32)
                    action_index = torch.tensor(actions,dtype = torch.float32)
                
                    pred = model(t_states)
                    target = pred.clone()
                    for i in range(target.shape[0]):
                        
                        Q_value_new = rewards[i] +(gamma*torch.max(model(n_states[i])))    
                        target[i][int(action_index[i].item())] = Q_value_new


                    optimizer.zero_grad()
                    l = loss(pred,target)
                    print(f'Epoch{epochs} Time Step:{ts} Loss:{l}')
                    
                    
                    l.backward()
                    optimizer.step()

            state = n_state

            if epochs >=  0:
                exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*epochs)
        
                
                
            if done == True:
                #print(done)
                if reward > 0:
                    memory.get_memory(state,action,n_state,reward)
                    
                break
            
            env.render()

    except KeyboardInterrupt:
        break



model = torch.load("CartPole_AI.pth")

model.eval()

for epoch in range(num_epochs):
    state = env.reset()
    for ts in range(200):
        state_t = torch.tensor([state],dtype = torch.float32)
        action = model(state_t)
        env.step(torch.argmax(action).item())
        env.render()





print('Do you want to save the current model(0/1):')
inp = int(input(">>>"))
file_name = "CartPole_AI.pth"
if inp == 1:
    torch.save(model,file_name)
    print(f"Model saved as:{file_name}")



