#test for derewarding function in progress
#should I use a Neural Network to recognise the base angle
#and other angles 

import numpy as np
import random

#The array is three dimensional because I have to find three angles
q_table = np.zeros((50,90,3))


state = 0
action = 0
action2 = 0
new_state = 0
reward = 0
p = 0
q = 0
u = 0
v = 0
for i in range(10):
    state = 0
    for j in range(20000):
        #state = random.choice(np.arange(0,2))
        
        action = random.choice(np.arange(1,90))
        action2 = random.choice(np.arange(1,90))
        action3 = random.choice(np.arange(1,90))

        X =9.*np.cos(action) + 7.5*np.cos(action + action2 + action3)
        Y = 9.*np.sin(action) + 7.5*np.sin(action + action2 + action3)
        state = int(2*X + Y)
        new_state = state
        #print(X)
        #print(Y)
        
        #val = int(input(">>>"))
        
        if X == 11.993348870114282 and Y == 10.232023875033168:
            
            reward = 1
            print("yay we got there")
            print(state)
            print(action)
            print(action2)
            print(action3)
            print(X)
            print(Y)
            p = state
            q = action
            
        else:
            reward = 0

        
        #Sir Richard Bellman's equation:
        #for rewarding the agent
        q_table[state][action][0] =  (1-0.1)*q_table[state][action][0] + 0.1*(reward + 0.99*np.max(q_table[state,:,0]))
        q_table[state][action2][1] =  (1-0.1)*q_table[state][action2][1] + 0.1*(reward + 0.99*np.max(q_table[state,:,1]))
        q_table[state][action3][2] =  (1-0.1)*q_table[state][action3][2] + 0.1*(reward + 0.99*np.max(q_table[state,:,2]))
        
        



print(q_table[p][q][0])
print(np.max(q_table[p,:,0]))
print(np.argmax(q_table[p,:,0]))
print(np.argmax(q_table[34,:,0]))
print(np.max(q_table[34,:,0]))
print("&&&&&&")
print(np.argmax(q_table[34,:,1]))
print(np.max(q_table[34,:,1]))
print("$$$$$$$$$$")
print(np.argmax(q_table[34,:,2]))
print(np.max(q_table[34,:,2]))
print("################")


I2 = np.argmax(q_table[34,:,1])
I3 = np.argmax(q_table[34,:,2])

#dereward:

q_table[34][I2][1] = -1*q_table[34][I2][1]
q_table[34][I3][2] = -1*q_table[34][I3][2]
print("################")
print("################")
print("################")
print("################")
print("################")
print("################")
# Just an attempt to find the other real solutions from the Qtable
'''
for i in range(100):
    X_ = 9.*np.cos(np.argmax(q_table[34,:,0])) + 7.5*np.cos(np.argmax(q_table[34,:,0]) + np.argmax(q_table[34,:,1]) + np.argmax(q_table[34,:,2]))
    Y_ = 9.*np.sin(np.argmax(q_table[34,:,0])) + 7.5*np.sin(np.argmax(q_table[34,:,0]) + np.argmax(q_table[34,:,1]) + np.argmax(q_table[34,:,2]))

    D = np.argmax(q_table[34,:,1])
    J = np.argmax(q_table[34,:,2])

    q_table[34][D][1] = -1*q_table[34][D][1]
    q_table[34][J][2] = -1*q_table[34][J][2]
    
    if X_ == 11.993348870114282 and Y_ == 10.232023875033168:
        print("true")
        break

print(np.argmax(q_table[34,:,0]))
print(np.max(q_table[34,:,0]))
print("&&&&&&")
print(np.argmax(q_table[34,:,1]))
print(np.max(q_table[34,:,1]))
print("$$$$$$$$$$")
print(np.argmax(q_table[34,:,2]))
print(np.max(q_table[34,:,2]))
print("################")

'''

#next attempt which is based on the observation that the total angle
#will be a constant for this coordinate its 89
#since the above derewarding function is not that much efficient:
   
D = np.argmax(q_table[34,:,1])

#dereward:
q_table[34][D][1] = -1*q_table[34][D][1]

third_angle = 89 - (np.argmax(q_table[34,:,0]) + np.argmax(q_table[34,:,1]))    
    
print(np.argmax(q_table[34,:,0]))
print(np.max(q_table[34,:,0]))
print("&&&&&&")
print(np.argmax(q_table[34,:,1]))
print(np.max(q_table[34,:,1]))
print("$$$$$$$$$$")
print(third_angle)
print("################")



