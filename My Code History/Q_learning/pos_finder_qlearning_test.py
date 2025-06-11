#pos_finding_rl
#angle incrementation test
import numpy as np
import random

q_table = np.zeros((50,3,3))


state = 0
action = 0
action2 = 0
action3 = 0
new_state = 0
target = 11.993348870114282 + 10.232023875033168
diff = 0

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.0001

for i in range(100):
    for j in range(10000):


        exploration_threshold = random.uniform(0,1)

        if exploration_threshold > exploration_rate:
            a1 = np.argmax(q_table[state,:,0])
            a2 = np.argmax(q_table[state,:,1])
            a3 = np.argmax(q_table[state,:,2])
        else:

            a1 = random.randint(-1,1)
            a2 = random.randint(-1,1)
            a3 = random.randint(-1,1)

        action = 1 + a1
        action2 = 1 + a2
        action3 = 1 + a3

        X = 9.*np.cos(action) + 7.5*np.cos(action + action2 + action3)
        Y = 9*np.sin(action) + 7.5*np.sin(action + action2 + action3)
        
        statef = X + Y

        difference = target - statef

        if difference < diff:
            reward = 1

        else:
            reward = 0
        


        diff = difference

        state = int(2*X + Y)

        q_table[state][a2][1] = (1-0.1)*q_table[state][a2][1] + 0.1*(reward + 0.99*np.argmax(q_table[new_state,:,1]))
        q_table[state][a1][0] = (1-0.1)*q_table[state][a1][0] + 0.1*(reward + 0.99*np.argmax(q_table[new_state,:,0]))
        q_table[state][a3][2] = (1-0.1)*q_table[state][a3][2] + 0.1*(reward + 0.99*np.argmax(q_table[new_state,:,2]))

        exploration_rate = min_exploration_rate + \
                           (max_exploration_rate - min_exploration_rate)*np.exp( -exploration_decay_rate*i)

        
        new_state = state

        
        
